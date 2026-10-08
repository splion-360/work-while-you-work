import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from service.reconciliation import reconcile
from service.scorer import score_input_fingerprint
from service.scoring_state import ScoringStore


def notion_page(key):
    return {"properties": {"Application key": {"title": [{"plain_text": key}]}}}


class FakeNotion:
    def __init__(self, applications, scores=()):
        self.applications = [notion_page(key) for key in applications]
        self.scores = [notion_page(key) for key in scores]

    def query_data_source(self, source_id, _filter=None):
        return self.applications if source_id == "applications" else self.scores


class ReconciliationTests(unittest.TestCase):
    def make_input(self, directory, key="app-1"):
        resume = Path(directory) / f"{key}.pdf"
        resume.write_bytes(b"resume")
        return {
            "application_key": key,
            "canonical_url": "https://example.com/jobs/1",
            "source_job_id": "1",
            "job_description": "Build ML systems.",
            "job_description_hash": "jd-1",
            "description_provenance": "original_saved",
            "captured_at": "2026-09-02T20:00:00Z",
            "extraction_quality": "complete",
            "resume_version": "professional/resume-v1",
            "resume_path": str(resume),
            "resume_content_hash": "a83a31320d921b888a48fa5edd0b4b5a29984de6e96bf7b8ac7d29ba06caf616",
        }

    def test_activates_waiting_and_recreates_missing_jobs_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            first = self.make_input(directory, "app-1")
            second = self.make_input(directory, "app-2")
            store.save_scoring_input(first)
            store.save_scoring_input(second)
            first_fingerprint = score_input_fingerprint(first, "v1", "model-v1")
            store.enqueue_score_job("app-1", first_fingerprint, status="waiting_application")

            report = reconcile(store, FakeNotion(["app-1", "app-2"]), "applications", "scores", "v1", "model-v1")
            second_report = reconcile(store, FakeNotion(["app-1", "app-2"]), "applications", "scores", "v1", "model-v1")
            jobs = store.list_score_jobs()

        self.assertEqual(report["activated"], 1)
        self.assertEqual(report["requeued"], 1)
        self.assertEqual(second_report["requeued"], 0)
        self.assertEqual([job["status"] for job in jobs], ["queued", "queued"])

    def test_can_reconcile_without_backfilling_historical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            scoring_input = self.make_input(directory)
            store.save_scoring_input(scoring_input)

            report = reconcile(
                store,
                FakeNotion(["app-1"]),
                "applications",
                "scores",
                "new-v2",
                "new-model",
                enqueue_missing_scores=False,
            )
            jobs = store.list_score_jobs()

        self.assertEqual(report["requeued"], 0)
        self.assertEqual(jobs, [])

    def test_audits_duplicates_orphans_expired_leases_and_unavailable_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            scoring_input = self.make_input(directory)
            store.save_scoring_input(scoring_input)
            fingerprint = score_input_fingerprint(scoring_input, "v1", "model-v1")
            store.enqueue_score_job("app-1", fingerprint, status="queued")
            store.claim_score_job("worker-1")
            with closing(store.connect()) as connection, connection:
                connection.execute("UPDATE score_jobs SET lease_until = '2000-01-01 00:00:00'")
            Path(scoring_input["resume_path"]).unlink()

            report = reconcile(
                store,
                FakeNotion(["app-1", "app-1", "no-input"], ["app-1", "app-1", "orphan"]),
                "applications",
                "scores",
                "v1",
                "model-v1",
            )
            job = store.list_score_jobs()[0]

        self.assertEqual(report["duplicate_application_keys"], 1)
        self.assertEqual(report["duplicate_score_keys"], 1)
        self.assertEqual(report["orphan_score_keys"], 1)
        self.assertEqual(report["missing_local_inputs"], 1)
        self.assertEqual(report["recovered_leases"], 1)
        self.assertEqual(job["status"], "queued")

    def test_expired_publication_is_completed_only_when_notion_has_the_score(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            scoring_input = self.make_input(directory)
            store.save_scoring_input(scoring_input)
            fingerprint = score_input_fingerprint(scoring_input, "v1", "model-v1")
            store.save_scoring_snapshot(fingerprint, scoring_input)
            store.enqueue_score_job("app-1", fingerprint, status="queued")
            job = store.claim_score_job("worker-1")
            store.begin_score_publication(job["id"], "worker-1", job["attempt_count"])
            with closing(store.connect()) as connection, connection:
                connection.execute("UPDATE score_jobs SET lease_until = '2000-01-01 00:00:00'")

            report = reconcile(
                store,
                FakeNotion(["app-1"], ["app-1"]),
                "applications",
                "scores",
                "v1",
                "model-v1",
            )
            recovered = store.list_score_jobs()[0]

        self.assertEqual(report["recovered_publications"], 1)
        self.assertEqual(report["uncertain_publications"], 0)
        self.assertEqual(recovered["status"], "completed")

    def test_missing_publication_is_retried_after_uncertainty_window(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            scoring_input = self.make_input(directory)
            store.save_scoring_input(scoring_input)
            fingerprint = score_input_fingerprint(scoring_input, "v1", "model-v1")
            store.save_scoring_snapshot(fingerprint, scoring_input)
            store.enqueue_score_job("app-1", fingerprint, status="queued")
            job = store.claim_score_job("worker-1")
            store.begin_score_publication(job["id"], "worker-1", job["attempt_count"])
            with closing(store.connect()) as connection, connection:
                connection.execute(
                    "UPDATE score_jobs SET lease_until = '2000-01-01', updated_at = '2000-01-01'"
                )

            report = reconcile(
                store,
                FakeNotion(["app-1"]),
                "applications",
                "scores",
                "v1",
                "model-v1",
                publication_uncertainty_seconds=900,
            )
            recovered = store.list_score_jobs()[0]

        self.assertEqual(report["retried_publications"], 1)
        self.assertEqual(report["uncertain_publications"], 0)
        self.assertEqual(recovered["status"], "queued")


if __name__ == "__main__":
    unittest.main()
