import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from service.scoring_state import ScoringStore
from service.worker import process_one_job


def input_record():
    return {
        "application_key": "app-1",
        "canonical_url": "https://example.com/jobs/1",
        "source_job_id": "1",
        "job_description": "Build ML systems.",
        "job_description_hash": "jd-1",
        "description_provenance": "original_saved",
        "captured_at": "2026-09-02T20:00:00Z",
        "extraction_quality": "complete",
        "resume_version": "professional/resume-v1",
        "resume_path": "/resumes/resume-v1.pdf",
        "resume_content_hash": "resume-1",
        "role_company_key": "company-role-1",
    }


class FakeScorer:
    def __init__(self):
        self.calls = 0
        self.scorer_version = "v1"
        self.embedding_model = "embedding-v1"
        self.fingerprint_revision = ""

    def score(self, scoring_input):
        self.calls += 1
        return {
            "application_key": scoring_input["application_key"],
            "input_fingerprint": "fingerprint-1",
            "scorer_version": "v1",
            "embedding_model": "embedding-v1",
            "score": 80.0,
            "band": "Strong",
            "matched_terms": ["Python"],
            "category_breakdown": {"skills": 80.0},
        }


class QueueTests(unittest.TestCase):
    def make_store(self, directory):
        store = ScoringStore(Path(directory) / "state.sqlite3")
        store.initialize()
        record = input_record()
        store.save_scoring_input(record)
        store.save_scoring_snapshot("fingerprint-1", record)
        return store

    def test_jobs_are_idempotent_and_claimed_with_a_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            first = store.enqueue_score_job("app-1", "fingerprint-1", status="queued")
            second = store.enqueue_score_job("app-1", "fingerprint-1", status="queued")
            claimed = store.claim_score_job("worker-1", lease_seconds=60)
            blocked = store.claim_score_job("worker-2", lease_seconds=60)

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(claimed["lease_owner"], "worker-1")
        self.assertEqual(claimed["attempt_count"], 1)
        self.assertIsNone(blocked)

    def test_expired_lease_can_be_reclaimed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.enqueue_score_job("app-1", "fingerprint-1", status="queued")
            store.claim_score_job("worker-1", lease_seconds=60)
            with closing(store.connect()) as connection, connection:
                connection.execute("UPDATE score_jobs SET lease_until = '2000-01-01 00:00:00'")
            reclaimed = store.claim_score_job("worker-2", lease_seconds=60)

        self.assertEqual(reclaimed["lease_owner"], "worker-2")
        self.assertEqual(reclaimed["attempt_count"], 2)

    def test_publication_fence_prevents_an_expired_worker_from_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.enqueue_score_job("app-1", "fingerprint-1", status="queued")
            first = store.claim_score_job("worker-1", lease_seconds=60)
            with closing(store.connect()) as connection, connection:
                connection.execute("UPDATE score_jobs SET lease_until = '2000-01-01 00:00:00'")
            second = store.claim_score_job("worker-2", lease_seconds=60)

            stale_can_publish = store.begin_score_publication(
                first["id"], "worker-1", first["attempt_count"], 60
            )
            current_can_publish = store.begin_score_publication(
                second["id"], "worker-2", second["attempt_count"], 60
            )
            third_claim = store.claim_score_job("worker-3", lease_seconds=60)

        self.assertFalse(stale_can_publish)
        self.assertTrue(current_can_publish)
        self.assertIsNone(third_claim)

    def test_worker_completes_a_queued_score(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.enqueue_score_job("app-1", "fingerprint-1", status="queued")
            persisted = []

            worked = process_one_job(
                store,
                FakeScorer(),
                lambda result: persisted.append(result) or "created",
                "worker-1",
            )
            jobs = store.list_score_jobs()

        self.assertTrue(worked)
        self.assertEqual(persisted[0]["application_key"], "app-1")
        self.assertEqual(jobs[0]["status"], "completed")

    def test_worker_reuses_a_preview_cache_result(self):
        from service.scorer import score_cache_key

        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.enqueue_score_job("app-1", "fingerprint-1", status="queued")
            scorer = FakeScorer()
            cache_key = score_cache_key(input_record(), "v1", "embedding-v1")
            store.save_cached_score(cache_key, {
                "scorer_version": "v1",
                "embedding_model": "embedding-v1",
                "score": 80.0,
                "band": "Strong",
                "matched_terms": ["Python"],
                "category_breakdown": {},
            })
            persisted = []

            process_one_job(
                store, scorer, lambda result: persisted.append(result), "worker-1"
            )

        self.assertEqual(scorer.calls, 0)
        self.assertEqual(persisted[0]["application_key"], "app-1")

    def test_worker_retries_then_marks_terminal_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.enqueue_score_job("app-1", "fingerprint-1", status="queued")

            for _ in range(3):
                with closing(store.connect()) as connection, connection:
                    connection.execute("UPDATE score_jobs SET next_run_at = CURRENT_TIMESTAMP")
                process_one_job(
                    store,
                    FakeScorer(),
                    lambda _result: (_ for _ in ()).throw(RuntimeError("Notion unavailable")),
                    "worker-1",
                    max_attempts=3,
                    retry_delay_seconds=0,
                )
            job = store.list_score_jobs()[0]

        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["attempt_count"], 3)
        self.assertIn("Notion unavailable", job["last_error"])

    def test_notion_retry_reuses_the_completed_local_score(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.enqueue_score_job("app-1", "fingerprint-1", status="queued")
            scorer = FakeScorer()
            persistence_calls = 0

            def persist(_result):
                nonlocal persistence_calls
                persistence_calls += 1
                if persistence_calls == 1:
                    raise RuntimeError("Notion unavailable")
                return "created"

            process_one_job(store, scorer, persist, "worker-1", retry_delay_seconds=0, retry_jitter_seconds=0)
            with closing(store.connect()) as connection, connection:
                connection.execute("UPDATE score_jobs SET next_run_at = CURRENT_TIMESTAMP")
            process_one_job(store, scorer, persist, "worker-1", retry_delay_seconds=0, retry_jitter_seconds=0)
            job = store.list_score_jobs()[0]

        self.assertEqual(scorer.calls, 1)
        self.assertEqual(persistence_calls, 2)
        self.assertEqual(job["status"], "completed")

    def test_low_quality_snapshot_is_not_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.enqueue_score_job("app-1", "fingerprint-1", status="queued")
            scorer = FakeScorer()

            process_one_job(
                store,
                scorer,
                lambda _result: "created",
                "worker-1",
                min_description_chars=200,
                retry_jitter_seconds=0,
            )
            job = store.list_score_jobs()[0]

        self.assertEqual(scorer.calls, 0)
        self.assertEqual(job["status"], "queued")
        self.assertIn("quality requirements", job["last_error"])


if __name__ == "__main__":
    unittest.main()
