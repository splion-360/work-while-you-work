import tempfile
import threading
import time
import unittest
from pathlib import Path

from service.score_persistence import ScorePersistenceError, normalize_score_result, persist_score_result
from service.scoring_contract import score_input_fingerprint
from service.scoring_state import ScoringStore


class FakeNotion:
    def __init__(self, application_exists=True):
        self.application_exists = application_exists
        self.score_pages = []
        self.created = []
        self.updated = []

    def query_data_source(self, source_id, _filter):
        if source_id == "applications-source":
            return [{"id": "application-page"}] if self.application_exists else []
        return list(self.score_pages)

    def create_page(self, database_id, properties):
        self.created.append((database_id, properties))
        page = {"id": "score-page"}
        self.score_pages = [page]
        return page

    def update_page(self, page_id, properties):
        self.updated.append((page_id, properties))
        return {"id": page_id}


class SlowNotion(FakeNotion):
    def query_data_source(self, source_id, filter_value):
        result = super().query_data_source(source_id, filter_value)
        if source_id != "applications-source":
            time.sleep(0.05)
        return result


def scoring_input(job_hash="jd-1"):
    return {
        "application_key": "app-1",
        "canonical_url": "https://example.com/jobs/1",
        "source_job_id": "1",
        "job_description": "Build ML systems.",
        "job_description_hash": job_hash,
        "description_provenance": "original_saved",
        "captured_at": "2026-09-02T20:00:00Z",
        "extraction_quality": "complete",
        "resume_version": "professional/resume-v1",
        "resume_path": "/resumes/resume-v1.pdf",
        "resume_content_hash": "resume-1",
        "role_company_key": "company-role-1",
    }


FINGERPRINT_1 = score_input_fingerprint(scoring_input(), "weights-v1", "embedding-v1")
FINGERPRINT_2 = score_input_fingerprint(scoring_input("jd-2"), "weights-v1", "embedding-v1")


def score_result(fingerprint=FINGERPRINT_1, score=82.0):
    return {
        "application_key": "app-1",
        "input_fingerprint": fingerprint,
        "scorer_version": "weights-v1",
        "embedding_model": "embedding-v1",
        "score": score,
        "band": "Strong",
        "matched_terms": ["Python", "PyTorch", "Python"],
        "category_breakdown": {"skills": 80.0, "requirements": 84.0},
    }


class ScorePersistenceTests(unittest.TestCase):
    def make_store(self, directory):
        store = ScoringStore(Path(directory) / "state.sqlite3")
        store.initialize()
        record = scoring_input()
        store.save_scoring_input(record)
        store.save_scoring_snapshot(FINGERPRINT_1, record)
        store.save_scoring_snapshot(FINGERPRINT_2, scoring_input("jd-2"))
        return store

    def test_first_score_is_stored_and_published_to_notion(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            notion = FakeNotion()

            outcome = persist_score_result(
                store, notion, "applications-source", "scores-database", "scores-source", score_result()
            )

            runs = store.list_score_runs("app-1")
        self.assertEqual(outcome, "created")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["is_application_time"], 1)
        database_id, properties = notion.created[0]
        self.assertEqual(database_id, "scores-database")
        self.assertEqual(set(properties), {
            "Application key", "Score", "Band", "Matched terms", "Category breakdown",
            "Cache key", "Fit label", "Fit probability", "Decision threshold",
        })
        self.assertNotIn("Created at", properties)
        self.assertNotIn("Updated at", properties)

    def test_same_fingerprint_updates_but_changed_input_stays_local(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            notion = FakeNotion()
            persist_score_result(
                store, notion, "applications-source", "scores-database", "scores-source", score_result()
            )
            corrected = persist_score_result(
                store, notion, "applications-source", "scores-database", "scores-source", score_result(score=85.0)
            )
            experimental = persist_score_result(
                store, notion, "applications-source", "scores-database", "scores-source",
                score_result(fingerprint=FINGERPRINT_2, score=90.0),
            )
            runs = store.list_score_runs("app-1")

        self.assertEqual(corrected, "updated")
        self.assertEqual(experimental, "stored_locally")
        self.assertEqual(len(notion.updated), 1)
        self.assertEqual(len(runs), 2)
        self.assertEqual(sum(run["is_application_time"] for run in runs), 1)

    def test_unknown_application_key_is_rejected_before_local_or_notion_write(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            notion = FakeNotion(application_exists=False)
            with self.assertRaisesRegex(ScorePersistenceError, "does not exist"):
                persist_score_result(
                    store, notion, "applications-source", "scores-database", "scores-source", score_result()
                )
            self.assertEqual(store.list_score_runs("app-1"), [])
            self.assertEqual(notion.created, [])

    def test_duplicate_notion_score_rows_fail_visibly(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            notion = FakeNotion()
            notion.score_pages = [{"id": "one"}, {"id": "two"}]
            with self.assertRaisesRegex(ScorePersistenceError, "multiple"):
                persist_score_result(
                    store, notion, "applications-source", "scores-database", "scores-source", score_result()
                )

    def test_unknown_input_fingerprint_is_rejected_before_score_write(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            notion = FakeNotion()
            with self.assertRaisesRegex(ScorePersistenceError, "stored scoring snapshot"):
                persist_score_result(
                    store,
                    notion,
                    "applications-source",
                    "scores-database",
                    "scores-source",
                    score_result(fingerprint="invented-fingerprint"),
                )

            self.assertEqual(store.list_score_runs("app-1"), [])
            self.assertEqual(notion.created, [])

    def test_concurrent_publications_create_only_one_notion_row(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            notion = SlowNotion()
            errors = []

            def publish():
                try:
                    persist_score_result(
                        store,
                        notion,
                        "applications-source",
                        "scores-database",
                        "scores-source",
                        score_result(),
                    )
                except Exception as error:
                    errors.append(error)

            threads = [threading.Thread(target=publish) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(notion.created), 1)
        self.assertEqual(len(notion.updated), 1)

    def test_rejects_inconsistent_band_and_invalid_category_values(self):
        with self.assertRaisesRegex(ScorePersistenceError, "Band must be Weak"):
            normalize_score_result(score_result(score=20.0))
        invalid = score_result() | {"category_breakdown": {"skills": 101}}
        with self.assertRaisesRegex(ScorePersistenceError, "between 0 and 100"):
            normalize_score_result(invalid)
        malformed_terms = score_result() | {"matched_terms": ["Python", 42]}
        with self.assertRaisesRegex(ScorePersistenceError, "only strings"):
            normalize_score_result(malformed_terms)
        malformed_categories = score_result() | {"category_breakdown": {"": 80}}
        with self.assertRaisesRegex(ScorePersistenceError, "non-empty strings"):
            normalize_score_result(malformed_categories)


if __name__ == "__main__":
    unittest.main()
