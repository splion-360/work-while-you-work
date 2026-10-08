import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service.backfill import run_backfill
from service.scoring_state import ScoringStore


def page(key="app-1", resume_label="Professional | resume-v1"):
    def text(value):
        return {"rich_text": [{"plain_text": value}]}

    return {
        "created_time": "2026-01-01T00:00:00Z",
        "properties": {
            "Application key": text(key),
            "Resume version": {"select": {"name": resume_label}},
            "Job URL": {"url": "https://example.com/jobs/1"},
        },
    }


class FakeNotion:
    def __init__(self, pages):
        self.pages = pages

    def query_data_source(self, _source_id):
        return self.pages


class BackfillTests(unittest.TestCase):
    def test_queues_only_rows_with_supplied_provenance_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "resume-v1.pdf"
            resume_path.write_bytes(b"historical resume")
            resumes = [{
                "version": "professional/resume-v1",
                "label": "Professional | resume-v1",
                "path": str(resume_path),
            }]
            supplied = {"app-1": {
                "job_description": "Build reliable production machine learning systems with Python and PyTorch. " * 2,
                "provenance": "recaptured_current_page",
                "captured_at": "2026-09-02T20:00:00Z",
                "extraction_quality": "complete",
                "resume_content_hash": hashlib.sha256(b"historical resume").hexdigest(),
            }}
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            with patch.dict("os.environ", {"SCORING_DB_PATH": str(Path(directory) / "state.sqlite3")}, clear=False):
                report = run_backfill(store, FakeNotion([page()]), "applications", resumes, supplied, execute=True)

            saved = store.get_scoring_input("app-1")
            jobs = store.list_score_jobs()

        self.assertEqual(report["completed"], 1)
        self.assertEqual(saved["description_provenance"], "recaptured_current_page")
        self.assertEqual(saved["resume_content_hash"], hashlib.sha256(b"historical resume").hexdigest())
        self.assertEqual(jobs[0]["status"], "queued")

    def test_dry_run_reports_missing_and_low_quality_descriptions_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            supplied = {"app-2": {"job_description": "too short", "provenance": "manual_supplied"}}
            pages = [page("app-1"), page("app-2")]

            report = run_backfill(store, FakeNotion(pages), "applications", [], supplied, execute=False)
            jobs = store.list_score_jobs()

        self.assertEqual(report["completed"], 0)
        self.assertEqual(report["eligible"], 0)
        self.assertEqual(report["skipped"], 2)
        self.assertEqual(report["reasons"], {"low_quality_description": 1, "missing_description": 1})
        self.assertEqual(jobs, [])

    def test_resume_label_must_resolve_to_one_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            supplied = {"app-1": {"job_description": "A sufficiently detailed job description. " * 4}}
            supplied["app-1"]["resume_content_hash"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            duplicate = {"version": "v1", "label": "Professional | resume-v1", "path": __file__}

            report = run_backfill(
                store, FakeNotion([page()]), "applications", [duplicate, duplicate], supplied, execute=True
            )
            jobs = store.list_score_jobs()

        self.assertEqual(report["reasons"], {"ambiguous_resume_snapshot": 1})
        self.assertEqual(jobs, [])

    def test_dry_run_counts_verifiable_rows_as_eligible_not_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "resume-v1.pdf"
            resume_path.write_bytes(b"historical resume")
            resumes = [{
                "version": "professional/resume-v1",
                "label": "Professional | resume-v1",
                "path": str(resume_path),
            }]
            supplied = {"app-1": {
                "job_description": "Build reliable production machine learning systems with Python and PyTorch. " * 2,
                "captured_at": "2026-09-02T20:00:00Z",
                "resume_content_hash": hashlib.sha256(b"historical resume").hexdigest(),
            }}
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()

            report = run_backfill(store, FakeNotion([page()]), "applications", resumes, supplied)
            jobs = store.list_score_jobs()

        self.assertEqual(report["eligible"], 1)
        self.assertEqual(report["completed"], 0)
        self.assertEqual(jobs, [])


if __name__ == "__main__":
    unittest.main()
