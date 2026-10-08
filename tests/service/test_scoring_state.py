import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from service.config import ScoringSettings
from service.scoring_state import ScoringStore


class ScoringSettingsTests(unittest.TestCase):
    def test_reads_explicit_runtime_settings(self):
        values = {
            "SCORING_DB_PATH": "/tmp/scores.sqlite3",
            "EMBEDDING_MODEL": "embeddinggemma:latest",
            "SCORER_VERSION": "v2",
            "MAX_JOB_DESCRIPTION_CHARS": "42000",
            "MIN_JOB_DESCRIPTION_CHARS": "250",
            "JOB_DESCRIPTION_RETENTION_DAYS": "180",
            "RECONCILIATION_INTERVAL_SECONDS": "600",
            "PUBLICATION_UNCERTAINTY_SECONDS": "1200",
            "SCORING_BACKEND": "huggingface",
            "HF_SPACE_URL": "https://private-space.example/",
            "HF_TOKEN": "secret-token",
            "HF_MODEL_REVISION": "hf-revision",
            "HF_MODEL_FINGERPRINT": "bundle-fingerprint",
            "MLFLOW_MODEL_VERSION": "2",
        }
        with patch.dict(os.environ, values, clear=True):
            settings = ScoringSettings.from_env()
            hosted_values = {
                "backend": settings.scoring_backend,
                "url": settings.hf_space_url,
                "token": settings.hf_token,
                "revision": settings.hf_model_revision,
                "fingerprint": settings.hf_model_fingerprint,
                "mlflow_version": settings.mlflow_model_version,
            }

        self.assertEqual(settings.database_path, Path("/tmp/scores.sqlite3"))
        self.assertEqual(settings.max_job_description_chars, 42000)
        self.assertEqual(settings.min_job_description_chars, 250)
        self.assertEqual(settings.job_description_retention_days, 180)
        self.assertEqual(settings.reconciliation_interval_seconds, 600)
        self.assertEqual(settings.publication_uncertainty_seconds, 1200)
        self.assertEqual(hosted_values["backend"], "huggingface")
        self.assertEqual(hosted_values["url"], "https://private-space.example")
        self.assertEqual(hosted_values["token"], "secret-token")
        self.assertEqual(hosted_values["revision"], "hf-revision")
        self.assertEqual(hosted_values["fingerprint"], "bundle-fingerprint")
        self.assertEqual(hosted_values["mlflow_version"], "2")

    def test_model_revision_changes_score_cache_identity(self):
        scoring_input = {
            "application_key": "app-1",
            "resume_content_hash": "resume-hash",
            "job_description_hash": "job-hash",
        }
        from service.scoring_contract import score_input_fingerprint

        first = score_input_fingerprint(scoring_input, "scorer", "model", "revision-1")
        second = score_input_fingerprint(scoring_input, "scorer", "model", "revision-2")
        self.assertNotEqual(first, second)

    def test_comparison_cache_ignores_application_identity(self):
        from service.scoring_contract import role_company_key, score_cache_key

        first = {
            "application_key": "app-1",
            "resume_content_hash": "resume-hash",
            "job_description_hash": "job-hash",
            "role_company_key": "company-role",
        }
        second = {
            **first,
            "application_key": "app-2",
            "job_description_hash": "different-job-description",
        }

        self.assertEqual(
            score_cache_key(first, "scorer", "model", "revision"),
            score_cache_key(second, "scorer", "model", "revision"),
        )
        self.assertNotEqual(
            score_cache_key(first, "scorer", "model", "revision"),
            score_cache_key(
                {**second, "resume_content_hash": "different-resume"},
                "scorer",
                "model",
                "revision",
            ),
        )
        self.assertEqual(
            role_company_key(" Example AI ", "ML  Engineer"),
            role_company_key("example ai", "ml engineer"),
        )
        self.assertNotEqual(
            role_company_key("Example AI", "ML Engineer"),
            role_company_key("Example AI", "Data Engineer"),
        )

    def test_rejects_an_unknown_scoring_backend(self):
        with patch.dict(
            os.environ,
            {"SCORING_BACKEND": "unknown"},
            clear=True,
        ):
            settings = ScoringSettings.from_env()
            with self.assertRaisesRegex(ValueError, "SCORING_BACKEND"):
                _ = settings.scoring_backend

class ScoringStoreTests(unittest.TestCase):
    def test_initialization_is_idempotent_and_enables_safety_pragmas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = ScoringStore(path)
            store.initialize()
            store.initialize()

            with closing(sqlite3.connect(path)) as connection, connection:
                tables = {row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )}
                journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                with closing(store.connect()) as store_connection:
                    foreign_keys = store_connection.execute("PRAGMA foreign_keys").fetchone()[0]

        self.assertTrue({"scoring_inputs", "score_runs", "score_jobs"}.issubset(tables))
        self.assertEqual(journal_mode, "wal")
        self.assertEqual(foreign_keys, 1)

    def test_schema_enforces_unique_fingerprints_and_foreign_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            with closing(store.connect()) as connection, connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "INSERT INTO score_jobs (application_key, idempotency_key, input_fingerprint) VALUES (?, ?, ?)",
                        ("missing", "job-1", "input-1"),
                    )

    def test_renames_application_key_across_scoring_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            scoring_input = {
                "application_key": "app-1",
                "canonical_url": "https://example.com/jobs/1",
                "source_job_id": "1",
                "job_description": "Original description",
                "job_description_hash": "jd-1",
                "description_provenance": "original_saved",
                "captured_at": "2026-09-01T00:00:00Z",
                "extraction_quality": "complete",
                "resume_version": "resume-v1",
                "resume_path": "/resume.pdf",
                "resume_content_hash": "resume-1",
                "role_company_key": "role-1",
            }
            store.save_scoring_input(scoring_input)
            store.save_scoring_snapshot("fingerprint-1", scoring_input)
            store.enqueue_score_job("app-1", "fingerprint-1")
            store.save_score_run({
                "application_key": "app-1",
                "input_fingerprint": "fingerprint-1",
                "scorer_version": "v1",
                "embedding_model": "model",
                "deployment_revision": "",
                "score": 80,
                "band": "Strong",
                "matched_terms": [],
                "category_breakdown": {},
            })

            renamed = store.rename_application("app-1", "app-2")
            self.assertTrue(renamed)
            self.assertIsNone(store.get_scoring_input("app-1"))
            self.assertEqual(store.get_scoring_input("app-2")["job_description_hash"], "jd-1")
            self.assertEqual(store.list_score_runs("app-2")[0]["input_fingerprint"], "fingerprint-1")
            self.assertEqual(store.list_scoring_snapshots()[0]["application_key"], "app-2")
            self.assertEqual(store.list_score_jobs()[0]["application_key"], "app-2")

    def test_score_cache_expires_after_its_ttl(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            store.save_cached_score("cache-1", {"score": 81.0}, ttl_seconds=300)
            fresh = store.get_cached_score("cache-1")
            with closing(store.connect()) as connection, connection:
                connection.execute(
                    "UPDATE score_cache SET expires_at = '2000-01-01 00:00:00'"
                )
            expired = store.get_cached_score("cache-1")

        self.assertEqual(fresh, {"score": 81.0})
        self.assertIsNone(expired)

    def test_scoring_input_is_immutable_after_first_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            original = {
                "application_key": "app-1",
                "canonical_url": "https://example.com/jobs/1",
                "source_job_id": "1",
                "job_description": "Original description",
                "job_description_hash": "jd-1",
                "description_provenance": "original_saved",
                "captured_at": "2026-09-02T20:00:00Z",
                "extraction_quality": "complete",
                "resume_version": "professional/resume-v1",
                "resume_path": "/resumes/resume-v1.pdf",
                "resume_content_hash": "resume-1",
            }
            self.assertTrue(store.save_scoring_input(original))
            changed = {**original, "job_description": "Later description", "job_description_hash": "jd-2"}
            self.assertFalse(store.save_scoring_input(changed))

            saved = store.get_scoring_input("app-1")
            self.assertEqual(saved["job_description"], "Original description")
            self.assertEqual(saved["job_description_hash"], "jd-1")

    def test_worker_heartbeat_is_health_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            self.assertFalse(store.worker_is_healthy(600))

            store.record_worker_heartbeat("worker-1")

            self.assertTrue(store.worker_is_healthy(600))
            self.assertTrue(store.is_healthy())

    def test_changed_inputs_keep_separate_immutable_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            original = {
                "application_key": "app-1",
                "canonical_url": "https://example.com/jobs/1",
                "source_job_id": "1",
                "job_description": "Original description",
                "job_description_hash": "jd-1",
                "description_provenance": "original_saved",
                "captured_at": "2026-09-02T20:00:00Z",
                "extraction_quality": "complete",
                "resume_version": "professional/resume-v1",
                "resume_path": "/resumes/resume-v1.pdf",
                "resume_content_hash": "resume-1",
            }
            changed = {**original, "job_description": "Changed description", "job_description_hash": "jd-2"}
            store.save_scoring_input(original)
            store.save_scoring_snapshot("fingerprint-1", original)
            store.save_scoring_snapshot("fingerprint-2", changed)

            first = store.get_scoring_snapshot("fingerprint-1")
            second = store.get_scoring_snapshot("fingerprint-2")

        self.assertEqual(first["job_description_hash"], "jd-1")
        self.assertEqual(second["job_description_hash"], "jd-2")

    def test_deleting_application_cascades_through_scoring_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            scoring_input = {
                "application_key": "app-1",
                "canonical_url": "https://example.com/jobs/1",
                "source_job_id": "1",
                "job_description": "Description",
                "job_description_hash": "jd-1",
                "description_provenance": "original_saved",
                "captured_at": "2026-09-02T20:00:00Z",
                "extraction_quality": "complete",
                "resume_version": "professional/resume-v1",
                "resume_path": "/resumes/resume-v1.pdf",
                "resume_content_hash": "resume-1",
            }
            store.save_scoring_input(scoring_input)
            store.save_scoring_snapshot("fingerprint-1", scoring_input)
            store.enqueue_score_job("app-1", "fingerprint-1")

            self.assertTrue(store.delete_application("app-1"))

            self.assertIsNone(store.get_scoring_input("app-1"))
            self.assertIsNone(store.get_scoring_snapshot("fingerprint-1"))
            self.assertEqual(store.list_score_jobs(), [])

if __name__ == "__main__":
    unittest.main()
