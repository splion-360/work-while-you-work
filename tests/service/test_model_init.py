import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service.config import ScoringSettings
from service import model_init


class ModelInitializationTests(unittest.TestCase):
    def settings(self, directory, gpu_mode="required"):
        return ScoringSettings(
            database_path=Path(directory) / "state.sqlite3",
            ollama_url="http://ollama:11434",
            embedding_model="embedding-v1",
            scorer_version="v1",
            max_job_description_chars=50000,
            min_job_description_chars=200,
            ollama_timeout_seconds=60,
            job_description_retention_days=365,
            gpu_mode=gpu_mode,
            job_lease_seconds=300,
            max_score_attempts=5,
            worker_poll_seconds=2,
            reconciliation_interval_seconds=900,
            publication_uncertainty_seconds=900,
        )

    def test_writes_readiness_after_model_uses_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            responses = [{}, {}, {}, {"models": [{"name": "embedding-v1:latest", "size_vram": 42}]}]
            with patch.object(model_init, "READINESS_PATH", Path(directory) / "ready.json"):
                with patch.object(model_init, "request_json", side_effect=responses):
                    result = model_init.initialize_model(self.settings(directory), retries=1)

            self.assertTrue(result["gpu_ready"])
            self.assertTrue((Path(directory) / "ready.json").is_file())

    def test_required_gpu_fails_when_model_uses_no_vram(self):
        with tempfile.TemporaryDirectory() as directory:
            responses = [{}, {}, {}, {"models": [{"name": "embedding-v1", "size_vram": 0}]}]
            with patch.object(model_init, "request_json", side_effect=responses):
                with self.assertRaisesRegex(RuntimeError, "without GPU memory"):
                    model_init.initialize_model(self.settings(directory), retries=1)


if __name__ == "__main__":
    unittest.main()
