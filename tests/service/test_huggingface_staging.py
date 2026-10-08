import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "stage_huggingface_space.py"
SPEC = importlib.util.spec_from_file_location("stage_huggingface_space", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HuggingFaceSpaceStagingTests(unittest.TestCase):
    def test_stages_runtime_without_copying_repository_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "space"
            with MODULE.staged_space(
                destination, deployment_metadata={"hf_release_revision": "commit-1"}
            ):
                self.assertTrue((destination / "app.py").is_file())
                self.assertTrue((destination / "model" / "classifier.pt").is_file())
                self.assertTrue(
                    (destination / "resume_jd_scoring" / "inference.py").is_file()
                )
                self.assertFalse(
                    (destination / "resume_jd_scoring" / "__pycache__").exists()
                )
                self.assertFalse((destination / ".env").exists())
                deployment = destination / "model" / "deployment.json"
                self.assertIn("commit-1", deployment.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
