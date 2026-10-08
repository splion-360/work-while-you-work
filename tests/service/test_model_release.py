import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "release_mlflow_champion.py"


class ModelReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            spec = importlib.util.spec_from_file_location("release_mlflow_champion", SCRIPT)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        except ModuleNotFoundError:
            cls.module = None

    def test_validates_bundle_and_records_mlflow_lineage(self):
        if self.module is None:
            self.skipTest("MLflow deployment dependencies are not installed")
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            classifier = b"classifier"
            (bundle / "classifier.pt").write_bytes(classifier)
            manifest = {
                "classifier": {
                    "path": "classifier.pt",
                    "sha256": hashlib.sha256(classifier).hexdigest(),
                },
                "encoder": {"fingerprint": "encoder-fingerprint"},
                "source_parent_mlflow_run_id": "parent-run",
                "source_mlflow_run_id": "source-run",
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            fingerprint, loaded = self.module.bundle_fingerprint(bundle)
            metadata = self.module.release_metadata(
                SimpleNamespace(name="scorer", version="3", run_id="registration-run"),
                fingerprint,
                loaded,
            )

        self.assertEqual(metadata["mlflow_model_version"], "3")
        self.assertEqual(metadata["mlflow_registration_run_id"], "registration-run")
        self.assertEqual(metadata["source_mlflow_run_id"], "source-run")
        self.assertEqual(metadata["encoder_fingerprint"], "encoder-fingerprint")


if __name__ == "__main__":
    unittest.main()
