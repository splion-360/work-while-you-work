import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from service import healthcheck


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class HealthcheckTests(unittest.TestCase):
    @patch("service.healthcheck.urllib.request.urlopen")
    def test_hosted_health_checks_gradio_api_without_scoring(self, urlopen):
        urlopen.return_value = FakeResponse(
            json.dumps({"named_endpoints": {"/score": {}}}).encode()
        )
        settings = SimpleNamespace(
            scoring_backend="huggingface",
            hf_space_url="https://private-space.example",
            hf_token="secret-token",
        )

        self.assertTrue(healthcheck.scorer_healthy(settings))

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://private-space.example/gradio_api/info",
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")

    @patch("service.healthcheck.urllib.request.urlopen")
    def test_local_health_requires_gpu_ready_engine(self, urlopen):
        urlopen.return_value = FakeResponse(json.dumps({"ok": True, "gpu": "RTX"}).encode())
        settings = SimpleNamespace(
            scoring_backend="local", scoring_engine_url="http://scoring-engine:8770"
        )
        self.assertTrue(healthcheck.scorer_healthy(settings))

    @patch.object(healthcheck, "scorer_healthy", return_value=True)
    @patch.object(healthcheck.ScoringSettings, "from_env")
    @patch.object(healthcheck, "ScoringStore")
    def test_worker_health_requires_store_heartbeat_and_backend(
        self, store_class, from_env, scorer_healthy
    ):
        settings = SimpleNamespace(database_path="/data/state.sqlite3")
        from_env.return_value = settings
        store = Mock()
        store.is_healthy.return_value = True
        store.worker_is_healthy.return_value = True
        store_class.return_value = store

        self.assertTrue(healthcheck.worker_healthy())
        scorer_healthy.assert_called_once_with(settings)


if __name__ == "__main__":
    unittest.main()
