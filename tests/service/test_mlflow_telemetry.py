import unittest

from service.mlflow_telemetry import MlflowTelemetry


class MlflowTelemetryTests(unittest.TestCase):
    def test_publishes_one_finished_run_with_tags_and_metrics(self):
        telemetry = object.__new__(MlflowTelemetry)
        telemetry.tracking_uri = "http://mlflow:5000"
        telemetry.experiment_name = "production"
        telemetry.timeout = 5
        calls = []

        def request(path, payload=None):
            calls.append((path, payload))
            if "get-by-name" in path:
                return {"experiment": {"experiment_id": "12"}}
            if path.endswith("runs/create"):
                return {"run": {"info": {"run_id": "run-1"}}}
            return {}

        telemetry._request = request
        telemetry._publish({
            "tags": {"outcome": "completed", "input_fingerprint": "abc"},
            "metrics": {"score": 82.0, "gateway_latency_ms": 1200},
        })

        create = next(payload for path, payload in calls if path.endswith("runs/create"))
        batch = next(payload for path, payload in calls if path.endswith("runs/log-batch"))
        update = next(payload for path, payload in calls if path.endswith("runs/update"))
        self.assertEqual(create["experiment_id"], "12")
        self.assertEqual(batch["run_id"], "run-1")
        self.assertEqual({item["key"] for item in batch["metrics"]}, {"score", "gateway_latency_ms"})
        self.assertEqual(update["status"], "FINISHED")


if __name__ == "__main__":
    unittest.main()
