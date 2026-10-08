import json
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache


class MlflowTelemetry:
    def __init__(self, tracking_uri, experiment_name, timeout=5):
        self.tracking_uri = tracking_uri.rstrip("/")
        self.experiment_name = experiment_name
        self.timeout = timeout
        self.events = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="mlflow-telemetry").start()

    def submit(self, event):
        self.events.put(dict(event))

    def _request(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.tracking_uri}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _experiment_id(self):
        query = urllib.parse.urlencode({"experiment_name": self.experiment_name})
        try:
            result = self._request(f"/api/2.0/mlflow/experiments/get-by-name?{query}")
            return result["experiment"]["experiment_id"]
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        result = self._request(
            "/api/2.0/mlflow/experiments/create", {"name": self.experiment_name}
        )
        return result["experiment_id"]

    def _publish(self, event):
        now = int(time.time() * 1000)
        tags = [
            {"key": str(key), "value": str(value)}
            for key, value in sorted(event.get("tags", {}).items())
        ]
        created = self._request(
            "/api/2.0/mlflow/runs/create",
            {
                "experiment_id": self._experiment_id(),
                "start_time": now,
                "run_name": "hosted-production-inference",
                "tags": tags,
            },
        )
        run_id = created["run"]["info"]["run_id"]
        metrics = [
            {"key": str(key), "value": float(value), "timestamp": now, "step": 0}
            for key, value in sorted(event.get("metrics", {}).items())
        ]
        if metrics:
            self._request(
                "/api/2.0/mlflow/runs/log-batch",
                {"run_id": run_id, "metrics": metrics, "params": [], "tags": []},
            )
        self._request(
            "/api/2.0/mlflow/runs/update",
            {
                "run_id": run_id,
                "status": event.get("status", "FINISHED"),
                "end_time": int(time.time() * 1000),
            },
        )

    def _run(self):
        while True:
            event = self.events.get()
            try:
                self._publish(event)
            except Exception as error:
                print(json.dumps({
                    "event": "mlflow_inference_log_failed",
                    "error": str(error)[:500],
                }))
            finally:
                self.events.task_done()


@lru_cache(maxsize=4)
def get_mlflow_telemetry(tracking_uri, experiment_name):
    if not tracking_uri:
        return None
    return MlflowTelemetry(tracking_uri, experiment_name)
