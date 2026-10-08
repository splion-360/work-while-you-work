#!/usr/bin/env python3
import hashlib
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlflow
import torch
from mlflow import MlflowClient
from resume_jd_scoring.runtime import build_scoring_runtime


class InferenceTelemetry:
    def __init__(self, client, experiment_id):
        self.client = client
        self.experiment_id = experiment_id
        self.events = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="mlflow-telemetry").start()

    def submit(self, event):
        self.events.put(event)

    def _run(self):
        while True:
            event = self.events.get()
            try:
                run = self.client.create_run(
                    self.experiment_id,
                    tags={key: str(value) for key, value in event["tags"].items()},
                    run_name="production-inference",
                )
                timestamp = int(time.time() * 1000)
                for name, value in event["metrics"].items():
                    self.client.log_metric(run.info.run_id, name, float(value), timestamp, 0)
                self.client.set_terminated(run.info.run_id, event.get("status", "FINISHED"))
            except Exception as error:
                print(json.dumps({"event": "mlflow_inference_log_failed", "error": str(error)[:500]}))
            finally:
                self.events.task_done()


class ScoringRuntime:
    def __init__(self):
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        model_name = os.getenv("MLFLOW_MODEL_NAME", "resume-jd-binary-scorer")
        alias = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
        version = self.client.get_model_version_by_alias(model_name, alias)
        bundle = Path(mlflow.artifacts.download_artifacts(artifact_uri=version.source))
        self.scorer_version = os.getenv("SCORER_VERSION", "bge-m3-knrm-binary-v1")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3-colbert")
        self.pooling_device = os.getenv("SCORING_DEVICE", "cuda:0")
        if self.pooling_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA scoring was requested but is unavailable")
        model_path = Path(os.getenv("BGE_MODEL_PATH", "/models/bge-m3"))
        self.runtime = build_scoring_runtime(
            bundle,
            model_path,
            scorer_version=self.scorer_version,
            embedding_model=self.embedding_model,
            device=self.pooling_device,
        )
        experiment = mlflow.set_experiment(
            os.getenv("MLFLOW_INFERENCE_EXPERIMENT", "resume-jd-production-inference")
        )
        self.telemetry = InferenceTelemetry(self.client, experiment.experiment_id)
        self.model_version = str(version.version)
        self.inference_lock = threading.Lock()

    def score(self, payload):
        scoring_input = payload["scoring_input"]
        expected_hash = scoring_input["resume_content_hash"]
        started = time.perf_counter()
        try:
            with self.inference_lock:
                response = self.runtime.score(payload)
        except Exception:
            self.telemetry.submit({
                "tags": {
                    "outcome": "failed",
                    "application_key_hash": hashlib.sha256(
                        scoring_input["application_key"].encode()
                    ).hexdigest(),
                    "input_fingerprint": payload["input_fingerprint"],
                    "model_version": self.model_version,
                },
                "metrics": {"latency_ms": (time.perf_counter() - started) * 1000},
                "status": "FAILED",
            })
            raise
        self.telemetry.submit({
            "tags": {
                "outcome": "completed",
                "application_key_hash": hashlib.sha256(
                    scoring_input["application_key"].encode()
                ).hexdigest(),
                "input_fingerprint": payload["input_fingerprint"],
                "resume_content_hash": expected_hash,
                "job_description_hash": scoring_input["job_description_hash"],
                "model_version": self.model_version,
                "fit_label": response["fit_label"],
            },
            "metrics": {
                "latency_ms": (time.perf_counter() - started) * 1000,
                "score": response["score"],
                "fit_probability": response["fit_probability"],
                "decision_threshold": response["decision_threshold"],
            },
        })
        return response


class Handler(BaseHTTPRequestHandler):
    runtime = None

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/healthz":
            self._json({"error": "Not found"}, 404)
            return
        self._json({
            "ok": True,
            "model_version": self.runtime.model_version,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        })

    def do_POST(self):
        if self.path != "/score":
            self._json({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            self._json(self.runtime.score(payload))
        except (KeyError, TypeError, ValueError) as error:
            self._json({"error": str(error)}, 400)
        except Exception as error:
            self._json({"error": str(error)}, 500)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    Handler.runtime = ScoringRuntime()
    ThreadingHTTPServer(("0.0.0.0", int(os.getenv("SCORING_ENGINE_PORT", "8770"))), Handler).serve_forever()
