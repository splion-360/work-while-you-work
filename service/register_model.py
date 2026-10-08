#!/usr/bin/env python3
import hashlib
import json
import os
import time
from pathlib import Path

import mlflow
from mlflow import MlflowClient


BUNDLE_DIR = Path(os.getenv("SCORER_BUNDLE_PATH", "/app/ml/runtime/binary-scorer"))
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "resume-jd-binary-scorer")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


def bundle_fingerprint(bundle_dir):
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for name in ("classifier.pt", "manifest.json"):
        digest.update(name.encode())
        digest.update((bundle_dir / name).read_bytes())
    return digest.hexdigest(), manifest


def register_bundle(bundle_dir=BUNDLE_DIR):
    fingerprint, manifest = bundle_fingerprint(bundle_dir)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "resume-jd-production"))
    client = MlflowClient()
    try:
        client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(MODEL_NAME)

    for version in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if version.tags.get("bundle_fingerprint") == fingerprint:
            client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version.version)
            return version

    with mlflow.start_run(run_name=f"register-{fingerprint[:12]}") as run:
        mlflow.set_tags({
            "purpose": "production-model-registration",
            "bundle_fingerprint": fingerprint,
            "classifier_sha256": manifest["classifier"]["sha256"],
            "encoder_fingerprint": manifest["encoder"]["fingerprint"],
            "source_parent_mlflow_run_id": manifest["source_parent_mlflow_run_id"],
            "source_mlflow_run_id": manifest["source_mlflow_run_id"],
        })
        mlflow.log_artifacts(str(bundle_dir), artifact_path="binary-scorer")
        source = mlflow.get_artifact_uri("binary-scorer")
        version = client.create_model_version(
            MODEL_NAME,
            source,
            run_id=run.info.run_id,
            tags={
                "bundle_fingerprint": fingerprint,
                "classifier_sha256": manifest["classifier"]["sha256"],
                "encoder_fingerprint": manifest["encoder"]["fingerprint"],
                "source_parent_mlflow_run_id": manifest["source_parent_mlflow_run_id"],
                "source_mlflow_run_id": manifest["source_mlflow_run_id"],
            },
        )
    for _ in range(60):
        version = client.get_model_version(MODEL_NAME, version.version)
        if version.status == "READY":
            break
        if version.status == "FAILED_REGISTRATION":
            raise RuntimeError(version.status_message or "MLflow model registration failed")
        time.sleep(1)
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version.version)
    return version


if __name__ == "__main__":
    registered = register_bundle()
    print(json.dumps({"model": MODEL_NAME, "alias": MODEL_ALIAS, "version": registered.version}))
