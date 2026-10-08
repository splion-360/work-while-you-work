#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import mlflow
from huggingface_hub import HfApi, hf_hub_download
from mlflow import MlflowClient


def bundle_fingerprint(bundle_dir: Path) -> tuple[str, dict]:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for name in (manifest["classifier"]["path"], "manifest.json"):
        path = bundle_dir / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    if hashlib.sha256(
        (bundle_dir / manifest["classifier"]["path"]).read_bytes()
    ).hexdigest() != manifest["classifier"]["sha256"]:
        raise ValueError("classifier hash does not match the bundle manifest")
    return digest.hexdigest(), manifest


def release_metadata(version, fingerprint: str, manifest: dict) -> dict:
    return {
        "format_version": 1,
        "bundle_fingerprint": fingerprint,
        "classifier_sha256": manifest["classifier"]["sha256"],
        "encoder_fingerprint": manifest["encoder"]["fingerprint"],
        "mlflow_model_name": version.name,
        "mlflow_model_version": str(version.version),
        "mlflow_registration_run_id": version.run_id,
        "source_parent_mlflow_run_id": manifest.get("source_parent_mlflow_run_id"),
        "source_mlflow_run_id": manifest["source_mlflow_run_id"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--model-name", default="resume-jd-binary-scorer")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--repo-id", default="splion360/resume-jd-binary-scorer")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias(args.model_name, args.alias)
    bundle = Path(mlflow.artifacts.download_artifacts(artifact_uri=version.source))
    fingerprint, manifest = bundle_fingerprint(bundle)
    metadata = release_metadata(version, fingerprint, manifest)
    release_id = f"mlflow-v{version.version}-{fingerprint[:12]}"
    release_path = f"releases/{release_id}"
    release_file = f"{release_path}/release.json"
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)

    if api.file_exists(args.repo_id, release_file, repo_type="model"):
        existing = json.loads(
            Path(
                hf_hub_download(
                    args.repo_id,
                    release_file,
                    repo_type="model",
                    token=token,
                )
            ).read_text(encoding="utf-8")
        )
        if existing["bundle_fingerprint"] != fingerprint:
            raise RuntimeError("existing release ID has a different bundle fingerprint")
        print(json.dumps(existing, indent=2))
        return

    with tempfile.TemporaryDirectory(prefix="job-tracker-model-release-") as directory:
        staging = Path(directory)
        for name in (manifest["classifier"]["path"], "manifest.json"):
            (staging / name).write_bytes((bundle / name).read_bytes())
        (staging / "release.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=staging,
            path_in_repo=release_path,
            commit_message=f"Publish {release_id}",
        )

    metadata["hf_bundle_commit"] = artifact_commit.oid
    metadata["hf_repo_id"] = args.repo_id
    metadata["hf_release_path"] = release_path
    with tempfile.TemporaryDirectory(prefix="job-tracker-release-metadata-") as directory:
        metadata_path = Path(directory) / "release.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        release_commit = api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=metadata_path,
            path_in_repo=release_file,
            commit_message=f"Record provenance for {release_id}",
        )
    api.create_tag(
        args.repo_id,
        tag=release_id,
        revision=release_commit.oid,
        repo_type="model",
        tag_message=f"MLflow champion version {version.version}",
    )
    metadata["hf_release_revision"] = release_commit.oid
    metadata["hf_tag"] = release_id
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
