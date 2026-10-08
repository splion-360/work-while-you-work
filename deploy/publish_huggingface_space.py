#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from stage_huggingface_space import staged_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="splion360/resume-jd-scorer")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--hardware", default="zerogpu")
    parser.add_argument(
        "--model-repo-id", default="splion360/resume-jd-binary-scorer"
    )
    parser.add_argument("--model-revision", default="mlflow-v2-10cabd579086")
    parser.add_argument(
        "--model-release-path", default="releases/mlflow-v2-10cabd579086"
    )
    parser.add_argument(
        "--bge-revision", default="5617a9f61b028005a4858fdac845db406aefb181"
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    api = HfApi(token=token)
    release_revision = api.model_info(
        args.model_repo_id, revision=args.model_revision
    ).sha
    snapshot = Path(
        snapshot_download(
            args.model_repo_id,
            repo_type="model",
            revision=release_revision,
            allow_patterns=f"{args.model_release_path}/*",
            token=token,
        )
    )
    bundle = snapshot / args.model_release_path
    release = json.loads((bundle / "release.json").read_text(encoding="utf-8"))
    deployment = {
        **release,
        "hf_release_revision": release_revision,
        "hf_model_revision": args.model_revision,
        "bge_revision": args.bge_revision,
    }
    repo = api.create_repo(
        args.repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware=args.hardware,
        private=not args.public,
        exist_ok=True,
    )
    with staged_space(model_bundle=bundle, deployment_metadata=deployment) as folder:
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=folder,
            commit_message="Deploy BGE-M3 resume scorer",
        )
    runtime = api.get_space_runtime(args.repo_id)
    print(
        json.dumps(
            {
                "repo": str(repo),
                "commit": commit.oid,
                "stage": runtime.stage,
                "hardware": runtime.hardware,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
