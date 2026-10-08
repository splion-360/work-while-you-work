#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.huggingface_space_runtime import wait_until_running
from service.scoring_client import HuggingFaceSpaceClient


def main():
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="splion360/resume-jd-scorer")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--mlflow-model-version", required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    wait_until_running(HfApi(token=token), args.repo_id)

    scorer_version = "bge-m3-knrm-binary-v1"
    embedding_model = "BAAI/bge-m3-colbert"
    resume_text = (
        "Machine learning engineer experienced in Python, PyTorch, model evaluation, "
        "production inference, data pipelines, and AWS deployments."
    )
    job_description = (
        "Build and deploy machine learning models using Python and PyTorch. Develop "
        "production inference services and scalable cloud data pipelines."
    )
    scoring_input = {
        "application_key": "zero-gpu-smoke-test",
        "resume_path": "synthetic.pdf",
        "resume_content_hash": hashlib.sha256(resume_text.encode()).hexdigest(),
        "job_description": job_description,
        "job_description_hash": hashlib.sha256(job_description.encode()).hexdigest(),
    }
    client = HuggingFaceSpaceClient(
        f"https://{args.repo_id.replace('/', '-')}.hf.space",
        token,
        scorer_version,
        embedding_model,
        expected_model_revision=args.model_revision,
        expected_model_fingerprint=args.model_fingerprint,
        expected_mlflow_model_version=args.mlflow_model_version,
        pdf_extractor=lambda _path: resume_text,
        file_hasher=lambda _path: hashlib.sha256(resume_text.encode()).hexdigest(),
    )
    result = client.score(scoring_input)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "score",
                    "band",
                    "fit_label",
                    "fit_probability",
                    "decision_threshold",
                    "scorer_version",
                    "embedding_model",
                    "mlflow_model_version",
                    "mlflow_run_id",
                    "hf_model_revision",
                    "model_fingerprint",
                    "inference_ms",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
