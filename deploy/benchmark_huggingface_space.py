#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.huggingface_space_runtime import wait_until_running
from service.scoring_client import HuggingFaceSpaceClient

WORKLOADS = {
    "typical": (6_000, 6_000),
    "large": (16_000, 20_000),
    "maximum": (32_000, 50_000),
}


def percentile(values, percentile_value):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one value is required")
    position = (len(ordered) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def quota_capacity(quota_seconds, gpu_seconds):
    return int(quota_seconds // gpu_seconds) if gpu_seconds > 0 else 0


def synthetic_text(length, subject):
    sentence = (
        f"{subject} Python PyTorch machine learning model evaluation production inference "
        "distributed systems cloud data pipelines monitoring experimentation. "
    )
    return (sentence * (length // len(sentence) + 1))[:length]


def score_once(client, workload, resume_text, job_text):
    scoring_input = {
        "application_key": f"zerogpu-benchmark-{workload}",
        "resume_path": "synthetic.pdf",
        "resume_content_hash": hashlib.sha256(resume_text.encode()).hexdigest(),
        "job_description": job_text,
        "job_description_hash": hashlib.sha256(job_text.encode()).hexdigest(),
    }
    started = time.perf_counter()
    result = client.score(scoring_input)
    return {
        "wall_seconds": time.perf_counter() - started,
        "gpu_seconds": float(result["inference_ms"]) / 1000,
        "score": result["score"],
        "band": result["band"],
        "input_fingerprint": result["input_fingerprint"],
        "model_fingerprint": result["model_fingerprint"],
        "model_revision": result["hf_model_revision"],
        "mlflow_model_version": result["mlflow_model_version"],
    }


def summarize(samples):
    wall = [sample["wall_seconds"] for sample in samples]
    gpu = [sample["gpu_seconds"] for sample in samples]
    median_gpu = statistics.median(gpu)
    p95_gpu = percentile(gpu, 95)
    return {
        "sample_count": len(samples),
        "wall_seconds_p50": round(percentile(wall, 50), 3),
        "wall_seconds_p95": round(percentile(wall, 95), 3),
        "gpu_seconds_p50": round(percentile(gpu, 50), 3),
        "gpu_seconds_p95": round(percentile(gpu, 95), 3),
        "estimated_daily_scores": {
            "free_at_p50_gpu_seconds": quota_capacity(300, median_gpu),
            "free_at_p95_gpu_seconds": quota_capacity(300, p95_gpu),
            "pro_at_p50_gpu_seconds": quota_capacity(2400, median_gpu),
            "pro_at_p95_gpu_seconds": quota_capacity(2400, p95_gpu),
        },
    }


def main():
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="splion360/resume-jd-scorer")
    parser.add_argument("--space-url", default="https://splion360-resume-jd-scorer.hf.space")
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--mlflow-model-version", required=True)
    args = parser.parse_args()
    if args.warm_runs < 1:
        raise ValueError("--warm-runs must be positive")
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    api = HfApi(token=token)
    client = HuggingFaceSpaceClient(
        args.space_url,
        token,
        "bge-m3-knrm-binary-v1",
        "BAAI/bge-m3-colbert",
        expected_model_revision=args.model_revision,
        expected_model_fingerprint=args.model_fingerprint,
        expected_mlflow_model_version=args.mlflow_model_version,
        timeout=180,
    )
    report = {
        "space": args.repo_id,
        "model_revision": args.model_revision,
        "model_fingerprint": args.model_fingerprint,
        "mlflow_model_version": args.mlflow_model_version,
        "warm_runs_per_workload": args.warm_runs,
        "workloads": {},
    }
    for name, (resume_chars, job_chars) in WORKLOADS.items():
        resume_text = synthetic_text(resume_chars, "Resume")
        job_text = synthetic_text(job_chars, "Job description")
        client.pdf_extractor = lambda _path, text=resume_text: text
        client.file_hasher = lambda _path, text=resume_text: hashlib.sha256(text.encode()).hexdigest()

        api.restart_space(args.repo_id)
        wait_until_running(api, args.repo_id)
        cold = score_once(client, name, resume_text, job_text)
        warm = [score_once(client, name, resume_text, job_text) for _ in range(args.warm_runs)]
        report["workloads"][name] = {
            "resume_characters": resume_chars,
            "job_description_characters": job_chars,
            "cold": cold,
            "warm": summarize(warm),
            "warm_samples": warm,
        }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
