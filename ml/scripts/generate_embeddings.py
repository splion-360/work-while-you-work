import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from resume_jd_scoring.embeddings import (
    DocumentEmbedder,
    TokenChunker,
    model_fingerprint,
    sha256_file,
    validate_embedding_rows,
    write_embedding_cache,
)

DEFAULT_MODEL_DIR = Path("data/models/embeddinggemma-300m")
DEFAULT_MODEL_NAME = "google/embeddinggemma-300m"
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT_ROOT = Path("data/embeddings")
DEFAULT_REPORT = Path("artifacts/model__embedding_generation.json")
PAIR_CLASSIFICATION_PROMPT = "task: sentence similarity | query: "
CONTEXT_LENGTH = 2048
OVERLAP_TOKENS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen dense document vectors.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--overlap-tokens", type=int, default=OVERLAP_TOKENS)
    parser.add_argument("--prompt", default=PAIR_CLASSIFICATION_PROMPT)
    parser.add_argument("--prompt-name", default="PairClassification")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def load_documents(path: Path, hash_column: str) -> dict[str, str]:
    table = pq.read_table(path, columns=[hash_column, "text"])
    return dict(zip(table.column(hash_column).to_pylist(), table.column("text").to_pylist()))


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")
    return requested


def main() -> None:
    args = parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, use_fast=True
    )
    chunker = TokenChunker(
        tokenizer,
        max_length=args.context_length,
        overlap_tokens=args.overlap_tokens,
        prompt=args.prompt,
    )
    encoding = {
        "pipeline_version": 1,
        "context_length": args.context_length,
        "overlap_tokens": args.overlap_tokens,
        "content_capacity": chunker.content_capacity,
        "prompt_name": args.prompt_name,
        "prompt": args.prompt,
        "chunk_pooling": "arithmetic_mean",
        "chunk_normalization": "l2_after_mean",
    }
    if args.model_name != DEFAULT_MODEL_NAME:
        encoding["model_name"] = args.model_name
    fingerprint, model_files = model_fingerprint(args.model_dir, encoding)
    output_dir = args.output_root / fingerprint
    resume_path = output_dir / "resumes.parquet"
    jd_path = output_dir / "jds.parquet"
    metadata_path = output_dir / "metadata.json"

    device = resolve_device(args.device)
    model = SentenceTransformer(
        str(args.model_dir),
        device=device,
        local_files_only=True,
    )
    if model.max_seq_length != args.context_length:
        raise ValueError(
            f"model max sequence length {model.max_seq_length} differs from {args.context_length}"
        )
    dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        raise ValueError("model did not report an embedding dimension")

    resumes = load_documents(args.processed_dir / "resumes.parquet", "resume_hash")
    jds = load_documents(args.processed_dir / "jds.parquet", "jd_hash")
    embedder = DocumentEmbedder(
        model,
        chunker,
        prompt=args.prompt,
        batch_size=args.batch_size,
    )

    started = time.perf_counter()
    resume_rows = embedder.embed_documents(resumes)
    resume_seconds = time.perf_counter() - started
    started = time.perf_counter()
    jd_rows = embedder.embed_documents(jds)
    jd_seconds = time.perf_counter() - started

    resume_validation = validate_embedding_rows(resume_rows, expected_dimension=dimension)
    jd_validation = validate_embedding_rows(jd_rows, expected_dimension=dimension)
    write_embedding_cache(resume_rows, resume_path, dimension=dimension)
    write_embedding_cache(jd_rows, jd_path, dimension=dimension)

    report = {
        "fingerprint": fingerprint,
        "model": {
            "name": args.model_name,
            "path": str(args.model_dir),
            "files": model_files,
            "embedding_dimension": dimension,
            "frozen": True,
            "dtype": str(next(model.parameters()).dtype),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "batch_size": args.batch_size,
            "resume_seconds": round(resume_seconds, 3),
            "jd_seconds": round(jd_seconds, 3),
        },
        "encoding": encoding,
        "validation": {"resumes": resume_validation, "jds": jd_validation},
        "outputs": {
            "resumes": file_record(resume_path),
            "jds": file_record(jd_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
