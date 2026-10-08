import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from resume_jd_scoring.embeddings import sha256_file
from resume_jd_scoring.token_lengths import (
    HuggingFaceTokenCounter,
    count_documents,
    token_length_summary,
)

DEFAULT_MODEL_DIR = Path("data/models/embeddinggemma-300m")
DEFAULT_CONTEXT_LENGTH = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical document token lengths.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data__token_length_analysis.json")
    )
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    return parser.parse_args()


def load_documents(path: Path, hash_column: str) -> dict[str, str]:
    table = pq.read_table(path, columns=[hash_column, "text"])
    return dict(zip(table.column(hash_column).to_pylist(), table.column("text").to_pylist()))


def longest_documents(
    documents: dict[str, str], lengths: dict[str, int], context_length: int
) -> list[dict[str, Any]]:
    rows = []
    for item_hash, token_count in sorted(
        lengths.items(), key=lambda item: (-item[1], item[0])
    )[:10]:
        text = documents[item_hash]
        rows.append(
            {
                "hash": item_hash,
                "tokens": token_count,
                "tokens_over_context": max(0, token_count - context_length),
                "characters": len(text),
                "head_preview": text[:240],
                "tail_preview": text[-240:],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.context_length <= 0:
        raise ValueError("context length must be positive")
    counter = HuggingFaceTokenCounter.from_pretrained(args.model_dir)
    if counter.tokenizer.model_max_length != args.context_length:
        raise ValueError(
            f"tokenizer context {counter.tokenizer.model_max_length} differs from {args.context_length}"
        )
    resumes = load_documents(args.processed_dir / "resumes.parquet", "resume_hash")
    jds = load_documents(args.processed_dir / "jds.parquet", "jd_hash")
    resume_lengths = count_documents(resumes, counter)
    jd_lengths = count_documents(jds, counter)
    report = {
        "configuration": {
            "model_dir": str(args.model_dir),
            "context_length": args.context_length,
        },
        "model": {
            "name": "google/embeddinggemma-300m",
            "weights_sha256": sha256_file(args.model_dir / "model.safetensors"),
            "embedding_length": 768,
        },
        "tokenizer": {
            **counter.metadata(),
            "tokenizer_json_sha256": sha256_file(args.model_dir / "tokenizer.json"),
        },
        "resumes": {
            "summary": token_length_summary(resume_lengths.values(), args.context_length),
            "longest": longest_documents(resumes, resume_lengths, args.context_length),
        },
        "jds": {
            "summary": token_length_summary(jd_lengths.values(), args.context_length),
            "longest": longest_documents(jds, jd_lengths, args.context_length),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
