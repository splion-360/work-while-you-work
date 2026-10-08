import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from resume_jd_scoring.embeddings import model_fingerprint, sha256_file
from resume_jd_scoring.multivector import validate_normalized_token_vectors

DEFAULT_MODEL_DIR = Path("data/models/bge-m3")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT_ROOT = Path("data/multivector")
DEFAULT_REPORT = Path("artifacts/model__bge_m3_multivector_generation.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen BGE-M3 ColBERT token vectors.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_documents(path: Path, hash_column: str) -> list[tuple[str, str]]:
    table = pq.read_table(path, columns=[hash_column, "text"])
    return list(zip(table.column(hash_column).to_pylist(), table.column("text").to_pylist()))


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def token_count(tokenizer, text: str, *, max_length: int) -> int:
    encoded = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
    return len(encoded["input_ids"]) - 1


def batches(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def encode_corpus(
    model,
    documents: list[tuple[str, str]],
    output_dir: Path,
    corpus_name: str,
    *,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    measured = [
        (document_hash, text, token_count(model.tokenizer, text, max_length=max_length))
        for document_hash, text in documents
    ]
    measured.sort(key=lambda row: (-row[2], row[0]))
    if any(count <= 0 for _, _, count in measured):
        raise ValueError(f"{corpus_name} contains a document with no ColBERT tokens")

    first_batch = measured[:batch_size]
    first_output = model.encode(
        [text for _, text, _ in first_batch],
        batch_size=len(first_batch),
        max_length=max_length,
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True,
    )["colbert_vecs"]
    dimension = int(first_output[0].shape[1])
    total_tokens = sum(count for _, _, count in measured)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = output_dir / f"{corpus_name}_vectors.npy"
    index_path = output_dir / f"{corpus_name}_index.parquet"
    vectors = np.lib.format.open_memmap(
        vectors_path,
        mode="w+",
        dtype=np.float16,
        shape=(total_tokens, dimension),
    )
    index_rows = []
    offset = 0

    def write_batch(rows, encoded_vectors) -> None:
        nonlocal offset
        if len(rows) != len(encoded_vectors):
            raise ValueError("BGE-M3 returned an unexpected number of documents")
        for (document_hash, _, expected_tokens), document_vectors in zip(
            rows, encoded_vectors, strict=True
        ):
            document_vectors = np.asarray(document_vectors, dtype=np.float32)
            if document_vectors.shape != (expected_tokens, dimension):
                raise ValueError(
                    f"unexpected token-vector shape for {document_hash}: "
                    f"{document_vectors.shape}, expected {(expected_tokens, dimension)}"
                )
            validate_normalized_token_vectors(document_vectors)
            end = offset + expected_tokens
            vectors[offset:end] = document_vectors.astype(np.float16)
            index_rows.append(
                {
                    "document_hash": document_hash,
                    "offset": offset,
                    "token_count": expected_tokens,
                }
            )
            offset = end

    write_batch(first_batch, first_output)
    remaining = measured[len(first_batch) :]
    with tqdm(total=len(measured), initial=len(first_batch), desc=corpus_name, unit="document") as bar:
        for rows in batches(remaining, batch_size):
            output = model.encode(
                [text for _, text, _ in rows],
                batch_size=len(rows),
                max_length=max_length,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )["colbert_vecs"]
            write_batch(rows, output)
            bar.update(len(rows))
    if offset != total_tokens:
        raise ValueError(f"{corpus_name} token cache ended at {offset}, expected {total_tokens}")
    vectors.flush()
    pq.write_table(
        pa.table(
            {
                "document_hash": [row["document_hash"] for row in index_rows],
                "offset": pa.array([row["offset"] for row in index_rows], type=pa.int64()),
                "token_count": pa.array(
                    [row["token_count"] for row in index_rows], type=pa.int32()
                ),
            }
        ),
        index_path,
        compression="zstd",
    )
    return {
        "documents": len(measured),
        "tokens": total_tokens,
        "minimum_tokens": min(count for _, _, count in measured),
        "maximum_tokens": max(count for _, _, count in measured),
        "dimension": dimension,
        "dtype": "float16",
        "vectors": file_record(vectors_path),
        "index": file_record(index_path),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 1:
        raise ValueError("batch size must be positive and max length must exceed one")
    import torch
    from FlagEmbedding import BGEM3FlagModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")
    encoding = {
        "pipeline_version": 1,
        "model_name": "BAAI/bge-m3",
        "representation": "normalized_colbert_token_vectors",
        "max_length": args.max_length,
        "cache_dtype": "float16",
        "excluded_token": "cls",
    }
    fingerprint, model_files = model_fingerprint(
        args.model_dir,
        encoding,
        additional_files=("colbert_linear.pt",),
    )
    output_dir = args.output_root / fingerprint
    model = BGEM3FlagModel(
        str(args.model_dir),
        use_fp16=True,
        devices=[args.device],
        batch_size=args.batch_size,
        passage_max_length=args.max_length,
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True,
    )

    started = time.perf_counter()
    resumes = encode_corpus(
        model,
        load_documents(args.processed_dir / "resumes.parquet", "resume_hash"),
        output_dir,
        "resumes",
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    resume_seconds = time.perf_counter() - started
    started = time.perf_counter()
    jds = encode_corpus(
        model,
        load_documents(args.processed_dir / "jds.parquet", "jd_hash"),
        output_dir,
        "jds",
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    jd_seconds = time.perf_counter() - started
    if resumes["dimension"] != jds["dimension"]:
        raise ValueError("resume and JD token-vector dimensions differ")

    report = {
        "fingerprint": fingerprint,
        "model": {
            "name": "BAAI/bge-m3",
            "path": str(args.model_dir),
            "files": model_files,
            "frozen": True,
            "inference_dtype": "float16",
            "token_vector_dimension": resumes["dimension"],
        },
        "encoding": encoding,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "batch_size": args.batch_size,
            "resume_seconds": round(resume_seconds, 3),
            "jd_seconds": round(jd_seconds, 3),
        },
        "outputs": {"resumes": resumes, "jds": jds},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
