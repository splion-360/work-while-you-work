import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from resume_jd_scoring.token_lengths import PretrainedTokenizer

EMBEDDINGGEMMA_FINGERPRINT_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "3_Dense/config.json",
    "3_Dense/model.safetensors",
)
MODEL_FINGERPRINT_SUFFIXES = (".bin", ".json", ".model", ".safetensors", ".txt")
MODEL_FINGERPRINT_EXCLUDED_DIRECTORIES = {"onnx"}


class EmbeddingModel(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class TokenChunk:
    text: str
    token_start: int
    token_end: int
    encoded_tokens: int


class TokenChunker:
    def __init__(
        self,
        tokenizer: PretrainedTokenizer,
        *,
        max_length: int,
        overlap_tokens: int,
        prompt: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap_tokens = overlap_tokens
        self.prompt = prompt
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        self.content_capacity = max_length - prompt_tokens - special_tokens
        if self.content_capacity <= 0:
            raise ValueError("prompt and special tokens exhaust the context window")
        if overlap_tokens < 0 or overlap_tokens >= self.content_capacity:
            raise ValueError("overlap must be non-negative and smaller than content capacity")

    def _encoded_length(self, text: str) -> int:
        return len(self.tokenizer(f"{self.prompt}{text}", add_special_tokens=True)["input_ids"])

    def chunk(self, text: str) -> list[TokenChunk]:
        if not text or not text.strip():
            raise ValueError("cannot embed an empty document")
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if not token_ids:
            raise ValueError("cannot embed a document with no tokens")

        if len(token_ids) <= self.content_capacity and self._encoded_length(text) <= self.max_length:
            return [TokenChunk(text, 0, len(token_ids), self._encoded_length(text))]

        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + self.content_capacity, len(token_ids))
            chunk_text = text[offsets[start][0] : offsets[end - 1][1]]
            encoded_tokens = self._encoded_length(chunk_text)
            while encoded_tokens > self.max_length and end > start + 1:
                end -= 1
                chunk_text = text[offsets[start][0] : offsets[end - 1][1]]
                encoded_tokens = self._encoded_length(chunk_text)
            if encoded_tokens > self.max_length:
                raise ValueError("a single source token cannot fit in the context window")
            chunks.append(TokenChunk(chunk_text, start, end, encoded_tokens))
            if end == len(token_ids):
                break
            next_start = end - self.overlap_tokens
            if next_start <= start:
                raise RuntimeError("chunking failed to make progress")
            start = next_start
        return chunks


class DocumentEmbedder:
    def __init__(
        self,
        model: EmbeddingModel,
        chunker: TokenChunker,
        *,
        prompt: str,
        batch_size: int,
    ) -> None:
        self.model = model
        self.chunker = chunker
        self.prompt = prompt
        self.batch_size = batch_size

    def embed(self, document_hash: str, text: str) -> dict[str, Any]:
        return self.embed_documents({document_hash: text})[0]

    def embed_documents(self, documents: Mapping[str, str]) -> list[dict[str, Any]]:
        chunk_rows: list[tuple[str, TokenChunk]] = []
        token_counts: dict[str, int] = {}
        for document_hash, text in sorted(documents.items()):
            chunks = self.chunker.chunk(text)
            chunk_rows.extend((document_hash, chunk) for chunk in chunks)
            token_counts[document_hash] = len(
                self.chunker.tokenizer(text, add_special_tokens=False)["input_ids"]
            )

        encode_options = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": True,
        }
        if self.prompt:
            encode_options["prompt"] = self.prompt
        vectors = np.asarray(
            self.model.encode(
                [chunk.text for _, chunk in chunk_rows],
                **encode_options,
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(chunk_rows):
            raise ValueError("embedding model returned an unexpected shape")

        grouped: dict[str, list[np.ndarray]] = {document_hash: [] for document_hash in documents}
        for (document_hash, _), vector in zip(chunk_rows, vectors, strict=True):
            grouped[document_hash].append(vector)

        rows = []
        for document_hash in sorted(documents):
            document_vector = np.mean(grouped[document_hash], axis=0, dtype=np.float32)
            norm = float(np.linalg.norm(document_vector))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError(f"invalid pooled embedding for {document_hash}")
            document_vector /= norm
            rows.append(
                {
                    "document_hash": document_hash,
                    "token_count": token_counts[document_hash],
                    "chunk_count": len(grouped[document_hash]),
                    "embedding": document_vector,
                }
            )
        return rows


def validate_embedding_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_dimension: int
) -> dict[str, Any]:
    hashes = [str(row["document_hash"]) for row in rows]
    vectors = [np.asarray(row["embedding"], dtype=np.float32) for row in rows]
    dimensions = [vector.shape == (expected_dimension,) for vector in vectors]
    finite = [bool(np.isfinite(vector).all()) for vector in vectors]
    unit_norm = [bool(np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)) for vector in vectors]
    report = {
        "rows": len(rows),
        "unique_hashes": len(set(hashes)),
        "all_hashes_unique": len(set(hashes)) == len(hashes),
        "expected_dimension": expected_dimension,
        "all_dimensions_valid": all(dimensions),
        "all_finite": all(finite),
        "all_unit_norm": all(unit_norm),
        "chunk_counts": dict(sorted(Counter(int(row["chunk_count"]) for row in rows).items())),
    }
    if not all(
        report[key]
        for key in ("all_hashes_unique", "all_dimensions_valid", "all_finite", "all_unit_norm")
    ):
        raise ValueError(f"embedding validation failed: {report}")
    return report


def _fixed_vectors(values: Sequence[Sequence[float] | np.ndarray], dimension: int) -> pa.Array:
    return pa.array(
        [np.asarray(value, dtype=np.float32).tolist() for value in values],
        type=pa.list_(pa.float32(), dimension),
    )


def write_embedding_cache(
    rows: Sequence[Mapping[str, Any]], path: Path, *, dimension: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "document_hash": [row["document_hash"] for row in rows],
            "token_count": pa.array([row["token_count"] for row in rows], type=pa.int32()),
            "chunk_count": pa.array([row["chunk_count"] for row in rows], type=pa.int16()),
            "embedding": _fixed_vectors([row["embedding"] for row in rows], dimension),
        }
    )
    pq.write_table(table, path, compression="zstd", write_statistics=True)


def read_embedding_cache(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def build_pair_features(
    pair_rows: Sequence[Mapping[str, Any]],
    resume_rows: Sequence[Mapping[str, Any]],
    jd_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    resumes = {str(row["document_hash"]): np.asarray(row["embedding"]) for row in resume_rows}
    jds = {str(row["document_hash"]): np.asarray(row["embedding"]) for row in jd_rows}
    features = []
    for pair in pair_rows:
        resume_hash = str(pair["resume_hash"])
        jd_hash = str(pair["jd_hash"])
        if resume_hash not in resumes:
            raise ValueError(f"missing resume embedding: {resume_hash}")
        if jd_hash not in jds:
            raise ValueError(f"missing JD embedding: {jd_hash}")
        delta = resumes[resume_hash] - jds[jd_hash]
        features.append({**pair, "features": delta.astype(np.float32).tolist()})
    return features


def write_pair_features(
    rows: Sequence[Mapping[str, Any]], path: Path, *, dimension: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {key: [row[key] for row in rows] for key in rows[0] if key != "features"}
    columns["features"] = _fixed_vectors([row["features"] for row in rows], dimension)
    pq.write_table(pa.table(columns), path, compression="zstd", write_statistics=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(
    model_dir: Path,
    encoding: Mapping[str, Any],
    *,
    additional_files: Sequence[str] = (),
) -> tuple[str, dict[str, str]]:
    embeddinggemma_files = [model_dir / relative for relative in EMBEDDINGGEMMA_FINGERPRINT_FILES]
    if all(path.is_file() for path in embeddinggemma_files):
        model_files = embeddinggemma_files
    else:
        model_files = sorted(
            path
            for path in model_dir.rglob("*")
            if path.is_file()
            and path.suffix in MODEL_FINGERPRINT_SUFFIXES
            and not any(part.startswith(".") for part in path.relative_to(model_dir).parts)
            and not MODEL_FINGERPRINT_EXCLUDED_DIRECTORIES.intersection(
                path.relative_to(model_dir).parts
            )
        )
    known_paths = {path.resolve() for path in model_files}
    for relative in additional_files:
        path = model_dir / relative
        if not path.is_file():
            raise ValueError(f"required model file does not exist: {path}")
        if path.resolve() not in known_paths:
            model_files.append(path)
            known_paths.add(path.resolve())
    model_files = sorted(model_files)
    if not model_files:
        raise ValueError(f"no inference files found in model directory: {model_dir}")
    file_hashes = {
        str(path.relative_to(model_dir)): sha256_file(path) for path in model_files
    }
    payload = json.dumps(
        {"files": file_hashes, "encoding": dict(encoding)}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest(), file_hashes
