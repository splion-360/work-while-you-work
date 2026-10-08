from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import pytest

from resume_jd_scoring.embeddings import (
    EMBEDDINGGEMMA_FINGERPRINT_FILES,
    DocumentEmbedder,
    TokenChunker,
    build_pair_features,
    model_fingerprint,
    validate_embedding_rows,
    write_embedding_cache,
    write_pair_features,
)


@dataclass
class WordTokenizer:
    model_max_length: int = 8

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ):
        words = text.split()
        offsets = []
        cursor = 0
        for word in words:
            start = text.index(word, cursor)
            offsets.append((start, start + len(word)))
            cursor = start + len(word)
        ids = list(range(10, 10 + len(words)))
        if add_special_tokens and words:
            ids = [1, *ids, 2]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert not pair
        return 2


class FakeModel:
    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return np.asarray(
            [[float(len(text.split())), float(index + 1)] for index, text in enumerate(texts)],
            dtype=np.float32,
        )


def test_chunker_keeps_short_document_unchanged():
    chunker = TokenChunker(WordTokenizer(), max_length=8, overlap_tokens=2, prompt="task ")

    chunks = chunker.chunk("one two three")

    assert [chunk.text for chunk in chunks] == ["one two three"]
    assert chunks[0].token_start == 0
    assert chunks[0].token_end == 3


def test_chunker_is_deterministic_and_covers_long_document():
    chunker = TokenChunker(WordTokenizer(), max_length=8, overlap_tokens=2, prompt="task ")
    text = "zero one two three four five six seven eight"

    first = chunker.chunk(text)
    second = chunker.chunk(text)

    assert first == second
    assert [(chunk.token_start, chunk.token_end) for chunk in first] == [(0, 5), (3, 8), (6, 9)]
    assert all(chunk.encoded_tokens <= 8 for chunk in first)
    covered = {index for chunk in first for index in range(chunk.token_start, chunk.token_end)}
    assert covered == set(range(9))


def test_chunker_handles_exact_capacity_and_one_token_over():
    chunker = TokenChunker(WordTokenizer(), max_length=8, overlap_tokens=2, prompt="task ")

    exact = chunker.chunk("zero one two three four")
    overflow = chunker.chunk("zero one two three four five")

    assert len(exact) == 1
    assert [(chunk.token_start, chunk.token_end) for chunk in overflow] == [(0, 5), (3, 6)]


def test_chunker_rejects_empty_documents_and_invalid_overlap():
    with pytest.raises(ValueError, match="empty"):
        TokenChunker(WordTokenizer(), max_length=8, overlap_tokens=2, prompt="task ").chunk("")
    with pytest.raises(ValueError, match="overlap"):
        TokenChunker(WordTokenizer(), max_length=8, overlap_tokens=5, prompt="task ")


def test_document_embedder_mean_pools_then_normalizes():
    chunker = TokenChunker(WordTokenizer(), max_length=8, overlap_tokens=2, prompt="task ")
    embedder = DocumentEmbedder(FakeModel(), chunker, prompt="task ", batch_size=2)

    row = embedder.embed("resume", "zero one two three four five six seven eight")

    expected = np.asarray([13.0 / 3.0, 2.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert row["document_hash"] == "resume"
    assert row["token_count"] == 9
    assert row["chunk_count"] == 3
    np.testing.assert_allclose(row["embedding"], expected, rtol=1e-6)


def test_embedding_cache_and_pair_features_round_trip(tmp_path):
    resume_rows = [
        {
            "document_hash": "r1",
            "token_count": 3,
            "chunk_count": 1,
            "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        }
    ]
    jd_rows = [
        {
            "document_hash": "j1",
            "token_count": 2,
            "chunk_count": 1,
            "embedding": np.asarray([0.25, 0.5], dtype=np.float32),
        }
    ]
    pair_rows = [
        {
            "pair_hash": "p1",
            "resume_hash": "r1",
            "jd_hash": "j1",
            "label": "Good Fit",
            "fold_0_role": "test",
            "fold_1_role": "train",
            "fold_2_role": "discarded",
        }
    ]

    assert validate_embedding_rows(resume_rows, expected_dimension=2)["all_unit_norm"]
    resume_path = tmp_path / "resumes.parquet"
    write_embedding_cache(resume_rows, resume_path, dimension=2)
    loaded = pq.read_table(resume_path).to_pylist()
    assert loaded[0]["embedding"] == [1.0, 0.0]

    features = build_pair_features(pair_rows, resume_rows, jd_rows)
    assert features[0]["features"] == pytest.approx([0.75, -0.5])
    assert features[0]["fold_1_role"] == "train"
    feature_path = tmp_path / "pairs.parquet"
    write_pair_features(features, feature_path, dimension=2)
    assert pq.read_table(feature_path).to_pylist()[0]["features"] == pytest.approx([0.75, -0.5])


def test_pair_features_reject_missing_embeddings():
    pairs = [{"pair_hash": "p", "resume_hash": "missing", "jd_hash": "j", "label": "No Fit"}]
    jds = [{"document_hash": "j", "embedding": np.asarray([1.0], dtype=np.float32)}]

    with pytest.raises(ValueError, match="missing resume"):
        build_pair_features(pairs, [], jds)


def test_model_fingerprint_changes_with_inference_file_or_encoding(tmp_path):
    for relative in ("model.safetensors", "config.json", "tokenizer.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    original, _ = model_fingerprint(tmp_path, {"overlap": 256})
    changed_encoding, _ = model_fingerprint(tmp_path, {"overlap": 128})
    (tmp_path / "config.json").write_text("changed", encoding="utf-8")
    changed_file, _ = model_fingerprint(tmp_path, {"overlap": 256})

    assert original != changed_encoding
    assert original != changed_file


def test_model_fingerprint_supports_different_checkpoint_layouts(tmp_path):
    (tmp_path / "model.safetensors").write_text("weights", encoding="utf-8")
    (tmp_path / "config.json").write_text("config", encoding="utf-8")
    (tmp_path / "sentencepiece.bpe.model").write_text("tokenizer", encoding="utf-8")
    (tmp_path / "README.md").write_text("documentation", encoding="utf-8")
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "config.json").write_text("unused export", encoding="utf-8")

    fingerprint, files = model_fingerprint(tmp_path, {"context_length": 8192})

    assert len(fingerprint) == 64
    assert set(files) == {"config.json", "model.safetensors", "sentencepiece.bpe.model"}


def test_model_fingerprint_preserves_embeddinggemma_file_contract(tmp_path):
    for relative in EMBEDDINGGEMMA_FINGERPRINT_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    (tmp_path / "generation_config.json").write_text("not in contract", encoding="utf-8")

    _, files = model_fingerprint(tmp_path, {"pipeline_version": 1})

    assert set(files) == set(EMBEDDINGGEMMA_FINGERPRINT_FILES)


def test_model_fingerprint_includes_required_additional_file(tmp_path):
    for relative in ("model.safetensors", "config.json", "tokenizer.json"):
        (tmp_path / relative).write_text(relative, encoding="utf-8")
    projection = tmp_path / "colbert_linear.pt"
    projection.write_text("projection-v1", encoding="utf-8")

    first, files = model_fingerprint(
        tmp_path,
        {"representation": "colbert"},
        additional_files=("colbert_linear.pt",),
    )
    projection.write_text("projection-v2", encoding="utf-8")
    second, _ = model_fingerprint(
        tmp_path,
        {"representation": "colbert"},
        additional_files=("colbert_linear.pt",),
    )

    assert "colbert_linear.pt" in files
    assert first != second
