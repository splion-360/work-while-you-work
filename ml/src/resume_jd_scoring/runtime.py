import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from resume_jd_scoring.embeddings import model_fingerprint
from resume_jd_scoring.hosted import HostedScoringRuntime
from resume_jd_scoring.inference import BinaryResumeJDScorer


def load_manifest(bundle_path: Path) -> dict[str, Any]:
    return json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))


def encoder_contract(max_length: int) -> dict[str, Any]:
    return {
        "pipeline_version": 1,
        "model_name": "BAAI/bge-m3",
        "representation": "normalized_colbert_token_vectors",
        "max_length": max_length,
        "cache_dtype": "float16",
        "excluded_token": "cls",
    }


def verify_encoder(model_path: Path, manifest: dict[str, Any]) -> None:
    fingerprint, _ = model_fingerprint(
        model_path,
        encoder_contract(int(manifest["encoder"]["max_length"])),
        additional_files=("colbert_linear.pt",),
    )
    if fingerprint != manifest["encoder"]["fingerprint"]:
        raise RuntimeError("BGE-M3 files do not match the classifier bundle")


def build_scoring_runtime(
    bundle_path: Path,
    model_path: Path,
    *,
    scorer_version: str,
    embedding_model: str,
    device: str,
    provenance: dict[str, str] | None = None,
    encoder_factory: Callable[..., Any] | None = None,
) -> HostedScoringRuntime:
    manifest = load_manifest(bundle_path)
    verify_encoder(model_path, manifest)
    if encoder_factory is None:
        from FlagEmbedding import BGEM3FlagModel

        encoder_factory = BGEM3FlagModel
    max_length = int(manifest["encoder"]["max_length"])
    encoder = encoder_factory(
        str(model_path),
        use_fp16=device.startswith("cuda"),
        devices=[device],
        batch_size=1,
        passage_max_length=max_length,
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True,
    )
    return HostedScoringRuntime(
        BinaryResumeJDScorer.from_bundle(bundle_path),
        encoder,
        scorer_version=scorer_version,
        embedding_model=embedding_model,
        max_length=max_length,
        pooling_device=device,
        provenance=provenance,
    )
