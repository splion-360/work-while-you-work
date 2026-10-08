import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from resume_jd_scoring.calibration import BinaryPlattCalibrator
from resume_jd_scoring.embeddings import sha256_file
from resume_jd_scoring.model import ResumeJDLinearClassifier
from resume_jd_scoring.multivector import (
    DISTRIBUTION_STATISTICS,
    distributional_knrm_pool_similarity,
    standard_knrm_kernels,
    validate_normalized_token_vectors,
)


class MultivectorEncoder(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]: ...


class BinaryResumeJDScorer:
    def __init__(
        self,
        checkpoint: dict,
        calibrator: BinaryPlattCalibrator,
        decision_threshold: float = 0.5,
    ) -> None:
        if checkpoint.get("label_mode") != "binary-potential-positive":
            raise ValueError("scorer requires the binary Potential Fit-positive checkpoint")
        if tuple(checkpoint.get("labels", ())) != ("No Fit", "Good Fit"):
            raise ValueError("checkpoint labels are incompatible with binary scoring")
        expected_dimension = len(standard_knrm_kernels().means) * len(DISTRIBUTION_STATISTICS)
        if int(checkpoint["input_dimension"]) != expected_dimension:
            raise ValueError("checkpoint dimension is incompatible with row-distribution pooling")
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("decision threshold must be between zero and one")
        self.checkpoint = checkpoint
        self.calibrator = calibrator
        self.decision_threshold = decision_threshold
        self.model = ResumeJDLinearClassifier(expected_dimension, class_count=2)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        calibrator: BinaryPlattCalibrator,
        decision_threshold: float = 0.5,
    ) -> "BinaryResumeJDScorer":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls(checkpoint, calibrator, decision_threshold)

    @classmethod
    def from_bundle(cls, bundle_path: Path) -> "BinaryResumeJDScorer":
        manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
        classifier = manifest["classifier"]
        checkpoint_path = bundle_path / classifier["path"]
        if sha256_file(checkpoint_path) != classifier["sha256"]:
            raise ValueError("classifier checkpoint hash does not match the bundle manifest")
        calibrator = BinaryPlattCalibrator(
            slope=float(manifest["calibration"]["slope"]),
            intercept=float(manifest["calibration"]["intercept"]),
        )
        scorer = cls.from_checkpoint(
            checkpoint_path,
            calibrator,
            float(manifest["decision_threshold"]["threshold"]),
        )
        if int(classifier["input_dimension"]) != int(scorer.checkpoint["input_dimension"]):
            raise ValueError("classifier dimension does not match the bundle manifest")
        return scorer

    def score_token_vectors(
        self,
        resume_vectors: np.ndarray,
        jd_vectors: np.ndarray,
        *,
        pooling_device: str,
        resume_block_size: int = 4096,
    ) -> dict[str, float | str]:
        validate_normalized_token_vectors(resume_vectors)
        validate_normalized_token_vectors(jd_vectors)
        features = distributional_knrm_pool_similarity(
            jd_vectors,
            resume_vectors,
            standard_knrm_kernels(),
            device=pooling_device,
            resume_block_size=resume_block_size,
        )
        scaled = (features - self.checkpoint["scaler_mean"]) / self.checkpoint["scaler_scale"]
        with torch.no_grad():
            logits = self.model(torch.from_numpy(scaled.astype(np.float32))[None, :])[0]
        margin = float(logits[1] - logits[0])
        probability = float(self.calibrator.predict(np.asarray([margin]))[0])
        return {
            "score": 100.0 * probability,
            "probability": probability,
            "label": "Good Fit" if probability >= self.decision_threshold else "No Fit",
            "decision_threshold": self.decision_threshold,
        }

    def score_texts(
        self,
        encoder: MultivectorEncoder,
        resume_text: str,
        jd_text: str,
        *,
        pooling_device: str,
        max_length: int = 8192,
    ) -> dict[str, float | str]:
        if not resume_text.strip() or not jd_text.strip():
            raise ValueError("resume and JD text must be non-empty")

        def encode(text: str) -> np.ndarray:
            output = encoder.encode(
                [text],
                batch_size=1,
                max_length=max_length,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )["colbert_vecs"]
            if len(output) != 1:
                raise ValueError("BGE-M3 returned an unexpected number of token matrices")
            return np.asarray(output[0], dtype=np.float32)

        return self.score_token_vectors(
            encode(resume_text),
            encode(jd_text),
            pooling_device=pooling_device,
        )
