import json

import numpy as np
import pytest
import torch

from resume_jd_scoring.calibration import BinaryPlattCalibrator
from resume_jd_scoring.embeddings import sha256_file
from resume_jd_scoring.inference import BinaryResumeJDScorer
from resume_jd_scoring.model import ResumeJDLinearClassifier
from resume_jd_scoring.multivector import DISTRIBUTION_STATISTICS, standard_knrm_kernels


def binary_checkpoint() -> dict:
    dimension = len(standard_knrm_kernels().means) * len(DISTRIBUTION_STATISTICS)
    model = ResumeJDLinearClassifier(dimension, class_count=2)
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.bias[:] = torch.tensor([0.0, 1.0])
    return {
        "label_mode": "binary-potential-positive",
        "labels": ("No Fit", "Good Fit"),
        "input_dimension": dimension,
        "model_state_dict": model.state_dict(),
        "scaler_mean": np.zeros(dimension, dtype=np.float32),
        "scaler_scale": np.ones(dimension, dtype=np.float32),
    }


def test_binary_scorer_returns_calibrated_fit_probability():
    scorer = BinaryResumeJDScorer(
        binary_checkpoint(), BinaryPlattCalibrator(slope=1.0, intercept=0.0)
    )
    resume = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    jd = np.asarray([[1.0, 0.0]], dtype=np.float32)

    result = scorer.score_token_vectors(resume, jd, pooling_device="cpu")

    assert result["probability"] == pytest.approx(0.7310586)
    assert result["score"] == pytest.approx(73.10586)
    assert result["label"] == "Good Fit"
    assert result["decision_threshold"] == 0.5


def test_binary_scorer_uses_configured_decision_threshold():
    scorer = BinaryResumeJDScorer(
        binary_checkpoint(),
        BinaryPlattCalibrator(slope=1.0, intercept=0.0),
        decision_threshold=0.8,
    )
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)

    result = scorer.score_token_vectors(vectors, vectors, pooling_device="cpu")

    assert result["probability"] == pytest.approx(0.7310586)
    assert result["label"] == "No Fit"


def test_binary_scorer_rejects_three_class_checkpoint():
    checkpoint = binary_checkpoint()
    checkpoint["label_mode"] = "three-class"

    with pytest.raises(ValueError, match="binary"):
        BinaryResumeJDScorer(checkpoint, BinaryPlattCalibrator(1.0, 0.0))


def test_binary_scorer_loads_bundle(tmp_path):
    checkpoint_path = tmp_path / "classifier.pt"
    torch.save(binary_checkpoint(), checkpoint_path)
    manifest = {
        "classifier": {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "input_dimension": 77,
        },
        "calibration": {"slope": 1.0, "intercept": 0.0},
        "decision_threshold": {"threshold": 0.4},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    scorer = BinaryResumeJDScorer.from_bundle(tmp_path)

    assert scorer.calibrator == BinaryPlattCalibrator(1.0, 0.0)
    assert scorer.decision_threshold == 0.4


def test_binary_scorer_rejects_tampered_bundle(tmp_path):
    checkpoint_path = tmp_path / "classifier.pt"
    torch.save(binary_checkpoint(), checkpoint_path)
    manifest = {
        "classifier": {
            "path": checkpoint_path.name,
            "sha256": "incorrect",
            "input_dimension": 77,
        },
        "calibration": {"slope": 1.0, "intercept": 0.0},
        "decision_threshold": {"threshold": 0.5},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        BinaryResumeJDScorer.from_bundle(tmp_path)
