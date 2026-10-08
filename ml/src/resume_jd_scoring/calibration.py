from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryPlattCalibrator:
    slope: float
    intercept: float

    def predict(self, margins: np.ndarray) -> np.ndarray:
        return expit(self.slope * np.asarray(margins) + self.intercept)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BinaryThresholdSelection:
    threshold: float
    validation_macro_f1: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "objective": "maximum validation macro F1",
            "threshold": self.threshold,
            "validation_macro_f1": self.validation_macro_f1,
        }


def fit_binary_platt_calibrator(
    margins: np.ndarray, labels: np.ndarray
) -> BinaryPlattCalibrator:
    margins = np.asarray(margins, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if margins.ndim != 1 or labels.shape != margins.shape:
        raise ValueError("margins and labels must be aligned vectors")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("binary calibration requires both labels")
    model = LogisticRegression(C=1e6, solver="lbfgs").fit(margins[:, None], labels)
    return BinaryPlattCalibrator(
        slope=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )


def select_binary_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> BinaryThresholdSelection:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("labels and probabilities must be aligned vectors")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("binary threshold selection requires both labels")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values between zero and one")

    candidates = np.append(
        np.unique(probabilities), np.nextafter(probabilities.max(), np.inf)
    )
    best_threshold = 0.5
    best_macro_f1 = -1.0
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(np.int64)
        macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
        if macro_f1 > best_macro_f1 or (
            np.isclose(macro_f1, best_macro_f1) and threshold < best_threshold
        ):
            best_threshold = float(threshold)
            best_macro_f1 = macro_f1
    return BinaryThresholdSelection(best_threshold, best_macro_f1)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, boundaries[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            error += selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean())
    return float(error)


def binary_probability_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error_10_bins": expected_calibration_error(
            labels, probabilities, bins=10
        ),
    }
