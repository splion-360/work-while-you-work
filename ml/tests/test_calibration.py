import numpy as np
import pytest

from resume_jd_scoring.calibration import (
    binary_probability_metrics,
    expected_calibration_error,
    fit_binary_platt_calibrator,
    select_binary_threshold,
)


def test_platt_calibrator_learns_monotonic_binary_probability():
    margins = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    labels = np.asarray([0, 0, 0, 1, 1, 1])

    calibrator = fit_binary_platt_calibrator(margins, labels)
    probabilities = calibrator.predict(margins)

    assert calibrator.slope > 0
    assert np.all(np.diff(probabilities) > 0)
    assert set(calibrator.to_dict()) == {"slope", "intercept"}


def test_probability_metrics_are_exact_for_balanced_confident_predictions():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])

    metrics = binary_probability_metrics(labels, probabilities)

    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["brier_score"] == 0.0
    assert metrics["expected_calibration_error_10_bins"] == 0.0


def test_calibration_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="both labels"):
        fit_binary_platt_calibrator(np.asarray([0.0, 1.0]), np.asarray([1, 1]))
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error(np.asarray([0]), np.asarray([0.5]), bins=0)


def test_threshold_selection_maximizes_validation_macro_f1():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.45, 0.8])

    selection = select_binary_threshold(labels, probabilities)

    assert selection.threshold == pytest.approx(0.45)
    assert selection.validation_macro_f1 == 1.0
