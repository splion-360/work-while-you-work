import os

import numpy as np
import pytest

from resume_jd_scoring.model import ResumeJDLinearClassifier
from resume_jd_scoring.train import (
    BINARY_LABELS,
    LABELS,
    TrainingConfig,
    balanced_class_weights,
    build_inner_partition,
    encode_labels,
    fit_fold,
    set_reproducible_seed,
)


def test_binary_label_mode_merges_potential_and_good_fit():
    labels = encode_labels(
        ["No Fit", "Potential Fit", "Good Fit"],
        label_mode="binary-potential-positive",
    )

    assert BINARY_LABELS == ("No Fit", "Good Fit")
    assert labels.tolist() == [0, 1, 1]


def test_reproducible_seed_configures_deterministic_cublas_workspace(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    set_reproducible_seed(42)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_inner_partition_is_deterministic_cluster_disjoint_and_outer_train_only():
    rows = [
        {
            "resume_hash": f"r{resume}",
            "jd_hash": f"j{jd}",
            "fold_0_role": "train" if resume < 4 and jd < 4 else "test",
        }
        for resume in range(5)
        for jd in range(5)
    ]
    resume_clusters = {f"r{i}": f"rc{i // 2}" for i in range(5)}
    jd_clusters = {f"j{i}": f"jc{i // 2}" for i in range(5)}

    first = build_inner_partition(
        rows,
        resume_clusters,
        jd_clusters,
        outer_fold=0,
        validation_fraction=0.34,
        seed=42,
    )
    second = build_inner_partition(
        rows,
        resume_clusters,
        jd_clusters,
        outer_fold=0,
        validation_fraction=0.34,
        seed=42,
    )

    assert first == second
    assert first.train_indices
    assert first.validation_indices
    assert not set(first.train_resume_clusters) & set(first.validation_resume_clusters)
    assert not set(first.train_jd_clusters) & set(first.validation_jd_clusters)
    assert all(rows[index]["fold_0_role"] == "train" for index in first.train_indices)
    assert all(rows[index]["fold_0_role"] == "train" for index in first.validation_indices)


def test_balanced_class_weights_upweight_rare_classes():
    encoded = np.asarray([0, 0, 0, 1, 1, 2])

    weights = balanced_class_weights(encoded, class_count=3)

    assert weights.tolist() == pytest.approx([2 / 3, 1.0, 2.0])


def test_fit_fold_trains_and_restores_best_checkpoint_on_cpu():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(90, 8)).astype(np.float32)
    labels = np.asarray([index % len(LABELS) for index in range(90)], dtype=np.int64)
    features[:, 0] += labels * 2
    model = ResumeJDLinearClassifier(input_dimension=8)
    config = TrainingConfig(
        learning_rate=1e-2,
        weight_decay=1e-4,
        batch_size=16,
        max_epochs=5,
        patience=3,
        min_delta=0.0,
        validation_fraction=0.2,
        seed=42,
    )

    result = fit_fold(
        model,
        features[:72],
        labels[:72],
        features[72:],
        labels[72:],
        config=config,
        device="cpu",
    )

    assert 1 <= result.best_epoch <= 5
    assert len(result.history) <= 5
    assert np.isfinite(result.best_validation_loss)
    assert 0.0 <= result.best_validation_accuracy <= 1.0
    assert 0.0 <= result.best_validation_macro_f1 <= 1.0
    assert all(0.0 <= metrics.train_accuracy <= 1.0 for metrics in result.history)
    assert all(0.0 <= metrics.validation_accuracy <= 1.0 for metrics in result.history)
    assert (
        result.best_validation_accuracy
        == result.history[result.best_epoch - 1].validation_accuracy
    )
    assert result.scaler_mean.shape == (8,)
