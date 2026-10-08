import numpy as np
import pytest

from resume_jd_scoring.evaluate import (
    classification_metrics,
    cosine_separation,
    knn_label_purity,
    representation_metrics,
    tsne_projection,
)


def clustered_features():
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(3), 12)
    features = rng.normal(scale=0.05, size=(36, 6)).astype(np.float32)
    features[:, 0] += labels * 3
    return features, labels


def test_representation_metrics_detect_separated_classes():
    features, labels = clustered_features()

    metrics = representation_metrics(features, labels, neighbors=3, sample_size=36, seed=42)

    assert metrics["knn_label_purity"] == 1.0
    assert metrics["silhouette_score"] > 0.9
    assert metrics["cosine_separation"] > 0


def test_tsne_projection_returns_two_dimensions_and_trustworthiness():
    features, _ = clustered_features()

    projection, score = tsne_projection(features, perplexity=5, seed=42, max_iter=250)

    assert projection.shape == (36, 2)
    assert 0 <= score <= 1


def test_classification_metrics_reports_each_class():
    labels = np.asarray([0, 1, 2, 2])

    metrics = classification_metrics(labels, labels, label_names=("a", "b", "c"))

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["c"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 2,
    }


@pytest.mark.parametrize("neighbors", [0, 36])
def test_knn_purity_rejects_invalid_neighbor_counts(neighbors):
    features, labels = clustered_features()

    with pytest.raises(ValueError):
        knn_label_purity(features, labels, neighbors=neighbors)


def test_cosine_separation_is_reproducible_when_sampled():
    features, labels = clustered_features()

    first = cosine_separation(features, labels, sample_size=18, seed=42)
    second = cosine_separation(features, labels, sample_size=18, seed=42)

    assert first == second
