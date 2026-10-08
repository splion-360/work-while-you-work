from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def multiclass_eta_squared(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the fraction of each feature's variance explained by class labels."""
    overall_mean = features.mean(axis=0)
    total = np.square(features - overall_mean).sum(axis=0)
    between = np.zeros(features.shape[1], dtype=np.float64)
    for label in np.unique(labels):
        selected = features[labels == label]
        between += len(selected) * np.square(selected.mean(axis=0) - overall_mean)
    return np.divide(between, total, out=np.zeros_like(between), where=total > 0)


def cross_set_neighbor_profile(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    *,
    neighbors: int,
    class_count: int,
) -> dict[str, np.ndarray | float]:
    """Measure class composition around held-out queries using training points only."""
    if neighbors <= 0 or neighbors > len(reference_features):
        raise ValueError("neighbors must be positive and no larger than the reference set")
    indices = NearestNeighbors(n_neighbors=neighbors).fit(reference_features).kneighbors(
        query_features, return_distance=False
    )
    neighbor_labels = reference_labels[indices]
    composition = np.zeros((class_count, class_count), dtype=np.float64)
    same_label_fraction = np.zeros(class_count, dtype=np.float64)
    for label in range(class_count):
        selected = neighbor_labels[query_labels == label]
        if not len(selected):
            continue
        counts = np.stack([(selected == candidate).mean(axis=1) for candidate in range(class_count)])
        composition[label] = counts.mean(axis=1)
        same_label_fraction[label] = composition[label, label]
    return {
        "overall_purity": float(np.mean(neighbor_labels == query_labels[:, None])),
        "same_label_fraction": same_label_fraction,
        "neighbor_class_composition": composition,
    }


def tsne_projection(
    features: np.ndarray,
    *,
    perplexity: float,
    seed: int,
    max_iter: int = 1_000,
) -> tuple[np.ndarray, float]:
    if len(features) <= perplexity:
        raise ValueError("t-SNE perplexity must be smaller than the sample count")
    source = features
    if features.shape[1] > 50:
        source = PCA(n_components=50, random_state=seed).fit_transform(features)
    projection = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
        random_state=seed,
    ).fit_transform(source)
    return projection, float(trustworthiness(source, projection, n_neighbors=10))


def knn_label_purity(features: np.ndarray, labels: np.ndarray, *, neighbors: int) -> float:
    if neighbors <= 0 or neighbors >= len(features):
        raise ValueError("neighbors must be positive and smaller than the sample count")
    indices = NearestNeighbors(n_neighbors=neighbors + 1).fit(features).kneighbors(
        return_distance=False
    )
    neighbor_labels = labels[indices[:, 1:]]
    return float(np.mean(neighbor_labels == labels[:, None]))


def cosine_separation(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if len(features) > sample_size:
        indices = rng.choice(len(features), size=sample_size, replace=False)
        features = features[indices]
        labels = labels[indices]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, np.finfo(np.float32).eps)
    similarities = normalized @ normalized.T
    upper = np.triu(np.ones(similarities.shape, dtype=bool), k=1)
    same_class = upper & (labels[:, None] == labels[None, :])
    different_class = upper & ~same_class
    intra = float(np.mean(similarities[same_class]))
    inter = float(np.mean(similarities[different_class]))
    return {
        "intra_class_cosine_similarity": intra,
        "inter_class_cosine_similarity": inter,
        "cosine_separation": intra - inter,
    }


def representation_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    neighbors: int,
    sample_size: int,
    seed: int,
) -> dict[str, float]:
    silhouette_sample = min(sample_size, len(features))
    return {
        "knn_label_purity": knn_label_purity(features, labels, neighbors=neighbors),
        "silhouette_score": float(
            silhouette_score(
                features,
                labels,
                sample_size=silhouette_sample,
                random_state=seed,
            )
        ),
        **cosine_separation(features, labels, sample_size=sample_size, seed=seed),
    }


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    label_names: tuple[str, ...],
) -> dict[str, Any]:
    indices = np.arange(len(label_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=indices,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(label_names)
        },
    }
