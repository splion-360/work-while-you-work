import copy
import hashlib
import math
import os
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from resume_jd_scoring.model import ResumeJDLinearClassifier

LABELS = ("No Fit", "Potential Fit", "Good Fit")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
BINARY_LABELS = ("No Fit", "Good Fit")
LABEL_MODES = ("three-class", "binary-potential-positive")


def labels_for_mode(label_mode: str) -> tuple[str, ...]:
    if label_mode == "three-class":
        return LABELS
    if label_mode == "binary-potential-positive":
        return BINARY_LABELS
    raise ValueError(f"unknown label mode: {label_mode}")


def encode_labels(raw_labels: Sequence[str], *, label_mode: str) -> np.ndarray:
    label_names = labels_for_mode(label_mode)
    mapping = {label: index for index, label in enumerate(label_names)}
    if label_mode == "binary-potential-positive":
        mapping["Potential Fit"] = mapping["Good Fit"]
    try:
        return np.asarray([mapping[label] for label in raw_labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"unknown source label: {error.args[0]}") from error


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 5
    min_delta: float = 1e-4
    validation_fraction: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("batch size, epochs, and patience must be positive")
        if self.min_delta < 0:
            raise ValueError("minimum delta must be non-negative")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation fraction must be between zero and one")


@dataclass(frozen=True)
class InnerPartition:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    discarded_indices: tuple[int, ...]
    train_resume_clusters: tuple[str, ...]
    validation_resume_clusters: tuple[str, ...]
    train_jd_clusters: tuple[str, ...]
    validation_jd_clusters: tuple[str, ...]


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float


@dataclass(frozen=True)
class FoldTrainingResult:
    best_epoch: int
    best_validation_loss: float
    best_validation_accuracy: float
    best_validation_macro_f1: float
    history: tuple[EpochMetrics, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    class_weights: np.ndarray


def _selected_clusters(
    clusters: Sequence[str], validation_fraction: float, *, seed: int, namespace: str
) -> set[str]:
    unique = sorted(set(clusters))
    if len(unique) < 2:
        raise ValueError(f"{namespace} requires at least two clusters for validation")
    count = max(1, min(len(unique) - 1, round(len(unique) * validation_fraction)))
    ranked = sorted(
        unique,
        key=lambda cluster: hashlib.sha256(
            f"{seed}:{namespace}:{cluster}".encode()
        ).digest(),
    )
    return set(ranked[:count])


def build_inner_partition(
    rows: Sequence[Mapping[str, Any]],
    resume_clusters: Mapping[str, str],
    jd_clusters: Mapping[str, str],
    *,
    outer_fold: int,
    validation_fraction: float,
    seed: int,
) -> InnerPartition:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    role_column = f"fold_{outer_fold}_role"
    outer_train = [index for index, row in enumerate(rows) if row[role_column] == "train"]
    if not outer_train:
        raise ValueError(f"outer fold {outer_fold} has no training rows")

    resume_validation = _selected_clusters(
        [resume_clusters[str(rows[index]["resume_hash"])] for index in outer_train],
        validation_fraction,
        seed=seed + outer_fold,
        namespace="resume",
    )
    jd_validation = _selected_clusters(
        [jd_clusters[str(rows[index]["jd_hash"])] for index in outer_train],
        validation_fraction,
        seed=seed + outer_fold,
        namespace="jd",
    )
    train_indices = []
    validation_indices = []
    discarded_indices = []
    for index in outer_train:
        resume_cluster = resume_clusters[str(rows[index]["resume_hash"])]
        jd_cluster = jd_clusters[str(rows[index]["jd_hash"])]
        resume_held_out = resume_cluster in resume_validation
        jd_held_out = jd_cluster in jd_validation
        if resume_held_out and jd_held_out:
            validation_indices.append(index)
        elif not resume_held_out and not jd_held_out:
            train_indices.append(index)
        else:
            discarded_indices.append(index)
    if not train_indices or not validation_indices:
        raise ValueError(f"outer fold {outer_fold} produced an empty inner partition")

    train_resume = {
        resume_clusters[str(rows[index]["resume_hash"])] for index in train_indices
    }
    train_jds = {jd_clusters[str(rows[index]["jd_hash"])] for index in train_indices}
    return InnerPartition(
        train_indices=tuple(train_indices),
        validation_indices=tuple(validation_indices),
        discarded_indices=tuple(discarded_indices),
        train_resume_clusters=tuple(sorted(train_resume)),
        validation_resume_clusters=tuple(sorted(resume_validation)),
        train_jd_clusters=tuple(sorted(train_jds)),
        validation_jd_clusters=tuple(sorted(jd_validation)),
    )


def balanced_class_weights(labels: np.ndarray, *, class_count: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=class_count)
    if len(counts) != class_count or np.any(counts == 0):
        raise ValueError(f"every class must occur in training data: {counts.tolist()}")
    return (len(labels) / (class_count * counts)).astype(np.float32)


def set_reproducible_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _loader(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _run_epoch(
    model: ResumeJDLinearClassifier,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    *,
    device: str,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_weight = 0.0
    predictions = []
    targets = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for feature_batch, label_batch in loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(feature_batch)
            losses = criterion(logits, label_batch)
            batch_weight = criterion.weight[label_batch].sum()
            loss = losses.sum() / batch_weight
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(losses.detach().sum())
            total_weight += float(batch_weight)
            predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
            targets.append(label_batch.detach().cpu().numpy())
    return (
        total_loss / total_weight,
        np.concatenate(predictions),
        np.concatenate(targets),
    )


def fit_fold(
    model: ResumeJDLinearClassifier,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    config: TrainingConfig,
    device: str,
    class_count: int = len(LABELS),
    metric_callback: Callable[[EpochMetrics], None] | None = None,
) -> FoldTrainingResult:
    set_reproducible_seed(config.seed)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features).astype(np.float32)
    validation_scaled = scaler.transform(validation_features).astype(np.float32)
    weights = balanced_class_weights(train_labels, class_count=class_count)
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device), reduction="none"
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader = _loader(
        train_scaled,
        train_labels,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_loader = _loader(
        validation_scaled,
        validation_labels,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    best_loss = math.inf
    best_accuracy = 0.0
    best_f1 = 0.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_predictions, train_targets = _run_epoch(
            model, train_loader, criterion, device=device, optimizer=optimizer
        )
        validation_loss, predictions, targets = _run_epoch(
            model, validation_loader, criterion, device=device, optimizer=None
        )
        train_accuracy = float(np.mean(train_predictions == train_targets))
        validation_accuracy = float(np.mean(predictions == targets))
        macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            validation_macro_f1=macro_f1,
        )
        history.append(metrics)
        if metric_callback is not None:
            metric_callback(metrics)

        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_accuracy = validation_accuracy
            best_f1 = macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return FoldTrainingResult(
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        best_validation_accuracy=best_accuracy,
        best_validation_macro_f1=best_f1,
        history=tuple(history),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        class_weights=weights,
    )


def label_counts(
    labels: np.ndarray, *, label_names: tuple[str, ...] = LABELS
) -> dict[str, int]:
    counts = Counter(label_names[index] for index in labels)
    return {label: counts.get(label, 0) for label in label_names}
