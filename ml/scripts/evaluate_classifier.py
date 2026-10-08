import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/matplotlib").resolve()))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pyarrow.parquet as pq
import torch
from mlflow.tracking import MlflowClient

from resume_jd_scoring.evaluate import (
    classification_metrics,
    representation_metrics,
    tsne_projection,
)
from resume_jd_scoring.model import ResumeJDLinearClassifier
from resume_jd_scoring.train import encode_labels, labels_for_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare held-out pair features before and after the linear classifier."
    )
    parser.add_argument(
        "--training-report", type=Path, default=Path("artifacts/train__baseline_classifier.json")
    )
    parser.add_argument(
        "--feature-report", type=Path, default=Path("artifacts/data__pair_features.json")
    )
    parser.add_argument("--fold", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--output-root", type=Path, default=Path("data/analysis"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/eval__representation.json"))
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--metric-sample-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_test_features(
    feature_path: Path, *, fold: int, label_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(feature_path)
    roles = np.asarray(table.column(f"fold_{fold}_role").to_pylist())
    test_indices = np.flatnonzero(roles == "test")
    vector_column = table.column("features").combine_chunks()
    dimension = vector_column.type.list_size
    all_features = vector_column.values.to_numpy(zero_copy_only=False).reshape(-1, dimension)
    all_labels = encode_labels(
        table.column("label").to_pylist(), label_mode=label_mode
    )
    return all_features[test_indices].astype(np.float32), all_labels[test_indices]


def plot_tsne(
    before: np.ndarray,
    after: np.ndarray,
    labels: np.ndarray,
    *,
    output_path: Path,
    before_label: str,
    label_names: tuple[str, ...],
) -> None:
    colors = ("#d1495b", "#ed9b40", "#2a9d8f")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, projection, title in zip(
        axes,
        (before, after),
        (f"Before training: standardized {before_label}", "After training: classifier logits"),
        strict=True,
    ):
        for label_index, label_name in enumerate(label_names):
            selected = labels == label_index
            axis.scatter(
                projection[selected, 0],
                projection[selected, 1],
                s=7,
                alpha=0.55,
                color=colors[label_index],
                label=label_name,
                linewidths=0,
            )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    axes[1].legend(frameon=False, markerscale=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def mlflow_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    for stage in ("before", "after"):
        for name, value in report[stage]["metrics"].items():
            metrics[f"representation_{stage}_{name}"] = value
        metrics[f"representation_{stage}_tsne_trustworthiness"] = report[stage][
            "tsne_trustworthiness"
        ]
    metrics["outer_test_accuracy"] = report["classification"]["accuracy"]
    metrics["outer_test_macro_f1"] = report["classification"]["macro_f1"]
    for label, values in report["classification"]["per_class"].items():
        key = label.lower().replace(" ", "_")
        metrics[f"outer_test_{key}_precision"] = values["precision"]
        metrics[f"outer_test_{key}_recall"] = values["recall"]
        metrics[f"outer_test_{key}_f1"] = values["f1"]
    return metrics


def main() -> None:
    args = parse_args()
    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    feature_report = json.loads(args.feature_report.read_text(encoding="utf-8"))
    fold_summary = next(
        (entry for entry in training_report["folds"] if int(entry["fold"]) == args.fold), None
    )
    if fold_summary is None:
        raise ValueError(f"training report has no checkpoint for fold {args.fold}")
    checkpoint_path = Path(fold_summary["checkpoint"]["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "linear":
        raise ValueError("representation analysis requires a linear-classifier checkpoint")

    label_mode = checkpoint.get("label_mode", "three-class")
    label_names = labels_for_mode(label_mode)
    if tuple(checkpoint["labels"]) != label_names:
        raise ValueError("checkpoint labels do not match its label mode")
    features, labels = load_test_features(
        Path(feature_report["output"]["path"]),
        fold=args.fold,
        label_mode=label_mode,
    )
    scaled = (features - checkpoint["scaler_mean"]) / checkpoint["scaler_scale"]
    model = ResumeJDLinearClassifier(
        input_dimension=int(checkpoint["input_dimension"]), class_count=len(label_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(scaled.astype(np.float32))).numpy()
    predictions = logits.argmax(axis=1)

    before_projection, before_trustworthiness = tsne_projection(
        scaled, perplexity=args.perplexity, seed=args.seed
    )
    after_projection, after_trustworthiness = tsne_projection(
        logits, perplexity=args.perplexity, seed=args.seed
    )
    output_dir = (
        args.output_root / training_report["parent_mlflow_run_id"] / f"fold_{args.fold}"
    )
    plot_path = output_dir / "tsne_before_after.png"
    representation_name = feature_report["feature_definition"]
    representation_label = feature_report.get("representation_label", representation_name)
    plot_tsne(
        before_projection,
        after_projection,
        labels,
        output_path=plot_path,
        before_label=representation_label,
        label_names=label_names,
    )

    report = {
        "fold": args.fold,
        "rows": len(labels),
        "checkpoint": str(checkpoint_path),
        "mlflow_run_id": fold_summary["mlflow_run_id"],
        "before": {
            "representation": f"standardized {representation_name}",
            "metrics": representation_metrics(
                scaled,
                labels,
                neighbors=args.neighbors,
                sample_size=args.metric_sample_size,
                seed=args.seed,
            ),
            "tsne_trustworthiness": before_trustworthiness,
        },
        "after": {
            "representation": f"{len(label_names)} classifier logits",
            "metrics": representation_metrics(
                logits,
                labels,
                neighbors=args.neighbors,
                sample_size=args.metric_sample_size,
                seed=args.seed,
            ),
            "tsne_trustworthiness": after_trustworthiness,
        },
        "label_mode": label_mode,
        "classification": classification_metrics(
            labels, predictions, label_names=label_names
        ),
        "tsne": {
            "perplexity": args.perplexity,
            "seed": args.seed,
            "plot": str(plot_path),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    mlflow.set_tracking_uri(training_report["tracking_uri"])
    client = MlflowClient()
    for name, value in mlflow_metrics(report).items():
        client.log_metric(fold_summary["mlflow_run_id"], name, value)
    client.log_artifact(fold_summary["mlflow_run_id"], str(plot_path), "representation")
    client.log_artifact(fold_summary["mlflow_run_id"], str(args.report), "representation")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
