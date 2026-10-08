import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from mlflow.tracking import MlflowClient
from scipy.special import expit

from resume_jd_scoring.calibration import (
    binary_probability_metrics,
    fit_binary_platt_calibrator,
    select_binary_threshold,
)
from resume_jd_scoring.evaluate import classification_metrics
from resume_jd_scoring.model import ResumeJDLinearClassifier
from resume_jd_scoring.train import (
    build_inner_partition,
    encode_labels,
    labels_for_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit validation-only Platt calibration for binary fold classifiers."
    )
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path, default=Path("data/processed/resumes.parquet"))
    parser.add_argument("--jd-manifest", type=Path, default=Path("data/processed/jds.parquet"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/eval__binary_calibration.json"),
    )
    return parser.parse_args()


def matrix_from_table(table: pa.Table) -> np.ndarray:
    column = table.column("features").combine_chunks()
    return column.values.to_numpy(zero_copy_only=False).reshape(-1, column.type.list_size)


def load_clusters(path: Path, hash_column: str) -> dict[str, str]:
    table = pq.read_table(path, columns=[hash_column, "near_duplicate_cluster"])
    return dict(
        zip(
            table.column(hash_column).to_pylist(),
            table.column("near_duplicate_cluster").to_pylist(),
            strict=True,
        )
    )


def logits_for_indices(
    features: np.ndarray, indices: np.ndarray, checkpoint: dict
) -> np.ndarray:
    scaled = (features[indices] - checkpoint["scaler_mean"]) / checkpoint["scaler_scale"]
    model = ResumeJDLinearClassifier(int(checkpoint["input_dimension"]), class_count=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(scaled.astype(np.float32))).numpy()


def scored_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    label_names: tuple[str, ...],
    threshold: float = 0.5,
) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "probability": binary_probability_metrics(labels, probabilities),
        "classification_at_threshold": classification_metrics(
            labels, predictions, label_names=label_names
        ),
        "threshold": threshold,
    }


def main() -> None:
    args = parse_args()
    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    feature_report = json.loads(args.feature_report.read_text(encoding="utf-8"))
    label_mode = training_report["label_mode"]
    label_names = labels_for_mode(label_mode)
    if len(label_names) != 2:
        raise ValueError("calibration requires a binary training report")

    table = pq.read_table(feature_report["output"]["path"])
    features = matrix_from_table(table).astype(np.float32)
    labels = encode_labels(table.column("label").to_pylist(), label_mode=label_mode)
    metadata_columns = [name for name in table.column_names if name != "features"]
    rows = table.select(metadata_columns).to_pylist()
    resume_clusters = load_clusters(args.resume_manifest, "resume_hash")
    jd_clusters = load_clusters(args.jd_manifest, "jd_hash")
    validation_fraction = float(training_report["training_config"]["validation_fraction"])
    seed = int(training_report["training_config"]["seed"])

    fold_reports = []
    pooled_labels = []
    pooled_uncalibrated = []
    pooled_calibrated = []
    pooled_selected_predictions = []
    for fold_summary in training_report["folds"]:
        fold = int(fold_summary["fold"])
        checkpoint = torch.load(
            fold_summary["checkpoint"]["path"], map_location="cpu", weights_only=False
        )
        partition = build_inner_partition(
            rows,
            resume_clusters,
            jd_clusters,
            outer_fold=fold,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        validation_indices = np.asarray(partition.validation_indices)
        test_indices = np.flatnonzero(
            np.asarray([row[f"fold_{fold}_role"] for row in rows]) == "test"
        )
        validation_logits = logits_for_indices(features, validation_indices, checkpoint)
        validation_margins = validation_logits[:, 1] - validation_logits[:, 0]
        calibrator = fit_binary_platt_calibrator(
            validation_margins, labels[validation_indices]
        )
        validation_probabilities = calibrator.predict(validation_margins)
        threshold_selection = select_binary_threshold(
            labels[validation_indices], validation_probabilities
        )
        test_logits = logits_for_indices(features, test_indices, checkpoint)
        test_margins = test_logits[:, 1] - test_logits[:, 0]
        uncalibrated = expit(test_margins)
        calibrated = calibrator.predict(test_margins)
        test_labels = labels[test_indices]
        pooled_labels.extend(test_labels)
        pooled_uncalibrated.extend(uncalibrated)
        pooled_calibrated.extend(calibrated)
        pooled_selected_predictions.extend(
            (calibrated >= threshold_selection.threshold).astype(np.int64)
        )
        fold_reports.append(
            {
                "fold": fold,
                "rows": len(test_indices),
                "mlflow_run_id": fold_summary["mlflow_run_id"],
                "calibration_rows": len(validation_indices),
                "calibrator": calibrator.to_dict(),
                "threshold_selection": threshold_selection.to_dict(),
                "uncalibrated": scored_metrics(
                    test_labels, uncalibrated, label_names=label_names
                ),
                "calibrated": scored_metrics(
                    test_labels, calibrated, label_names=label_names
                ),
                "calibrated_at_selected_threshold": classification_metrics(
                    test_labels,
                    (calibrated >= threshold_selection.threshold).astype(np.int64),
                    label_names=label_names,
                ),
            }
        )

    pooled_labels_array = np.asarray(pooled_labels)
    report = {
        "label_mode": label_mode,
        "labels": label_names,
        "method": "Platt scaling on each fold's graph-disjoint inner-validation partition",
        "training_report": str(args.training_report),
        "feature_report": str(args.feature_report),
        "folds": fold_reports,
        "pooled_outer_test": {
            "rows": len(pooled_labels_array),
            "uncalibrated": scored_metrics(
                pooled_labels_array,
                np.asarray(pooled_uncalibrated),
                label_names=label_names,
            ),
            "calibrated": scored_metrics(
                pooled_labels_array,
                np.asarray(pooled_calibrated),
                label_names=label_names,
            ),
            "classification_at_fold_selected_thresholds": classification_metrics(
                pooled_labels_array,
                np.asarray(pooled_selected_predictions),
                label_names=label_names,
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    mlflow.set_tracking_uri(training_report["tracking_uri"])
    client = MlflowClient()
    for fold_report in fold_reports:
        run_id = fold_report["mlflow_run_id"]
        for stage in ("uncalibrated", "calibrated"):
            for name, value in fold_report[stage]["probability"].items():
                client.log_metric(run_id, f"outer_test_{stage}_{name}", value)
        client.log_dict(run_id, fold_report["calibrator"], "calibration/platt.json")
    client.log_artifact(
        training_report["parent_mlflow_run_id"], str(args.report), "calibration"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
