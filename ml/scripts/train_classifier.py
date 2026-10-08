import argparse
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# cuBLAS reads this setting when CUDA state is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from tqdm.auto import tqdm

from resume_jd_scoring.embeddings import sha256_file
from resume_jd_scoring.model import ResumeJDLinearClassifier
from resume_jd_scoring.train import (
    LABEL_MODES,
    TrainingConfig,
    build_inner_partition,
    encode_labels,
    fit_fold,
    label_counts,
    labels_for_mode,
    set_reproducible_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the three-fold resume-JD linear-classifier baseline."
    )
    parser.add_argument(
        "--feature-report", type=Path, default=Path("artifacts/data__pair_features.json")
    )
    parser.add_argument("--resume-manifest", type=Path, default=Path("data/processed/resumes.parquet"))
    parser.add_argument("--jd-manifest", type=Path, default=Path("data/processed/jds.parquet"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/train__baseline_classifier.json"))
    parser.add_argument("--tracking-db", type=Path, default=Path("data/mlflow/tracking.db"))
    parser.add_argument("--artifact-root", type=Path, default=Path("data/mlflow/artifacts"))
    parser.add_argument("--experiment", default="resume-jd-baseline")
    parser.add_argument("--fold", type=int, action="append", choices=(0, 1, 2))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="three-class")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")
    return requested


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def load_features(
    path: Path, *, label_mode: str
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    vector_column = table.column("features").combine_chunks()
    dimension = vector_column.type.list_size
    features = vector_column.values.to_numpy(zero_copy_only=False).reshape(-1, dimension)
    labels = encode_labels(table.column("label").to_pylist(), label_mode=label_mode)
    metadata_columns = [name for name in table.column_names if name != "features"]
    rows = table.select(metadata_columns).to_pylist()
    return rows, features.astype(np.float32, copy=False), labels.astype(np.int64)


def load_clusters(path: Path, hash_column: str) -> dict[str, str]:
    table = pq.read_table(path, columns=[hash_column, "near_duplicate_cluster"])
    return dict(
        zip(
            table.column(hash_column).to_pylist(),
            table.column("near_duplicate_cluster").to_pylist(),
        )
    )


def configure_mlflow(tracking_db: Path, artifact_root: Path, experiment_name: str) -> str:
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve()}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_root.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment_id)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def checkpoint_payload(
    model: ResumeJDLinearClassifier,
    result,
    config: TrainingConfig,
    *,
    fold: int,
    input_dimension: int,
    embedding_fingerprint: str,
    label_names: tuple[str, ...],
    label_mode: str,
) -> dict[str, Any]:
    return {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "scaler_mean": result.scaler_mean,
        "scaler_scale": result.scaler_scale,
        "labels": label_names,
        "label_mode": label_mode,
        "model_type": "linear",
        "input_dimension": input_dimension,
        "training_config": asdict(config),
        "outer_fold": fold,
        "embedding_fingerprint": embedding_fingerprint,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "best_validation_accuracy": result.best_validation_accuracy,
        "best_validation_macro_f1": result.best_validation_macro_f1,
    }


def main() -> None:
    args = parse_args()
    import mlflow

    feature_report = json.loads(args.feature_report.read_text(encoding="utf-8"))
    feature_path = Path(feature_report["output"]["path"])
    embedding_fingerprint = feature_report["embedding_fingerprint"]
    label_names = labels_for_mode(args.label_mode)
    rows, features, labels = load_features(feature_path, label_mode=args.label_mode)
    reported_dimension = int(feature_report["dimension"])
    if features.shape[1] != reported_dimension:
        raise ValueError(
            f"feature width {features.shape[1]} differs from reported dimension "
            f"{reported_dimension}"
        )
    resume_clusters = load_clusters(args.resume_manifest, "resume_hash")
    jd_clusters = load_clusters(args.jd_manifest, "jd_hash")
    device = resolve_device(args.device)
    folds = sorted(set(args.fold or (0, 1, 2)))
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    experiment_id = configure_mlflow(args.tracking_db, args.artifact_root, args.experiment)
    run_summaries = []

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"baseline-{embedding_fingerprint[:8]}",
        tags={"git_commit": git_commit(), "stage": "train"},
    ) as parent_run:
        checkpoint_dir = (
            args.checkpoint_root / embedding_fingerprint / parent_run.info.run_id
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        mlflow.log_params(
            {
                "embedding_fingerprint": embedding_fingerprint,
                "feature_definition": feature_report["feature_definition"],
                "feature_rows": len(rows),
                "input_dimension": features.shape[1],
                "labels": json.dumps(label_names),
                "label_mode": args.label_mode,
                "model_type": "linear",
                **asdict(config),
            }
        )
        for fold in folds:
            fold_config = replace(config, seed=config.seed + fold)
            partition = build_inner_partition(
                rows,
                resume_clusters,
                jd_clusters,
                outer_fold=fold,
                validation_fraction=fold_config.validation_fraction,
                seed=config.seed,
            )
            train_indices = np.asarray(partition.train_indices)
            validation_indices = np.asarray(partition.validation_indices)
            set_reproducible_seed(fold_config.seed)
            model = ResumeJDLinearClassifier(
                input_dimension=features.shape[1],
                class_count=len(label_names),
            )
            with mlflow.start_run(run_name=f"fold-{fold}", nested=True) as fold_run:
                mlflow.log_params(
                    {
                        "outer_fold": fold,
                        "model_parameter_count": sum(
                            parameter.numel() for parameter in model.parameters()
                        ),
                        "train_rows": len(train_indices),
                        "validation_rows": len(validation_indices),
                        "inner_discarded_rows": len(partition.discarded_indices),
                        "train_label_counts": json.dumps(
                            label_counts(labels[train_indices], label_names=label_names)
                        ),
                        "validation_label_counts": json.dumps(
                            label_counts(labels[validation_indices], label_names=label_names)
                        ),
                        "seed": fold_config.seed,
                    }
                )

                with tqdm(
                    total=fold_config.max_epochs,
                    desc=f"Fold {fold}",
                    unit="epoch",
                    dynamic_ncols=True,
                ) as progress:

                    def log_epoch(metrics) -> None:
                        mlflow.log_metrics(
                            {
                                "train_loss": metrics.train_loss,
                                "train_accuracy": metrics.train_accuracy,
                                "validation_loss": metrics.validation_loss,
                                "validation_accuracy": metrics.validation_accuracy,
                                "validation_macro_f1": metrics.validation_macro_f1,
                            },
                            step=metrics.epoch,
                        )
                        progress.update()
                        progress.set_postfix(
                            train=f"{metrics.train_loss:.4f}",
                            train_acc=f"{metrics.train_accuracy:.4f}",
                            validation=f"{metrics.validation_loss:.4f}",
                            validation_acc=f"{metrics.validation_accuracy:.4f}",
                            macro_f1=f"{metrics.validation_macro_f1:.4f}",
                        )

                    result = fit_fold(
                        model,
                        features[train_indices],
                        labels[train_indices],
                        features[validation_indices],
                        labels[validation_indices],
                        config=fold_config,
                        device=device,
                        class_count=len(label_names),
                        metric_callback=log_epoch,
                    )
                    if len(result.history) < fold_config.max_epochs:
                        progress.total = len(result.history)
                        progress.set_description(f"Fold {fold} (early stopped)")
                        progress.refresh()
                checkpoint_path = checkpoint_dir / f"fold_{fold}.pt"
                torch.save(
                    checkpoint_payload(
                        model,
                        result,
                        fold_config,
                        fold=fold,
                        input_dimension=features.shape[1],
                        embedding_fingerprint=embedding_fingerprint,
                        label_names=label_names,
                        label_mode=args.label_mode,
                    ),
                    checkpoint_path,
                )
                summary = {
                    "fold": fold,
                    "mlflow_run_id": fold_run.info.run_id,
                    "best_epoch": result.best_epoch,
                    "epochs_run": len(result.history),
                    "best_validation_loss": result.best_validation_loss,
                    "best_validation_accuracy": result.best_validation_accuracy,
                    "best_validation_macro_f1": result.best_validation_macro_f1,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "inner_discarded_rows": len(partition.discarded_indices),
                    "train_labels": label_counts(
                        labels[train_indices], label_names=label_names
                    ),
                    "validation_labels": label_counts(
                        labels[validation_indices], label_names=label_names
                    ),
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "sha256": sha256_file(checkpoint_path),
                    },
                }
                summary_path = checkpoint_dir / f"fold_{fold}.json"
                summary_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                mlflow.log_metrics(
                    {
                        "best_validation_loss": result.best_validation_loss,
                        "best_validation_accuracy": result.best_validation_accuracy,
                        "best_validation_macro_f1": result.best_validation_macro_f1,
                        "best_epoch": result.best_epoch,
                    }
                )
                mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                mlflow.log_artifact(str(summary_path), artifact_path="summaries")
                run_summaries.append(summary)

        report = {
            "parent_mlflow_run_id": parent_run.info.run_id,
            "experiment_id": experiment_id,
            "tracking_uri": mlflow.get_tracking_uri(),
            "embedding_fingerprint": embedding_fingerprint,
            "feature_path": str(feature_path),
            "device": device,
            "labels": label_names,
            "label_mode": args.label_mode,
            "training_config": asdict(config),
            "folds": run_summaries,
            "outer_test_evaluated": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(args.report), artifact_path="reports")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
