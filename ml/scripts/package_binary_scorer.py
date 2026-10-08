import argparse
import json
import shutil
from pathlib import Path

from resume_jd_scoring.embeddings import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the selected binary scoring fold.")
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--token-metadata", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--output", type=Path, default=Path("exports/binary-scorer"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    features = json.loads(args.feature_report.read_text(encoding="utf-8"))
    tokens = json.loads(args.token_metadata.read_text(encoding="utf-8"))
    if training["label_mode"] != "binary-potential-positive":
        raise ValueError("training report is not the requested binary label mode")
    fold_summary = (
        max(training["folds"], key=lambda item: item["best_validation_macro_f1"])
        if args.fold is None
        else next(item for item in training["folds"] if item["fold"] == args.fold)
    )
    fold = int(fold_summary["fold"])
    calibration_summary = next(item for item in calibration["folds"] if item["fold"] == fold)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_source = Path(fold_summary["checkpoint"]["path"])
    checkpoint_output = args.output / "classifier.pt"
    shutil.copy2(checkpoint_source, checkpoint_output)
    manifest = {
        "format_version": 1,
        "labels": training["labels"],
        "label_mode": training["label_mode"],
        "selected_fold": fold,
        "selection_rule": "highest inner-validation macro F1",
        "source_mlflow_run_id": fold_summary["mlflow_run_id"],
        "classifier": {
            "path": checkpoint_output.name,
            "sha256": sha256_file(checkpoint_output),
            "input_dimension": features["dimension"],
        },
        "calibration": {
            "method": "Platt scaling",
            **calibration_summary["calibrator"],
        },
        "decision_threshold": calibration_summary["threshold_selection"],
        "encoder": {
            "name": tokens["model"]["name"],
            "fingerprint": tokens["fingerprint"],
            "max_length": tokens["encoding"]["max_length"],
            "representation": tokens["encoding"]["representation"],
        },
        "features": {
            "fingerprint": features["embedding_fingerprint"],
            "contract": features["contract"],
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
