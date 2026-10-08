import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from resume_jd_scoring.evaluate import cross_set_neighbor_profile, multiclass_eta_squared
from resume_jd_scoring.train import LABEL_TO_INDEX, LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose held-out BGE-M3 K-NRM features.")
    parser.add_argument(
        "--feature-reports",
        nargs="+",
        type=Path,
        default=[
            Path("artifacts/data__bge_m3_knrm_features.json"),
            Path("artifacts/data__bge_m3_knrm_features_fold1.json"),
            Path("artifacts/data__bge_m3_knrm_features_fold2.json"),
        ],
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        choices=(0, 1, 2),
        help="Fold numbers to read from one shared all-pairs feature report.",
    )
    parser.add_argument(
        "--token-metadata",
        type=Path,
        default=Path("artifacts/model__bge_m3_multivector_generation.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/analysis/knrm_diagnosis"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/eval__knrm_feature_diagnosis.json"))
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def feature_matrix(table: pa.Table) -> np.ndarray:
    column = table.column("features").combine_chunks()
    return column.values.to_numpy(zero_copy_only=False).reshape(-1, column.type.list_size)


def load_fold(
    report_path: Path, *, requested_fold: int | None = None
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fold = requested_fold if requested_fold is not None else int(report["contract"]["fold"])
    table = pq.read_table(report["output"]["path"])
    roles = np.asarray(table.column(f"fold_{fold}_role").to_pylist())
    labels = np.asarray([LABEL_TO_INDEX[value] for value in table.column("label").to_pylist()])
    hashes = {
        "resume": np.asarray(table.column("resume_hash").to_pylist()),
        "jd": np.asarray(table.column("jd_hash").to_pylist()),
    }
    features = feature_matrix(table).astype(np.float64)
    train = roles == "train"
    test = roles == "test"
    scaler = StandardScaler().fit(features[train])
    return fold, scaler.transform(features[train]), labels[train], scaler.transform(features[test]), {
        "test_labels": labels[test],
        "train_resume_hashes": hashes["resume"][train],
        "train_jd_hashes": hashes["jd"][train],
        "resume_hashes": hashes["resume"][test],
        "jd_hashes": hashes["jd"][test],
        "train_labels": labels[train],
    }


def spearman_correlations(features: np.ndarray, values: np.ndarray) -> np.ndarray:
    feature_ranks = np.apply_along_axis(rankdata, 0, features)
    value_ranks = rankdata(values)
    correlations = []
    for index in range(features.shape[1]):
        if np.std(feature_ranks[:, index]) == 0:
            correlations.append(0.0)
        else:
            correlations.append(np.corrcoef(feature_ranks[:, index], value_ranks)[0, 1])
    return np.asarray(correlations)


def class_statistics(features: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for kernel in range(features.shape[1]):
        per_class = {}
        for label_index, label in enumerate(LABELS):
            values = features[labels == label_index, kernel]
            per_class[label] = {
                "mean": float(values.mean()),
                "standard_deviation": float(values.std()),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
        rows.append({"kernel": kernel, "by_class": per_class})
    return rows


def plot_class_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    path: Path,
    *,
    feature_indices: np.ndarray,
) -> None:
    colors = ("#d1495b", "#ed9b40", "#2a9d8f")
    figure, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
    for position, axis in enumerate(axes.flat):
        if position >= len(feature_indices):
            axis.axis("off")
            continue
        feature_index = int(feature_indices[position])
        values = [features[labels == index, feature_index] for index in range(len(LABELS))]
        boxes = axis.boxplot(values, tick_labels=LABELS, showfliers=False, patch_artist=True)
        for box, color in zip(boxes["boxes"], colors, strict=True):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        axis.set_title(f"Feature {feature_index}")
        axis.tick_params(axis="x", labelrotation=25, labelsize=7)
        axis.axhline(0, color="#777777", linewidth=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_correlation(correlation: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(correlation)), labels=range(len(correlation)))
    axis.set_yticks(range(len(correlation)), labels=range(len(correlation)))
    axis.set_xlabel("Kernel")
    axis.set_ylabel("Kernel")
    axis.set_title("Pearson correlation across held-out K-NRM features")
    figure.colorbar(image, ax=axis, shrink=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def index_counts(path: str) -> dict[str, int]:
    table = pq.read_table(path, columns=["document_hash", "token_count"])
    return dict(zip(table.column("document_hash").to_pylist(), table.column("token_count").to_pylist(), strict=True))


def residualize_lengths(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_lengths: np.ndarray,
    test_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_design = np.column_stack(
        (np.ones(len(train_lengths)), train_lengths, np.log1p(train_lengths))
    )
    test_design = np.column_stack(
        (np.ones(len(test_lengths)), test_lengths, np.log1p(test_lengths))
    )
    coefficients = np.linalg.lstsq(train_design, train_features, rcond=None)[0]
    train_residuals = train_features - train_design @ coefficients
    test_residuals = test_features - test_design @ coefficients
    scaler = StandardScaler().fit(train_residuals)
    return scaler.transform(train_residuals), scaler.transform(test_residuals)


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.token_metadata.read_text(encoding="utf-8"))
    resume_counts = index_counts(metadata["outputs"]["resumes"]["index"]["path"])
    jd_counts = index_counts(metadata["outputs"]["jds"]["index"]["path"])

    test_features = []
    test_labels = []
    resume_lengths = []
    jd_lengths = []
    residual_test_features = []
    folds = []
    if args.folds:
        if len(args.feature_reports) != 1:
            raise ValueError("explicit folds require exactly one all-pairs feature report")
        fold_inputs = [(args.feature_reports[0], fold) for fold in args.folds]
    else:
        fold_inputs = [(report_path, None) for report_path in args.feature_reports]
    for report_path, requested_fold in fold_inputs:
        fold, train_x, train_y, test_x, info = load_fold(
            report_path, requested_fold=requested_fold
        )
        feature_profile = cross_set_neighbor_profile(
            train_x,
            train_y,
            test_x,
            info["test_labels"],
            neighbors=args.neighbors,
            class_count=len(LABELS),
        )
        train_lengths = np.column_stack(
            (
                [resume_counts[value] for value in info["train_resume_hashes"]],
                [jd_counts[value] for value in info["train_jd_hashes"]],
            )
        )
        test_lengths = np.column_stack(
            (
                [resume_counts[value] for value in info["resume_hashes"]],
                [jd_counts[value] for value in info["jd_hashes"]],
            )
        )
        length_scaler = StandardScaler().fit(np.log1p(train_lengths))
        length_profile = cross_set_neighbor_profile(
            length_scaler.transform(np.log1p(train_lengths)),
            train_y,
            length_scaler.transform(np.log1p(test_lengths)),
            info["test_labels"],
            neighbors=args.neighbors,
            class_count=len(LABELS),
        )
        residual_train, residual_test = residualize_lengths(
            train_x, test_x, train_lengths, test_lengths
        )
        residual_profile = cross_set_neighbor_profile(
            residual_train,
            train_y,
            residual_test,
            info["test_labels"],
            neighbors=args.neighbors,
            class_count=len(LABELS),
        )
        train_prevalence = np.bincount(train_y, minlength=len(LABELS)) / len(train_y)
        test_prevalence = (
            np.bincount(info["test_labels"], minlength=len(LABELS)) / len(info["test_labels"])
        )
        folds.append(
            {
                "fold": fold,
                "train_rows": len(train_y),
                "test_rows": len(info["test_labels"]),
                "overall_neighbor_purity": feature_profile["overall_purity"],
                "length_only_overall_neighbor_purity": length_profile["overall_purity"],
                "length_residualized_overall_neighbor_purity": residual_profile["overall_purity"],
                "class_prior_expected_neighbor_purity": float(train_prevalence @ test_prevalence),
                "train_class_prevalence": dict(zip(LABELS, train_prevalence.tolist(), strict=True)),
                "same_label_neighbor_fraction": dict(zip(LABELS, feature_profile["same_label_fraction"].tolist(), strict=True)),
                "length_only_same_label_neighbor_fraction": dict(
                    zip(LABELS, length_profile["same_label_fraction"].tolist(), strict=True)
                ),
                "length_residualized_same_label_neighbor_fraction": dict(
                    zip(LABELS, residual_profile["same_label_fraction"].tolist(), strict=True)
                ),
                "same_label_lift_over_train_prevalence": dict(
                    zip(LABELS, (feature_profile["same_label_fraction"] / train_prevalence).tolist(), strict=True)
                ),
                "neighbor_class_composition": {
                    label: dict(zip(LABELS, feature_profile["neighbor_class_composition"][index].tolist(), strict=True))
                    for index, label in enumerate(LABELS)
                },
            }
        )
        test_features.append(test_x)
        residual_test_features.append(residual_test)
        test_labels.append(info["test_labels"])
        resume_lengths.extend(resume_counts[value] for value in info["resume_hashes"])
        jd_lengths.extend(jd_counts[value] for value in info["jd_hashes"])

    features = np.concatenate(test_features)
    labels = np.concatenate(test_labels)
    eta_squared = multiclass_eta_squared(features, labels)
    mutual_information = mutual_info_classif(features, labels, random_state=args.seed)
    nonconstant = np.flatnonzero(features.std(axis=0) > 0)
    constant = np.flatnonzero(features.std(axis=0) == 0)
    informative_correlation = np.corrcoef(features[:, nonconstant], rowvar=False)
    correlation = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    correlation[np.ix_(nonconstant, nonconstant)] = informative_correlation
    eigenvalues = np.linalg.eigvalsh(informative_correlation).clip(min=0)
    effective_dimension = float(np.square(eigenvalues.sum()) / np.square(eigenvalues).sum())
    residual_features = np.concatenate(residual_test_features)
    residual_nonconstant = np.flatnonzero(residual_features.std(axis=0) > 0)
    residual_correlation = np.corrcoef(
        residual_features[:, residual_nonconstant], rowvar=False
    )
    residual_eigenvalues = np.linalg.eigvalsh(residual_correlation).clip(min=0)
    residual_effective_dimension = float(
        np.square(residual_eigenvalues.sum()) / np.square(residual_eigenvalues).sum()
    )
    redundant_pairs = [
        {"left_kernel": left, "right_kernel": right, "correlation": float(correlation[left, right])}
        for left in range(features.shape[1])
        for right in range(left + 1, features.shape[1])
        if abs(correlation[left, right]) >= 0.95
    ]
    distribution_path = args.output_root / "class_distributions.png"
    correlation_path = args.output_root / "kernel_correlation.png"
    plotted_feature_indices = np.argsort(eta_squared)[-min(12, features.shape[1]) :][::-1]
    plot_class_distributions(
        features,
        labels,
        distribution_path,
        feature_indices=plotted_feature_indices,
    )
    plot_correlation(correlation, correlation_path)

    report = {
        "scope": "all three graph-disjoint outer-test folds; each fold standardized from its outer-train rows",
        "rows": len(labels),
        "labels": {label: int((labels == index).sum()) for index, label in enumerate(LABELS)},
        "kernel_class_statistics": class_statistics(features, labels),
        "kernel_eta_squared": eta_squared.tolist(),
        "kernel_mutual_information": mutual_information.tolist(),
        "constant_feature_indices": constant.tolist(),
        "correlation_feature_indices": nonconstant.tolist(),
        "correlation_effective_dimension": effective_dimension,
        "correlation_eigenvalues": eigenvalues.tolist(),
        "length_residualized_kernel_eta_squared": multiclass_eta_squared(
            residual_features, labels
        ).tolist(),
        "length_residualized_correlation_effective_dimension": residual_effective_dimension,
        "length_residualized_correlation_eigenvalues": residual_eigenvalues.tolist(),
        "pearson_correlation": correlation.tolist(),
        "redundant_pairs_absolute_correlation_at_least_0_95": redundant_pairs,
        "spearman_token_length_correlation": {
            "resume": spearman_correlations(features, np.asarray(resume_lengths)).tolist(),
            "jd": spearman_correlations(features, np.asarray(jd_lengths)).tolist(),
        },
        "cross_set_neighbor_profiles": folds,
        "plots": {
            "class_distributions": str(distribution_path),
            "class_distribution_feature_indices": plotted_feature_indices.tolist(),
            "kernel_correlation": str(correlation_path),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
