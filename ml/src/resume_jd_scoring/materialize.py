import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from resume_jd_scoring.splits import (
    PairCorpus,
    PairRecord,
    build_pair_corpus,
    near_duplicate_groups,
    stratified_fold_assignment,
)

N_SPLITS = 3
SEED = 42
NEAR_DUPLICATE_THRESHOLD = 97.0


def pair_role(pair: PairRecord, resume_folds: Mapping[str, int], jd_folds: Mapping[str, int], fold: int) -> str:
    resume_held_out = resume_folds[pair.resume_hash] == fold
    jd_held_out = jd_folds[pair.jd_hash] == fold
    if resume_held_out and jd_held_out:
        return "test"
    if not resume_held_out and not jd_held_out:
        return "train"
    return "discarded"


def build_manifest_rows(
    corpus: PairCorpus,
    resume_groups: Mapping[str, str],
    jd_groups: Mapping[str, str],
    resume_folds: Mapping[str, int],
    jd_folds: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    resume_rows = [
        {
            "resume_hash": item_hash,
            "text": corpus.resumes[item_hash],
            "near_duplicate_cluster": resume_groups[item_hash],
            "fold": resume_folds[item_hash],
            "resume_domain": corpus.resume_domains[item_hash],
            "domain_assignment_confidence": corpus.resume_domain_confidence[item_hash],
        }
        for item_hash in sorted(corpus.resumes)
    ]
    jd_rows = [
        {
            "jd_hash": item_hash,
            "text": corpus.jds[item_hash],
            "near_duplicate_cluster": jd_groups[item_hash],
            "fold": jd_folds[item_hash],
            "jd_domain": corpus.jd_domains[item_hash],
            "domain_assignment_confidence": corpus.jd_domain_confidence[item_hash],
        }
        for item_hash in sorted(corpus.jds)
    ]
    pair_rows = []
    for pair in sorted(corpus.pairs, key=lambda item: (item.resume_hash, item.jd_hash, item.label)):
        pair_rows.append(
            {
                "pair_hash": hashlib.sha256(
                    f"{pair.resume_hash}:{pair.jd_hash}".encode()
                ).hexdigest(),
                "resume_hash": pair.resume_hash,
                "jd_hash": pair.jd_hash,
                "label": pair.label,
                **{
                    f"fold_{fold}_role": pair_role(pair, resume_folds, jd_folds, fold)
                    for fold in range(N_SPLITS)
                },
            }
        )
    return resume_rows, jd_rows, pair_rows


def validate_manifest_rows(
    resume_rows: list[dict[str, Any]],
    jd_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    resume_folds = {row["resume_hash"]: row["fold"] for row in resume_rows}
    jd_folds = {row["jd_hash"]: row["fold"] for row in jd_rows}
    resume_clusters: dict[str, set[int]] = {}
    jd_clusters: dict[str, set[int]] = {}
    for row in resume_rows:
        resume_clusters.setdefault(row["near_duplicate_cluster"], set()).add(row["fold"])
    for row in jd_rows:
        jd_clusters.setdefault(row["near_duplicate_cluster"], set()).add(row["fold"])

    fold_reports = []
    for fold in range(N_SPLITS):
        role_counts = Counter(row[f"fold_{fold}_role"] for row in pair_rows)
        train_rows = [row for row in pair_rows if row[f"fold_{fold}_role"] == "train"]
        test_rows = [row for row in pair_rows if row[f"fold_{fold}_role"] == "test"]
        train_resumes = {row["resume_hash"] for row in train_rows}
        test_resumes = {row["resume_hash"] for row in test_rows}
        train_jds = {row["jd_hash"] for row in train_rows}
        test_jds = {row["jd_hash"] for row in test_rows}
        fold_reports.append(
            {
                "fold": fold,
                "roles": dict(sorted(role_counts.items())),
                "test_labels": dict(
                    sorted(Counter(row["label"] for row in test_rows).items())
                ),
                "train_test_resume_overlap": len(train_resumes & test_resumes),
                "train_test_jd_overlap": len(train_jds & test_jds),
            }
        )

    checks = {
        "unique_resume_hashes": len(resume_folds) == len(resume_rows),
        "unique_jd_hashes": len(jd_folds) == len(jd_rows),
        "unique_pair_hashes": len({row["pair_hash"] for row in pair_rows}) == len(pair_rows),
        "resume_fold_coverage": set(resume_folds.values()) == set(range(N_SPLITS)),
        "jd_fold_coverage": set(jd_folds.values()) == set(range(N_SPLITS)),
        "resume_clusters_kept_together": all(len(folds) == 1 for folds in resume_clusters.values()),
        "jd_clusters_kept_together": all(len(folds) == 1 for folds in jd_clusters.values()),
        "pair_endpoints_exist": all(
            row["resume_hash"] in resume_folds and row["jd_hash"] in jd_folds
            for row in pair_rows
        ),
        "roles_complete": all(
            sum(report["roles"].values()) == len(pair_rows) for report in fold_reports
        ),
        "zero_train_test_document_overlap": all(
            report["train_test_resume_overlap"] == 0 and report["train_test_jd_overlap"] == 0
            for report in fold_reports
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Split manifest validation failed: {', '.join(failed)}")
    return {"checks": checks, "folds": fold_reports}


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", write_statistics=True)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_three_fold_split(
    rows: Iterable[Mapping[str, Any]],
    output_dir: Path,
    *,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    seed: int = SEED,
) -> dict[str, Any]:
    corpus = build_pair_corpus(rows)
    resume_groups, resume_group_report = near_duplicate_groups(
        corpus.resumes, threshold=near_duplicate_threshold
    )
    jd_groups, jd_group_report = near_duplicate_groups(
        corpus.jds, threshold=near_duplicate_threshold
    )
    resume_folds = stratified_fold_assignment(
        corpus.resume_domains, resume_groups, n_splits=N_SPLITS, seed=seed
    )
    jd_folds = stratified_fold_assignment(
        corpus.jd_domains, jd_groups, n_splits=N_SPLITS, seed=seed + 1
    )
    resume_rows, jd_rows, pair_rows = build_manifest_rows(
        corpus, resume_groups, jd_groups, resume_folds, jd_folds
    )
    validation = validate_manifest_rows(resume_rows, jd_rows, pair_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "resumes": output_dir / "resumes.parquet",
        "jds": output_dir / "jds.parquet",
        "pairs": output_dir / "pairs.parquet",
    }
    _write_parquet(resume_rows, paths["resumes"])
    _write_parquet(jd_rows, paths["jds"])
    _write_parquet(pair_rows, paths["pairs"])
    report = {
        "configuration": {
            "n_splits": N_SPLITS,
            "seed": seed,
            "resume_seed": seed,
            "jd_seed": seed + 1,
            "near_duplicate_threshold": near_duplicate_threshold,
        },
        "counts": {
            "resumes": len(resume_rows),
            "jds": len(jd_rows),
            "pairs": len(pair_rows),
        },
        "near_duplicate_groups": {
            "resumes": resume_group_report,
            "jds": jd_group_report,
        },
        "domain_conflicts": corpus.domain_conflicts,
        "validation": validation,
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _file_hash(path),
            }
            for name, path in paths.items()
        },
    }
    metadata_path = output_dir / "split_metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
