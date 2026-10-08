import argparse
import json
from pathlib import Path

from resume_jd_scoring.data import DATASET_NAME, load_dataset_rows
from resume_jd_scoring.materialize import materialize_three_fold_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the selected 3-fold split.")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--near-duplicate-threshold", type=float, default=97.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, source_splits = load_dataset_rows(args.dataset, revision=args.revision)
    report = materialize_three_fold_split(
        rows,
        args.output_dir,
        near_duplicate_threshold=args.near_duplicate_threshold,
        seed=args.seed,
    )
    report["dataset"] = {
        "name": args.dataset,
        "revision": args.revision,
        "source_splits": source_splits,
    }
    metadata_path = args.output_dir / "split_metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                **report["counts"],
                "checks": report["validation"]["checks"],
                "folds": report["validation"]["folds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
