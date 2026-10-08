import argparse
import json
from pathlib import Path

from resume_jd_scoring.data import DATASET_NAME, load_dataset_rows
from resume_jd_scoring.splits import analyze_split_feasibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze block-disjoint split feasibility.")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("artifacts/data__split_feasibility.json"))
    parser.add_argument("--near-duplicate-threshold", type=float, default=97.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, split_names = load_dataset_rows(args.dataset, revision=args.revision)
    report = analyze_split_feasibility(
        rows,
        near_duplicate_threshold=args.near_duplicate_threshold,
        seed=args.seed,
    )
    report["dataset"] = {
        "name": args.dataset,
        "revision": args.revision,
        "source_splits": split_names,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report["corpus"]}, indent=2))


if __name__ == "__main__":
    main()
