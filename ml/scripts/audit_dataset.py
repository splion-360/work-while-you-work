import argparse
import json
from pathlib import Path

from resume_jd_scoring.data import DATASET_NAME, audit_rows, load_dataset_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit resume/JD pairs without modifying them.")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("artifacts/data__dataset_audit.json"))
    parser.add_argument("--near-duplicate-threshold", type=float, default=97.0)
    parser.add_argument("--skip-near-duplicates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, split_names = load_dataset_rows(args.dataset, revision=args.revision)
    report = audit_rows(
        rows,
        near_duplicate_threshold=args.near_duplicate_threshold,
        include_near_duplicates=not args.skip_near_duplicates,
    )
    report["dataset"] = {
        "name": args.dataset,
        "revision": args.revision,
        "splits": split_names,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "output": str(args.output),
        **report["rows"],
        **report["unique"],
        "conflicting_pairs": report["conflicting_pairs"]["count"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
