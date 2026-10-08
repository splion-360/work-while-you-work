import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from resume_jd_scoring.embeddings import (
    build_pair_features,
    read_embedding_cache,
    sha256_file,
    write_pair_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize signed resume-minus-JD features.")
    parser.add_argument("--embedding-metadata", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/pairs.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("data/features"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/data__pair_features.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.embedding_metadata.read_text(encoding="utf-8"))
    fingerprint = metadata["fingerprint"]
    dimension = int(metadata["model"]["embedding_dimension"])
    resume_rows = read_embedding_cache(Path(metadata["outputs"]["resumes"]["path"]))
    jd_rows = read_embedding_cache(Path(metadata["outputs"]["jds"]["path"]))
    pair_rows = pq.read_table(args.pairs).to_pylist()
    features = build_pair_features(pair_rows, resume_rows, jd_rows)

    output_path = args.output_root / fingerprint / "pairs.parquet"
    write_pair_features(features, output_path, dimension=dimension)
    report = {
        "embedding_fingerprint": fingerprint,
        "feature_definition": "resume_embedding - jd_embedding",
        "dimension": dimension,
        "rows": len(features),
        "labels": sorted({row["label"] for row in features}),
        "fold_role_columns": sorted(key for key in features[0] if key.startswith("fold_")),
        "input_pairs_sha256": sha256_file(args.pairs),
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }
    output_path.with_name("metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
