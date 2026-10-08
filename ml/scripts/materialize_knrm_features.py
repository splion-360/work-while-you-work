import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

from resume_jd_scoring.embeddings import sha256_file, write_pair_features
from resume_jd_scoring.multivector import (
    DISTRIBUTION_STATISTICS,
    TokenVectorCache,
    distributional_knrm_pool_similarity_batch,
    knrm_pool_similarity_batch,
    standard_knrm_kernels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize BGE-M3 K-NRM pair features.")
    parser.add_argument("--token-metadata", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/pairs.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("data/features"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/data__bge_m3_knrm_features.json"))
    parser.add_argument("--fold", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Materialize every pair once; downstream fold roles still control selection.",
    )
    parser.add_argument("--kernel-count", type=int, default=11)
    parser.add_argument("--resume-block-size", type=int, default=4096)
    parser.add_argument("--resume-token-budget", type=int, default=32768)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--pooling", choices=("global", "row-distribution"), default="global"
    )
    return parser.parse_args()


def feature_fingerprint(token_fingerprint: str, contract: dict) -> str:
    payload = json.dumps(
        {"token_fingerprint": token_fingerprint, "contract": contract},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_from_output(output: dict) -> TokenVectorCache:
    return TokenVectorCache(Path(output["vectors"]["path"]), Path(output["index"]["path"]))


def materialize_batch(
    jd_vectors, rows, resume_vectors, kernels, *, pooling, device, block_size
):
    pool = (
        knrm_pool_similarity_batch
        if pooling == "global"
        else distributional_knrm_pool_similarity_batch
    )
    batch_features = pool(
        jd_vectors,
        resume_vectors,
        kernels,
        device=device,
        resume_block_size=block_size,
    )
    return [
        {**row, "features": features.tolist()}
        for row, features in zip(rows, batch_features, strict=True)
    ]


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")
    if args.resume_token_budget <= 0:
        raise ValueError("resume token budget must be positive")
    token_metadata = json.loads(args.token_metadata.read_text(encoding="utf-8"))
    resumes = cache_from_output(token_metadata["outputs"]["resumes"])
    jds = cache_from_output(token_metadata["outputs"]["jds"])
    if resumes.dimension != jds.dimension:
        raise ValueError("resume and JD token cache dimensions differ")
    kernels = standard_knrm_kernels(args.kernel_count)
    feature_definition = "BGE-M3 token similarity matrix with K-NRM kernel pooling"
    representation_label = "BGE-M3 K-NRM features"
    dimension = kernels.count
    contract = {
        "pipeline_version": 1,
        "feature_definition": "BGE-M3 token similarity matrix with K-NRM kernel pooling",
        "fold": args.fold,
        "included_roles": ["train", "test"],
        "jd_side": "query",
        "kernel_means": kernels.means,
        "kernel_standard_deviations": kernels.standard_deviations,
        "kernel_scale": kernels.scale,
        "minimum_soft_count": 1e-10,
        "resume_block_size": args.resume_block_size,
        "resume_token_budget": args.resume_token_budget,
        "compute_dtype": "float32",
    }
    if args.pooling == "row-distribution":
        feature_definition = "BGE-M3 row-distribution K-NRM features"
        representation_label = "BGE-M3 normalized row-distribution K-NRM features"
        dimension = kernels.count * len(DISTRIBUTION_STATISTICS)
        contract.update(
            {
                "pipeline_version": 2,
                "feature_definition": feature_definition,
                "pooling": args.pooling,
                "resume_soft_count_normalization": "divide by resume token count",
                "jd_aggregation": list(DISTRIBUTION_STATISTICS),
            }
        )
    if args.all_folds:
        contract.update(
            {
                "fold": "all",
                "included_roles": "all pairs; downstream consumers select fold roles",
            }
        )
    fingerprint = feature_fingerprint(token_metadata["fingerprint"], contract)
    role_column = f"fold_{args.fold}_role"
    all_pair_rows = pq.read_table(args.pairs).to_pylist()
    pair_rows = (
        all_pair_rows
        if args.all_folds
        else [
            row
            for row in all_pair_rows
            if row[role_column] in contract["included_roles"]
        ]
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in pair_rows:
        grouped[str(row["jd_hash"])].append(row)

    started = time.perf_counter()
    feature_rows = []
    with tqdm(total=len(pair_rows), desc="K-NRM pairs", unit="pair") as progress:
        for jd_hash in sorted(grouped):
            jd_vectors = jds.get(jd_hash)
            pending_rows = []
            pending_vectors = []
            pending_tokens = 0

            for row in grouped[jd_hash]:
                vectors = resumes.get(str(row["resume_hash"]))
                if pending_rows and pending_tokens + len(vectors) > args.resume_token_budget:
                    feature_rows.extend(
                        materialize_batch(
                            jd_vectors,
                            pending_rows,
                            pending_vectors,
                            kernels,
                            pooling=args.pooling,
                            device=args.device,
                            block_size=args.resume_block_size,
                        )
                    )
                    progress.update(len(pending_rows))
                    pending_rows = []
                    pending_vectors = []
                    pending_tokens = 0
                pending_rows.append(row)
                pending_vectors.append(vectors)
                pending_tokens += len(vectors)
            feature_rows.extend(
                materialize_batch(
                    jd_vectors,
                    pending_rows,
                    pending_vectors,
                    kernels,
                    pooling=args.pooling,
                    device=args.device,
                    block_size=args.resume_block_size,
                )
            )
            progress.update(len(pending_rows))
    elapsed = time.perf_counter() - started

    output_path = args.output_root / fingerprint / "pairs.parquet"
    write_pair_features(feature_rows, output_path, dimension=dimension)
    report = {
        "embedding_fingerprint": fingerprint,
        "token_fingerprint": token_metadata["fingerprint"],
        "feature_definition": feature_definition,
        "representation_label": representation_label,
        "dimension": dimension,
        "rows": len(feature_rows),
        "labels": sorted({row["label"] for row in feature_rows}),
        "fold_role_columns": sorted(
            key for key in feature_rows[0] if key.startswith("fold_")
        ),
        "contract": contract,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "seconds": round(elapsed, 3),
            "pairs_per_second": round(len(feature_rows) / elapsed, 3),
        },
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
