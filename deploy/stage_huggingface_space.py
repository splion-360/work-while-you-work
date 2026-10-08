#!/usr/bin/env python3
import argparse
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPACE_TEMPLATE = ROOT / "deploy" / "huggingface-space"
SCORER_PACKAGE = ROOT / "ml" / "src" / "resume_jd_scoring"
MODEL_BUNDLE = ROOT / "ml" / "runtime" / "binary-scorer"


@contextmanager
def staged_space(
    destination: Path | None = None,
    *,
    model_bundle: Path = MODEL_BUNDLE,
    deployment_metadata: dict | None = None,
):
    temporary = None
    if destination is None:
        temporary = tempfile.TemporaryDirectory(prefix="job-tracker-hf-space-")
        destination = Path(temporary.name)
    else:
        destination.mkdir(parents=True, exist_ok=False)
    try:
        for source in SPACE_TEMPLATE.iterdir():
            if source.is_file():
                shutil.copy2(source, destination / source.name)
        shutil.copytree(
            SCORER_PACKAGE,
            destination / "resume_jd_scoring",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(model_bundle, destination / "model")
        if deployment_metadata is not None:
            (destination / "model" / "deployment.json").write_text(
                json.dumps(deployment_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        yield destination
    finally:
        if temporary is not None:
            temporary.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with staged_space(args.destination) as path:
        print(path)


if __name__ == "__main__":
    main()
