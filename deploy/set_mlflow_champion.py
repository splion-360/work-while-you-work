#!/usr/bin/env python3
import argparse
import json

import mlflow
from mlflow import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--model-name", default="resume-jd-binary-scorer")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    version = client.get_model_version(args.model_name, args.version)
    if version.status != "READY":
        raise RuntimeError(f"model version {args.version} is not ready: {version.status}")
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    print(json.dumps({
        "model": args.model_name,
        "alias": args.alias,
        "version": str(args.version),
        "run_id": version.run_id,
    }, indent=2))


if __name__ == "__main__":
    main()
