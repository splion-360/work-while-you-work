#!/usr/bin/env python3
import json
import sys
import urllib.request

from service.config import ScoringSettings
from service.scoring_state import ScoringStore


def api_healthy():
    with urllib.request.urlopen("http://127.0.0.1:8765/healthz", timeout=3) as response:
        return response.status == 200 and json.load(response).get("ok") is True


def scorer_healthy(settings):
    if settings.scoring_backend == "huggingface":
        request = urllib.request.Request(
            f"{settings.hf_space_url}/gradio_api/info",
            headers={"Authorization": f"Bearer {settings.hf_token}"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            info = json.load(response)
        return response.status == 200 and "/score" in info.get("named_endpoints", {})

    with urllib.request.urlopen(
        f"{settings.scoring_engine_url}/healthz", timeout=3
    ) as response:
        engine = json.load(response)
    return response.status == 200 and engine.get("ok") is True and bool(engine.get("gpu"))


def worker_healthy():
    settings = ScoringSettings.from_env()
    store = ScoringStore(settings.database_path)
    return (
        store.is_healthy()
        and store.worker_is_healthy(600)
        and scorer_healthy(settings)
    )


if __name__ == "__main__":
    check = api_healthy if len(sys.argv) > 1 and sys.argv[1] == "api" else worker_healthy
    try:
        healthy = check()
    except Exception:
        healthy = False
    raise SystemExit(0 if healthy else 1)
