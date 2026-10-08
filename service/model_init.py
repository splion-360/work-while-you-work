#!/usr/bin/env python3
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from service.config import ScoringSettings
except ModuleNotFoundError:
    from config import ScoringSettings


READINESS_PATH = Path(os.getenv("MODEL_READINESS_PATH", "/data/model-readiness.json"))


def request_json(base_url, path, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def initialize_model(settings, retries=30):
    for attempt in range(retries):
        try:
            request_json(settings.ollama_url, "/api/tags", timeout=5)
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2)
    request_json(
        settings.ollama_url,
        "/api/pull",
        {"model": settings.embedding_model, "stream": False},
        timeout=max(600, settings.ollama_timeout_seconds),
    )
    request_json(
        settings.ollama_url,
        "/api/embed",
        {"model": settings.embedding_model, "input": ["readiness check"]},
        timeout=settings.ollama_timeout_seconds,
    )
    processes = request_json(settings.ollama_url, "/api/ps", timeout=10).get("models", [])
    model = next((item for item in processes if item.get("name", "").startswith(settings.embedding_model)), None)
    gpu_ready = bool(model and model.get("size_vram", 0) > 0)
    if not gpu_ready:
        raise RuntimeError("Ollama loaded the embedding model without GPU memory")
    readiness = {
        "model": settings.embedding_model,
        "model_ready": model is not None,
        "gpu_ready": gpu_ready,
        "gpu_mode": settings.gpu_mode,
    }
    if not readiness["model_ready"]:
        raise RuntimeError("Ollama did not report the initialized embedding model")
    READINESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = READINESS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(readiness, sort_keys=True))
    temporary.replace(READINESS_PATH)
    print(json.dumps({"event": "ollama_model_ready", **readiness}, sort_keys=True))
    return readiness


if __name__ == "__main__":
    initialize_model(ScoringSettings.from_env())
