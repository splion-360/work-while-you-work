import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScoringSettings:
    database_path: Path
    ollama_url: str
    embedding_model: str
    scorer_version: str
    max_job_description_chars: int
    min_job_description_chars: int
    ollama_timeout_seconds: int
    job_description_retention_days: int
    gpu_mode: str
    job_lease_seconds: int
    max_score_attempts: int
    worker_poll_seconds: int
    reconciliation_interval_seconds: int
    publication_uncertainty_seconds: int
    score_cache_ttl_seconds: int = 300

    @property
    def scoring_engine_url(self):
        return os.getenv("SCORING_ENGINE_URL", "http://scoring-engine:8770").rstrip("/")

    @property
    def scoring_engine_timeout_seconds(self):
        return _positive_int("SCORING_ENGINE_TIMEOUT_SECONDS", 180)

    @property
    def scoring_backend(self):
        backend = os.getenv("SCORING_BACKEND", "local").strip().lower()
        if backend not in {"local", "huggingface"}:
            raise ValueError("SCORING_BACKEND must be local or huggingface")
        return backend

    @property
    def hf_space_url(self):
        return os.getenv(
            "HF_SPACE_URL", "https://splion360-resume-jd-scorer.hf.space"
        ).rstrip("/")

    @property
    def hf_token(self):
        return os.getenv("HF_TOKEN", "").strip()

    @property
    def hf_model_revision(self):
        return os.getenv("HF_MODEL_REVISION", "").strip()

    @property
    def hf_model_fingerprint(self):
        return os.getenv("HF_MODEL_FINGERPRINT", "").strip()

    @property
    def mlflow_model_version(self):
        return os.getenv("MLFLOW_MODEL_VERSION", "").strip()

    @property
    def scoring_model_revision(self):
        if self.scoring_backend == "huggingface":
            return self.hf_model_revision
        return os.getenv("LOCAL_MODEL_REVISION", "").strip()

    @property
    def mlflow_tracking_uri(self):
        return os.getenv("MLFLOW_TRACKING_URI", "").rstrip("/")

    @property
    def mlflow_inference_experiment(self):
        return os.getenv(
            "MLFLOW_INFERENCE_EXPERIMENT", "resume-jd-production-inference"
        ).strip()

    @property
    def automatic_score_backfill(self):
        return os.getenv("AUTOMATIC_SCORE_BACKFILL", "false").strip().lower() == "true"

    @classmethod
    def from_env(cls):
        gpu_mode = os.getenv("GPU_MODE", "required").strip().lower()
        if gpu_mode != "required":
            raise ValueError("GPU_MODE must be required for the Docker GPU stack")
        settings = cls(
            database_path=Path(os.getenv("SCORING_DB_PATH", "/data/job-tracker.sqlite3")),
            ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3-colbert"),
            scorer_version=os.getenv("SCORER_VERSION", "bge-m3-knrm-binary-v1"),
            max_job_description_chars=_positive_int("MAX_JOB_DESCRIPTION_CHARS", 50000),
            min_job_description_chars=_positive_int("MIN_JOB_DESCRIPTION_CHARS", 200),
            ollama_timeout_seconds=_positive_int("OLLAMA_TIMEOUT_SECONDS", 60),
            job_description_retention_days=_positive_int("JOB_DESCRIPTION_RETENTION_DAYS", 365),
            gpu_mode=gpu_mode,
            job_lease_seconds=_positive_int("JOB_LEASE_SECONDS", 300),
            max_score_attempts=_positive_int("MAX_SCORE_ATTEMPTS", 5),
            worker_poll_seconds=_positive_int("WORKER_POLL_SECONDS", 2),
            reconciliation_interval_seconds=_positive_int("RECONCILIATION_INTERVAL_SECONDS", 900),
            publication_uncertainty_seconds=_positive_int("PUBLICATION_UNCERTAINTY_SECONDS", 900),
            score_cache_ttl_seconds=_positive_int("SCORE_CACHE_TTL_SECONDS", 300),
        )
        if not settings.ollama_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_URL must be an HTTP URL")
        return settings


def _positive_int(name, default):
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
