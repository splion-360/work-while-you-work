import hashlib
import time
from typing import Any

from resume_jd_scoring.inference import BinaryResumeJDScorer, MultivectorEncoder


def score_input_fingerprint(
    scoring_input: dict[str, Any], scorer_version: str, embedding_model: str,
    deployment_revision: str = "",
) -> str:
    source = "|".join(
        (
            scoring_input["application_key"],
            scoring_input["resume_content_hash"],
            scoring_input["job_description_hash"],
            scorer_version,
            embedding_model,
            deployment_revision,
        )
    )
    return hashlib.sha256(source.encode()).hexdigest()


def score_band(score: float) -> str:
    return "Strong" if score >= 75 else "Moderate" if score >= 55 else "Weak"


class HostedScoringRuntime:
    def __init__(
        self,
        scorer: BinaryResumeJDScorer,
        encoder: MultivectorEncoder,
        *,
        scorer_version: str,
        embedding_model: str,
        max_length: int,
        pooling_device: str = "cuda:0",
        provenance: dict[str, str] | None = None,
    ) -> None:
        self.scorer = scorer
        self.encoder = encoder
        self.scorer_version = scorer_version
        self.embedding_model = embedding_model
        self.max_length = max_length
        self.pooling_device = pooling_device
        self.provenance = dict(provenance or {})

    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        scoring_input = payload["scoring_input"]
        if "resume_path" in scoring_input:
            raise ValueError("remote scoring input must not contain a resume path")
        resume_text = str(scoring_input["resume_text"]).strip()
        job_description = str(scoring_input["job_description"])
        if not resume_text or not job_description.strip():
            raise ValueError("resume and job-description text are required")
        job_hash = hashlib.sha256(job_description.encode()).hexdigest()
        if job_hash != scoring_input["job_description_hash"]:
            raise ValueError("job description does not match its content hash")
        fingerprint = score_input_fingerprint(
            scoring_input,
            self.scorer_version,
            self.embedding_model,
            str(payload.get("deployment_revision", "")),
        )
        if payload["input_fingerprint"] != fingerprint:
            raise ValueError("scoring input fingerprint does not match its content hashes")

        started = time.perf_counter()
        result = self.scorer.score_texts(
            self.encoder,
            resume_text,
            job_description,
            pooling_device=self.pooling_device,
            max_length=self.max_length,
        )
        score = round(float(result["score"]), 1)
        return {
            "application_key": scoring_input["application_key"],
            "input_fingerprint": fingerprint,
            "scorer_version": self.scorer_version,
            "embedding_model": self.embedding_model,
            "score": score,
            "band": score_band(score),
            "fit_label": result["label"],
            "fit_probability": result["probability"],
            "decision_threshold": result["decision_threshold"],
            "matched_terms": [],
            "category_breakdown": {},
            "top_gaps": [],
            "inference_ms": round((time.perf_counter() - started) * 1000, 1),
            **self.provenance,
        }
