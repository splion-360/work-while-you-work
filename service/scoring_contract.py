import hashlib
import subprocess
from pathlib import Path


class ScoringError(RuntimeError):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pdf_text(path, timeout=30):
    try:
        result = subprocess.run(
            ["pdftotext", str(path), "-"],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ScoringError(f"Could not extract resume text: {error}") from error
    text = "\n".join(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not text:
        raise ScoringError("Resume PDF produced no text")
    return text


def score_band(score):
    return "Strong" if score >= 75 else "Moderate" if score >= 55 else "Weak"


def score_input_fingerprint(
    scoring_input, scorer_version, embedding_model, deployment_revision=""
):
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


def score_cache_key(scoring_input, scorer_version, embedding_model, deployment_revision=""):
    source = "|".join(
        (
            scoring_input["role_company_key"],
            scoring_input["resume_content_hash"],
            scorer_version,
            embedding_model,
            deployment_revision,
        )
    )
    return hashlib.sha256(source.encode()).hexdigest()


def role_company_key(company, job_title):
    normalized = "|".join(
        " ".join(str(value).strip().casefold().split())
        for value in (company, job_title)
    )
    return hashlib.sha256(normalized.encode()).hexdigest()
