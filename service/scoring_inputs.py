import hashlib
import re
from datetime import datetime, timezone

from service.config import ScoringSettings
from service.scoring_contract import file_hash, role_company_key

EXCLUDED_SECTION_HEADINGS = (
    "benefits",
    "benefits and perks",
    "compensation",
    "disclaimer",
    "eeo",
    "equal employment opportunity",
    "equal opportunity",
    "legal",
    "pay range",
    "perks",
    "privacy notice",
    "salary",
)


def normalize_structured_text(value):
    lines = []
    pending_blank = False
    for raw_line in str(value).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = " ".join(raw_line.split())
        if line:
            if pending_blank and lines:
                lines.append("")
            lines.append(line)
            pending_blank = False
        elif lines:
            pending_blank = True
    return "\n".join(lines)


def heading_key(line):
    normalized = re.sub(r"[^a-z0-9&/ ]+", "", str(line).lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def scoring_job_description(value):
    description = normalize_structured_text(value)
    lines = description.splitlines()
    excluded = set(EXCLUDED_SECTION_HEADINGS)
    for index, line in enumerate(lines):
        key = heading_key(line)
        if key in excluded or any(key.startswith(f"{heading} ") for heading in excluded):
            return "\n".join(lines[:index]).rstrip()
    return description


def prepare_scoring_input(application, resume, settings=None):
    settings = settings or ScoringSettings.from_env()
    description = scoring_job_description(application.get("job_description", ""))
    quality = str(application.get("extraction_quality", "complete"))
    if len(description) > settings.max_job_description_chars:
        description = description[: settings.max_job_description_chars].rstrip()
        quality = "partial"
    return {
        "application_key": application["application_key"],
        "canonical_url": str(application.get("canonical_url") or application["job_url"]),
        "source_job_id": str(application.get("source_job_id", "")),
        "job_description": description,
        "job_description_hash": hashlib.sha256(description.encode()).hexdigest(),
        "description_provenance": str(
            application.get("description_provenance") or "original_saved"
        ),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "extraction_quality": quality,
        "resume_version": application["resume_version"],
        "resume_path": resume["path"],
        "resume_content_hash": file_hash(resume["path"]),
        "role_company_key": role_company_key(
            application.get("company", ""), application.get("job_title", "")
        ),
    }
