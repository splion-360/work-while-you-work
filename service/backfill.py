#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from service.config import ScoringSettings
from service.notion import NotionClient
from service.scoring_contract import file_hash, score_input_fingerprint
from service.scoring_inputs import prepare_scoring_input
from service.scoring_state import ScoringStore
from service.server import resume_catalog

PROVENANCE = {"original_saved", "manual_supplied", "recaptured_current_page"}


def property_value(page, name):
    prop = page.get("properties", {}).get(name, {})
    if prop.get("select"):
        return str(prop["select"].get("name", "")).strip()
    if "url" in prop:
        return str(prop.get("url") or "").strip()
    values = prop.get("title") or prop.get("rich_text") or []
    return "".join(item.get("plain_text", item.get("text", {}).get("content", "")) for item in values).strip()


def _resume_index(resumes):
    index = {}
    for resume in resumes:
        index.setdefault(resume["label"], []).append(resume)
    return index


def run_backfill(store, notion, applications_source_id, resumes, supplied, execute=False, min_description_chars=100):
    report = {"completed": 0, "eligible": 0, "skipped": 0, "failed": 0, "reasons": Counter()}
    resume_index = _resume_index(resumes)
    settings = ScoringSettings.from_env()
    for page in notion.query_data_source(applications_source_id):
        key = property_value(page, "Application key")
        try:
            if not key:
                raise ValueError("missing_application_key")
            provided = supplied.get(key)
            saved_description = property_value(page, "Job description")
            if provided:
                description = str(provided.get("job_description", "")).strip()
                provenance = str(provided.get("provenance", "manual_supplied"))
                captured_at = str(provided.get("captured_at") or datetime.now(timezone.utc).isoformat())
                quality = str(provided.get("extraction_quality", "complete"))
            elif saved_description:
                description = saved_description
                provenance = "original_saved"
                captured_at = str(page.get("created_time") or datetime.now(timezone.utc).isoformat())
                quality = "complete"
            else:
                raise ValueError("missing_description")
            if provenance not in PROVENANCE:
                raise ValueError("invalid_provenance")
            if quality != "complete" or len(description) < min_description_chars:
                raise ValueError("low_quality_description")
            try:
                capture_time = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
                if capture_time.tzinfo is None:
                    raise ValueError
                capture_time = capture_time.astimezone(timezone.utc)
            except ValueError as error:
                raise ValueError("invalid_capture_time") from error
            age_days = (datetime.now(timezone.utc) - capture_time).days
            if age_days > settings.job_description_retention_days:
                raise ValueError("expired_description")

            label = property_value(page, "Resume version")
            matches = resume_index.get(label, [])
            if not matches:
                raise ValueError("missing_resume_snapshot")
            if len(matches) > 1:
                raise ValueError("ambiguous_resume_snapshot")
            resume = matches[0]
            if not Path(resume["path"]).is_file():
                raise ValueError("missing_resume_snapshot")
            expected_hash = file_hash(resume["path"])
            existing = store.get_scoring_input(key)
            historical_hash = (
                str(provided.get("resume_content_hash", "")).strip() if provided else ""
            ) or (existing["resume_content_hash"] if existing else "")
            if not historical_hash:
                raise ValueError("missing_historical_resume_hash")
            if historical_hash != expected_hash:
                raise ValueError("resume_snapshot_changed")

            application = {
                "application_key": key,
                "job_url": property_value(page, "Job URL"),
                "canonical_url": property_value(page, "Job URL"),
                "source_job_id": "",
                "job_description": description,
                "description_provenance": provenance,
                "extraction_quality": quality,
                "resume_version": resume["version"],
            }
            scoring_input = prepare_scoring_input(application, resume, settings)
            scoring_input["captured_at"] = captured_at
            if existing and existing["job_description_hash"] != scoring_input["job_description_hash"]:
                raise ValueError("immutable_input_conflict")
            if existing:
                scoring_input = existing
            fingerprint = score_input_fingerprint(
                scoring_input,
                settings.scorer_version,
                settings.embedding_model,
                settings.scoring_model_revision,
            )
            if execute:
                store.save_scoring_input(scoring_input)
                store.save_scoring_snapshot(fingerprint, scoring_input)
                store.enqueue_score_job(key, fingerprint, status="queued")
                report["completed"] += 1
            else:
                report["eligible"] += 1
        except ValueError as error:
            report["skipped"] += 1
            report["reasons"][str(error)] += 1
        except (OSError, KeyError, TypeError, OverflowError):
            report["failed"] += 1
            report["reasons"]["processing_error"] += 1
    report["reasons"] = dict(sorted(report["reasons"].items()))
    return report


def load_supplied(path):
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Descriptions file must contain an object keyed by Application key")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptions", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--min-description-chars", type=int, default=100)
    args = parser.parse_args()
    settings = ScoringSettings.from_env()
    store = ScoringStore(settings.database_path)
    store.initialize()
    report = run_backfill(
        store,
        NotionClient(os.getenv("NOTION_TOKEN", "")),
        os.getenv("NOTION_DATA_SOURCE_ID", ""),
        resume_catalog(),
        load_supplied(args.descriptions),
        execute=args.execute,
        min_description_chars=max(1, args.min_description_chars),
    )
    print(json.dumps({"mode": "execute" if args.execute else "dry_run", **report}, sort_keys=True))


if __name__ == "__main__":
    main()
