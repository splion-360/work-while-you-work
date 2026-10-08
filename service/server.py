#!/usr/bin/env python3
import json
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from service.application_data import (
    STATUS_COUNT_KEYS,
    VALID_STATUSES,
    application_check_response,
    application_date_from_page,
    application_from_notion_page,
    application_key,
    application_update_properties,
    month_bounds,
    normalize_application_update,
    notion_rich_text,
    notion_select_value,
    preview_application_key,
    score_from_notion_page,
)
from service.config import ScoringSettings
from service.notion import NotionClient
from service.resumes import build_resume_catalog
from service.score_persistence import ScorePersistenceError, persist_score_result
from service.scoring_client import create_scoring_client
from service.scoring_contract import score_cache_key, score_input_fingerprint
from service.scoring_inputs import prepare_scoring_input
from service.scoring_state import ScoringStore

RESUMES_DIR = Path(os.getenv("RESUMES_DIR", "/home/splion/Desktop/personal/resume/resumes"))
PORT = int(os.getenv("PORT", "8765"))
HOST = os.getenv("HOST", "127.0.0.1")
API_KEY = os.getenv("JOB_TRACKER_API_KEY", "")
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")
NOTION_DATA_SOURCE_ID = os.getenv("NOTION_DATA_SOURCE_ID", "")
NOTION_SCORE_DATABASE_ID = os.getenv("NOTION_SCORE_DATABASE_ID", "")
NOTION_SCORE_DATA_SOURCE_ID = os.getenv("NOTION_SCORE_DATA_SOURCE_ID", "")


def notion_client(notion_version="2026-03-11"):
    return NotionClient(NOTION_TOKEN, notion_version=notion_version)


def capture_scoring_input(application, resume):
    settings = ScoringSettings.from_env()
    description = str(application.get("job_description", "")).strip()
    if (
        len(description) < settings.min_job_description_chars
        or application.get("extraction_quality") not in {None, "", "complete"}
    ):
        return "unavailable", None
    store = ScoringStore(settings.database_path)
    store.initialize()
    scoring_input = prepare_scoring_input(application, resume, settings)
    created = store.save_scoring_input(scoring_input)
    fingerprint = score_input_fingerprint(
        scoring_input,
        settings.scorer_version,
        settings.embedding_model,
        settings.scoring_model_revision,
    )
    store.save_scoring_snapshot(fingerprint, scoring_input)
    store.enqueue_score_job(application["application_key"], fingerprint, status="waiting_application")
    return ("captured" if created else "already_captured"), fingerprint


def activate_captured_score(fingerprint):
    if not fingerprint:
        return "unavailable"
    try:
        settings = ScoringSettings.from_env()
        store = ScoringStore(settings.database_path)
        return "queued" if store.activate_score_job(fingerprint) else "pending_reconciliation"
    except (OSError, ValueError, sqlite3.Error):
        return "pending_reconciliation"


def capture_requested_score(application, resume):
    requested = application.get("score_application", True)
    if not isinstance(requested, bool):
        raise ValueError("score_application must be a boolean")
    if not requested:
        return "not_requested", None
    try:
        return capture_scoring_input(application, resume)
    except (OSError, ValueError, sqlite3.Error):
        return "unavailable", None


def score_with_identity(result, scoring_input, input_fingerprint):
    return {
        **result,
        "application_key": scoring_input["application_key"],
        "input_fingerprint": input_fingerprint,
    }


def score_from_notion_cache(cache_key, client=None):
    if not NOTION_SCORE_DATA_SOURCE_ID:
        return None
    pages = (client or notion_client()).query_data_source(
        NOTION_SCORE_DATA_SOURCE_ID,
        {"property": "Cache key", "rich_text": {"equals": cache_key}},
    )
    if len(pages) > 1:
        raise RuntimeError("Cache key matches multiple Resume Match Scores rows")
    return score_from_notion_page(pages[0]) if pages else None


def score_application_now(
    application,
    resume,
    store=None,
    settings=None,
    notion=None,
    force_recompute=False,
):
    if not isinstance(force_recompute, bool):
        raise ValueError("force_recompute must be a boolean")
    settings = settings or ScoringSettings.from_env()
    description = str(application.get("job_description", "")).strip()
    if (
        len(description) < settings.min_job_description_chars
        or application.get("extraction_quality") not in {None, "", "complete"}
    ):
        raise ValueError("A complete job description is required to calculate a score")
    store = store or ScoringStore(settings.database_path)
    store.initialize()
    scoring_input = prepare_scoring_input(application, resume, settings)
    fingerprint = score_input_fingerprint(
        scoring_input,
        settings.scorer_version,
        settings.embedding_model,
        settings.scoring_model_revision,
    )
    cache_key = score_cache_key(
        scoring_input,
        settings.scorer_version,
        settings.embedding_model,
        settings.scoring_model_revision,
    )
    if not force_recompute:
        cached = store.get_cached_score(cache_key)
        if cached:
            return score_with_identity(cached, scoring_input, fingerprint), "cached"
        fetched = score_from_notion_cache(cache_key, client=notion)
        if fetched:
            fetched.update({
                "scorer_version": settings.scorer_version,
                "embedding_model": settings.embedding_model,
                "deployment_revision": settings.scoring_model_revision,
            })
            store.save_cached_score(
                cache_key, fetched, getattr(settings, "score_cache_ttl_seconds", 300)
            )
            return score_with_identity(fetched, scoring_input, fingerprint), "fetched"
    scorer = create_scoring_client(settings)
    result = scorer.score(scoring_input)
    store.save_cached_score(
        cache_key, result, getattr(settings, "score_cache_ttl_seconds", 300)
    )
    return result, "computed"


def resume_catalog():
    return build_resume_catalog(RESUMES_DIR)


def create_notion_page(application):
    if not NOTION_TOKEN or not NOTION_DATABASE_ID:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATABASE_ID are required")

    properties = {
        "Company": {"title": [{"text": {"content": application["company"]}}]},
        "Job title": {"rich_text": [{"text": {"content": application["job_title"]}}]},
        "Job URL": {"url": application["job_url"]},
        "Job source": {"select": {"name": application["job_source"]}},
        "Application date": {"date": {"start": application["application_date"]}},
        "Resume version": {"select": {"name": application["resume_version_label"]}},
        "Status": {"status": {"name": application["status"]}},
        "Requested for sponsorship": {"checkbox": application["requested_for_sponsorship"]},
        "Application key": {"rich_text": [{"text": {"content": application["application_key"]}}]},
        "Referral": {"checkbox": application["referral"]},
        "Job description": {"rich_text": notion_rich_text(application.get("job_description", ""))},
    }
    return notion_client("2022-06-28").create_page(NOTION_DATABASE_ID, properties)


def find_notion_application(key):
    if not NOTION_TOKEN or not NOTION_DATA_SOURCE_ID:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATA_SOURCE_ID are required")
    return notion_client().query_data_source(
        NOTION_DATA_SOURCE_ID,
        {"property": "Application key", "rich_text": {"equals": key}},
    )


def dashboard_counts(reference_date, client=None):
    if not NOTION_TOKEN or not NOTION_DATA_SOURCE_ID:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATA_SOURCE_ID are required")
    week_start = reference_date - timedelta(days=reference_date.weekday())
    month_start = reference_date.replace(day=1)
    query_start = min(week_start, month_start)
    pages = (client or notion_client()).query_data_source(
        NOTION_DATA_SOURCE_ID,
        {
            "and": [
                {"property": "Application date", "date": {"on_or_after": query_start.isoformat()}},
                {"property": "Application date", "date": {"on_or_before": reference_date.isoformat()}},
            ]
        },
    )
    application_dates = [value for page in pages if (value := application_date_from_page(page))]
    return {
        "current_day": sum(value == reference_date for value in application_dates),
        "current_week": sum(week_start <= value <= reference_date for value in application_dates),
        "current_month": sum(month_start <= value <= reference_date for value in application_dates),
    }


def monthly_application_summary(month_start, month_end, client=None):
    if not NOTION_TOKEN or not NOTION_DATA_SOURCE_ID:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATA_SOURCE_ID are required")
    pages = (client or notion_client()).query_data_source(
        NOTION_DATA_SOURCE_ID,
        {
            "and": [
                {"property": "Application date", "date": {"on_or_after": month_start.isoformat()}},
                {"property": "Application date", "date": {"on_or_before": month_end.isoformat()}},
            ]
        },
    )
    status_counts = {key: 0 for key in STATUS_COUNT_KEYS.values()}
    total = 0
    for page in pages:
        application_date = application_date_from_page(page)
        if not application_date or not month_start <= application_date <= month_end:
            continue
        total += 1
        status = notion_select_value(page.get("properties", {}).get("Status", {}))
        if key := STATUS_COUNT_KEYS.get(status):
            status_counts[key] += 1
    return {"total": total, "statuses": status_counts}


def monthly_application_count(month_start, month_end, client=None):
    return monthly_application_summary(month_start, month_end, client=client)["total"]


def total_application_count(client=None):
    if not NOTION_TOKEN or not NOTION_DATA_SOURCE_ID:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATA_SOURCE_ID are required")
    return len((client or notion_client()).query_data_source(NOTION_DATA_SOURCE_ID))


def application_search_catalog(client=None):
    if not NOTION_TOKEN or not NOTION_DATA_SOURCE_ID:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATA_SOURCE_ID are required")
    pages = (client or notion_client()).query_data_source(NOTION_DATA_SOURCE_ID)
    applications = [application_from_notion_page(page) for page in pages]
    applications = [item for item in applications if item["application_key"]]
    applications.sort(
        key=lambda item: (item["application_date"], item["created_at"]),
        reverse=True,
    )
    summary_fields = (
        "application_key", "company", "job_title", "status", "application_date", "job_source"
    )
    return [{field: item[field] for field in summary_fields} for item in applications]


def application_insights(key, client=None):
    if not NOTION_TOKEN or not NOTION_DATA_SOURCE_ID:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATA_SOURCE_ID are required")
    client = client or notion_client()
    application_pages = client.query_data_source(
        NOTION_DATA_SOURCE_ID,
        {"property": "Application key", "rich_text": {"equals": key}},
    )
    if not application_pages:
        return None
    if len(application_pages) > 1:
        raise RuntimeError("Application key matches multiple Job Applications rows")
    score = None
    if NOTION_SCORE_DATA_SOURCE_ID:
        score_pages = client.query_data_source(
            NOTION_SCORE_DATA_SOURCE_ID,
            {"property": "Application key", "title": {"equals": key}},
        )
        if len(score_pages) > 1:
            raise RuntimeError("Application key matches multiple Resume Match Scores rows")
        if score_pages:
            score = score_from_notion_page(score_pages[0])
    return {"application": application_from_notion_page(application_pages[0]), "logged_score": score}


def update_application_status(key, status, client=None, write_client=None):
    if status not in VALID_STATUSES:
        raise ValueError("Invalid application status")
    if not NOTION_TOKEN or not NOTION_DATA_SOURCE_ID:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATA_SOURCE_ID are required")
    client = client or notion_client()
    write_client = write_client or client
    application_pages = client.query_data_source(
        NOTION_DATA_SOURCE_ID,
        {"property": "Application key", "rich_text": {"equals": key}},
    )
    if not application_pages:
        return None
    if len(application_pages) > 1:
        raise RuntimeError("Application key matches multiple Job Applications rows")
    updated = write_client.update_page(
        application_pages[0]["id"],
        {"Status": {"status": {"name": status}}},
    )
    return {
        "application_key": key,
        "status": status,
        "updated_at": str(updated.get("last_edited_time") or ""),
    }


def update_application_details(key, payload, client=None, write_client=None, store=None):
    if not NOTION_TOKEN or not NOTION_DATA_SOURCE_ID:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATA_SOURCE_ID are required")
    client = client or notion_client()
    write_client = write_client or client
    application_pages = client.query_data_source(
        NOTION_DATA_SOURCE_ID,
        {"property": "Application key", "rich_text": {"equals": key}},
    )
    if not application_pages:
        return None
    if len(application_pages) > 1:
        raise RuntimeError("Application key matches multiple Job Applications rows")
    page = application_pages[0]
    existing = application_from_notion_page(page)
    updated = normalize_application_update(existing, payload)
    if updated["application_key"] != key:
        duplicates = client.query_data_source(
            NOTION_DATA_SOURCE_ID,
            {"property": "Application key", "rich_text": {"equals": updated["application_key"]}},
        )
        if any(item.get("id") != page.get("id") for item in duplicates):
            raise ValueError("Updated application identity matches another application")
    updated_page = write_client.update_page(page["id"], application_update_properties(updated))
    if updated["application_key"] != key:
        if NOTION_SCORE_DATA_SOURCE_ID:
            score_pages = client.query_data_source(
                NOTION_SCORE_DATA_SOURCE_ID,
                {"property": "Application key", "title": {"equals": key}},
            )
            for score_page in score_pages:
                write_client.update_page(
                    score_page["id"],
                    {"Application key": {"title": [{"text": {"content": updated["application_key"]}}]}},
                )
        if store is None:
            settings = ScoringSettings.from_env()
            store = ScoringStore(settings.database_path)
            store.initialize()
        store.rename_application(key, updated["application_key"])
    return {
        "application": {
            **updated,
            "resume_version": existing.get("resume_version", ""),
            "created_at": existing.get("created_at", ""),
            "updated_at": str(updated_page.get("last_edited_time") or ""),
        }
    }


def delete_application(key, client=None, write_client=None, store=None):
    if not NOTION_TOKEN or not NOTION_DATA_SOURCE_ID:
        raise RuntimeError("NOTION_TOKEN and NOTION_DATA_SOURCE_ID are required")
    client = client or notion_client()
    write_client = write_client or client
    application_pages = client.query_data_source(
        NOTION_DATA_SOURCE_ID,
        {"property": "Application key", "rich_text": {"equals": key}},
    )
    if not application_pages:
        return None
    if len(application_pages) > 1:
        raise RuntimeError("Application key matches multiple Job Applications rows")
    score_pages = []
    if NOTION_SCORE_DATA_SOURCE_ID:
        score_pages = client.query_data_source(
            NOTION_SCORE_DATA_SOURCE_ID,
            {"property": "Application key", "title": {"equals": key}},
        )
    if store is None:
        settings = ScoringSettings.from_env()
        store = ScoringStore(settings.database_path)
        store.initialize()
    local_scoring_deleted = store.delete_application(key)
    for page in score_pages:
        write_client.trash_page(page["id"])
    write_client.trash_page(application_pages[0]["id"])
    return {
        "application_key": key,
        "scores_deleted": len(score_pages),
        "local_scoring_deleted": local_scoring_deleted,
    }


class Handler(BaseHTTPRequestHandler):
    def _authorized(self):
        return not API_KEY or self.headers.get("X-Job-Tracker-Key") == API_KEY

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Job-Tracker-Key")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Job-Tracker-Key")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            try:
                settings = ScoringSettings.from_env()
                store = ScoringStore(settings.database_path)
                store.initialize()
                sqlite_ready = store.is_healthy()
            except (OSError, ValueError, sqlite3.Error):
                sqlite_ready = False
                settings = None
            self._send_json(
                {
                    "ok": sqlite_ready,
                    "api": "ready",
                    "sqlite": "ready" if sqlite_ready else "unavailable",
                    "scoring_backend": settings.scoring_backend if settings else "unavailable",
                },
                200 if sqlite_ready else 503,
            )
            return
        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, 401)
            return
        if parsed.path == "/api/resumes":
            self._send_json({"resumes": [{"version": item["version"], "label": item["label"]} for item in resume_catalog()]})
            return
        if parsed.path == "/api/applications/search":
            try:
                self._send_json({"applications": application_search_catalog()})
            except (RuntimeError, OSError) as error:
                self._send_json({"error": str(error)}, 500)
            return
        if parsed.path.startswith("/api/applications/insights/"):
            key = unquote(parsed.path.removeprefix("/api/applications/insights/")).strip()
            if not key:
                self._send_json({"error": "Application key is required"}, 400)
                return
            try:
                insights = application_insights(key)
                if insights is None:
                    self._send_json({"error": "Application not found"}, 404)
                    return
                self._send_json(insights)
            except (RuntimeError, OSError) as error:
                self._send_json({"error": str(error)}, 500)
            return
        if parsed.path == "/api/dashboard":
            query = parse_qs(parsed.query)
            requested_date = query.get("date", [""])[0]
            requested_month = query.get("month", [""])[0]
            try:
                reference_date = date.fromisoformat(requested_date) if requested_date else datetime.now(timezone.utc).date()
                month_start, month_end = month_bounds(requested_month or reference_date.strftime("%Y-%m"))
            except ValueError:
                self._send_json({"error": "Dashboard date must use YYYY-MM-DD and month must use YYYY-MM"}, 400)
                return
            try:
                client = notion_client()
                counts = dashboard_counts(reference_date, client=client)
                selected_month_end = (
                    reference_date
                    if month_start == reference_date.replace(day=1)
                    else month_end
                )
                month_summary = monthly_application_summary(
                    month_start, selected_month_end, client=client
                )
                counts["selected_month"] = month_summary["total"]
                counts["statuses"] = month_summary["statuses"]
                counts["applications_so_far"] = total_application_count(client=client)
                self._send_json({
                    "date": reference_date.isoformat(),
                    "month": month_start.strftime("%Y-%m"),
                    "month_label": month_start.strftime("%B %Y"),
                    "counts": counts,
                })
            except (RuntimeError, OSError) as error:
                self._send_json({"error": str(error)}, 500)
            return
        if parsed.path.startswith("/resume/"):
            version = unquote(parsed.path.removeprefix("/resume/"))
            match = next((item for item in resume_catalog() if item["version"] == version), None)
            if not match:
                self._send_json({"error": "Resume not found"}, 404)
                return
            data = Path(match["path"]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f'inline; filename="{version}.pdf"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._send_json({"error": "Not found"}, 404)

    def do_PATCH(self):
        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, 401)
            return
        parsed = urlparse(self.path)
        prefix = "/api/applications/"
        suffix = "/status"
        if not parsed.path.startswith(prefix):
            self._send_json({"error": "Not found"}, 404)
            return
        is_status_update = parsed.path.endswith(suffix)
        key_end = -len(suffix) if is_status_update else None
        key = unquote(parsed.path[len(prefix):key_end]).strip("/")
        if not key:
            self._send_json({"error": "Application key is required"}, 400)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if is_status_update:
                if not isinstance(payload, dict) or set(payload) != {"status"}:
                    raise ValueError("Only status can be updated")
                result = update_application_status(key, str(payload["status"]))
            else:
                result = update_application_details(key, payload)
            if result is None:
                self._send_json({"error": "Application not found"}, 404)
                return
            self._send_json({"ok": True, **result})
        except (json.JSONDecodeError, ValueError) as error:
            self._send_json({"error": str(error)}, 400)
        except (RuntimeError, OSError, sqlite3.Error) as error:
            self._send_json({"error": str(error)}, 500)

    def do_DELETE(self):
        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, 401)
            return
        parsed = urlparse(self.path)
        prefix = "/api/applications/"
        if not parsed.path.startswith(prefix):
            self._send_json({"error": "Not found"}, 404)
            return
        key = unquote(parsed.path.removeprefix(prefix)).strip("/")
        if not key:
            self._send_json({"error": "Application key is required"}, 400)
            return
        try:
            result = delete_application(key)
            if result is None:
                self._send_json({"error": "Application not found"}, 404)
                return
            self._send_json({"ok": True, **result})
        except (RuntimeError, OSError, sqlite3.Error) as error:
            self._send_json({"error": str(error)}, 500)

    def do_POST(self):
        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, 401)
            return
        if self.path not in {
            "/api/applications", "/api/applications/check", "/api/applications/score",
            "/api/match-scores",
        }:
            self._send_json({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            application = json.loads(self.rfile.read(length))
            if self.path == "/api/match-scores":
                settings = ScoringSettings.from_env()
                store = ScoringStore(settings.database_path)
                store.initialize()
                outcome = persist_score_result(
                    store,
                    notion_client(),
                    NOTION_DATA_SOURCE_ID,
                    NOTION_SCORE_DATABASE_ID,
                    NOTION_SCORE_DATA_SOURCE_ID,
                    application,
                    write_client=notion_client("2022-06-28"),
                )
                self._send_json({"ok": True, "outcome": outcome}, 201 if outcome == "created" else 200)
                return
            if self.path == "/api/applications/score":
                score_fields = ("job_url", "resume_version", "job_description")
                missing = [field for field in score_fields if not str(application.get(field, "")).strip()]
                if missing:
                    self._send_json({"error": f"Missing fields: {', '.join(missing)}"}, 400)
                    return
                key = preview_application_key(application)
                resume = next(
                    (item for item in resume_catalog() if item["version"] == application["resume_version"]),
                    None,
                )
                if not resume:
                    self._send_json({"error": "Unknown resume version"}, 400)
                    return
                force_recompute = application.get("force_recompute", False)
                if not isinstance(force_recompute, bool):
                    self._send_json({"error": "force_recompute must be a boolean"}, 400)
                    return
                application["application_key"] = key
                result, score_source = score_application_now(
                    application, resume, force_recompute=force_recompute
                )
                self._send_json({
                    "application_key": key,
                    "state": "completed",
                    "preview": True,
                    "score": result["score"],
                    "band": result["band"],
                    "fit_label": result.get("fit_label"),
                    "fit_probability": result.get("fit_probability"),
                    "decision_threshold": result.get("decision_threshold"),
                    "matched_terms": result["matched_terms"],
                    "category_breakdown": result["category_breakdown"],
                    "top_gaps": result.get("top_gaps", []),
                    "score_source": score_source,
                })
                return
            required = ("company", "job_title", "job_url", "job_source", "resume_version")
            missing = [field for field in required if not str(application.get(field, "")).strip()]
            if missing:
                self._send_json({"error": f"Missing fields: {', '.join(missing)}"}, 400)
                return
            application["application_key"] = application_key(application["company"], application["job_title"], application["job_source"])
            existing = find_notion_application(application["application_key"])
            if self.path == "/api/applications/check":
                self._send_json(application_check_response(existing, application["application_key"]))
                return
            if existing:
                self._send_json({"error": "This application already exists in Notion", "already_applied": True}, 409)
                return
            resume = next((item for item in resume_catalog() if item["version"] == application["resume_version"]), None)
            if not resume:
                self._send_json({"error": "Unknown resume version"}, 400)
                return
            application["resume_version_label"] = resume["label"]
            application["application_date"] = str(application.get("application_date", "")).strip() or datetime.now(timezone.utc).date().isoformat()
            try:
                datetime.strptime(application["application_date"], "%Y-%m-%d")
            except ValueError:
                self._send_json({"error": "Application date must use YYYY-MM-DD"}, 400)
                return
            application["status"] = str(application.get("status", "Applied")).strip() or "Applied"
            if application["status"] not in VALID_STATUSES:
                self._send_json({"error": "Invalid application status"}, 400)
                return
            application["requested_for_sponsorship"] = application.get("requested_for_sponsorship", False) is True
            application["referral"] = application.get("referral", False) is True
            scoring_input_status, queued_fingerprint = capture_requested_score(
                application, resume
            )
            page = create_notion_page(application)
            scoring_status = (
                activate_captured_score(queued_fingerprint)
                if queued_fingerprint else scoring_input_status
            )
            self._send_json({
                "ok": True,
                "notion_page_id": page.get("id"),
                "application_key": application["application_key"],
                "scoring_input_status": scoring_input_status,
                "scoring_status": scoring_status,
            }, 201)
        except ScorePersistenceError as error:
            self._send_json({"error": str(error)}, 400)
        except ValueError as error:
            self._send_json({"error": str(error)}, 400)
        except (RuntimeError, OSError) as error:
            self._send_json({"error": str(error)}, 500)

    def log_message(self, format, *args):
        print(format % args)


if __name__ == "__main__":
    print(f"Serving job tracker on http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
