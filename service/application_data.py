import hashlib
import json
import re
from datetime import date, datetime, timedelta

VALID_STATUSES = {"Applied", "In Progress", "Rejected", "Cancelled"}
STATUS_COUNT_KEYS = {
    "Applied": "applied",
    "In Progress": "in_progress",
    "Rejected": "rejected",
    "Cancelled": "cancelled",
}


def application_key(company, job_title, job_source):
    values = (company, job_title, job_source)
    value = "|".join(" ".join(str(item).strip().lower().split()) for item in values)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def preview_application_key(application):
    return application_key(
        application.get("company") or application.get("canonical_url") or application["job_url"],
        application.get("job_title") or application.get("source_job_id") or "job",
        application.get("job_source") or "preview",
    )


def notion_rich_text(value, chunk_size=2000):
    text = str(value).strip()
    if chunk_size < 2:
        raise ValueError("Notion rich-text chunk size must be at least 2")
    chunks = []
    current = []
    current_units = 0
    for character in text:
        character_units = 2 if ord(character) > 0xFFFF else 1
        if current and current_units + character_units > chunk_size:
            chunks.append({"text": {"content": "".join(current)}})
            current = []
            current_units = 0
        current.append(character)
        current_units += character_units
    if current:
        chunks.append({"text": {"content": "".join(current)}})
    return chunks


def application_update_properties(application):
    return {
        "Company": {"title": [{"text": {"content": application["company"]}}]},
        "Job title": {"rich_text": [{"text": {"content": application["job_title"]}}]},
        "Job URL": {"url": application["job_url"]},
        "Job source": {"select": {"name": application["job_source"]}},
        "Application date": {"date": {"start": application["application_date"]}},
        "Status": {"status": {"name": application["status"]}},
        "Requested for sponsorship": {"checkbox": application["requested_for_sponsorship"]},
        "Application key": {
            "rich_text": [{"text": {"content": application["application_key"]}}]
        },
        "Referral": {"checkbox": application["referral"]},
        "Job description": {
            "rich_text": notion_rich_text(application.get("job_description", ""))
        },
    }


def notion_text_value(property_value):
    items = property_value.get("title") or property_value.get("rich_text") or []
    return "".join(
        str(item.get("plain_text") or item.get("text", {}).get("content") or "")
        for item in items
    )


def notion_select_value(property_value):
    selected = property_value.get("status") or property_value.get("select") or {}
    return str(selected.get("name") or "")


def application_from_notion_page(page):
    properties = page.get("properties", {})
    return {
        "application_key": notion_text_value(properties.get("Application key", {})),
        "company": notion_text_value(properties.get("Company", {})),
        "job_title": notion_text_value(properties.get("Job title", {})),
        "job_url": str(properties.get("Job URL", {}).get("url") or ""),
        "job_source": notion_select_value(properties.get("Job source", {})),
        "application_date": str(
            (properties.get("Application date", {}).get("date") or {}).get("start") or ""
        )[:10],
        "resume_version": notion_select_value(properties.get("Resume version", {})),
        "status": notion_select_value(properties.get("Status", {})),
        "requested_for_sponsorship": bool(
            properties.get("Requested for sponsorship", {}).get("checkbox", False)
        ),
        "referral": bool(properties.get("Referral", {}).get("checkbox", False)),
        "job_description": notion_text_value(properties.get("Job description", {})),
        "created_at": str(page.get("created_time") or ""),
        "updated_at": str(page.get("last_edited_time") or ""),
    }


def score_from_notion_page(page):
    properties = page.get("properties", {})

    def structured_value(name, expected_type):
        raw = notion_text_value(properties.get(name, {}))
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return expected_type()
        return value if isinstance(value, expected_type) else expected_type()

    result = {
        "score": properties.get("Score", {}).get("number"),
        "band": notion_select_value(properties.get("Band", {})),
        "matched_terms": structured_value("Matched terms", list),
        "category_breakdown": structured_value("Category breakdown", dict),
        "created_at": str(page.get("created_time") or ""),
        "updated_at": str(page.get("last_edited_time") or ""),
    }
    optional = {
        "fit_label": notion_text_value(properties.get("Fit label", {})),
        "fit_probability": properties.get("Fit probability", {}).get("number"),
        "decision_threshold": properties.get("Decision threshold", {}).get("number"),
    }
    result.update({name: value for name, value in optional.items() if value not in {None, ""}})
    return result


def application_check_response(existing, key):
    response = {"exists": bool(existing), "application_key": key}
    if existing:
        response["status"] = application_from_notion_page(existing[0]).get("status", "")
    return response


def application_date_from_page(page):
    date_property = page.get("properties", {}).get("Application date", {}).get("date") or {}
    value = date_property.get("start", "")
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def month_bounds(value):
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise ValueError("Dashboard month must use YYYY-MM")
    try:
        start = date.fromisoformat(f"{value}-01")
    except ValueError as error:
        raise ValueError("Dashboard month must use YYYY-MM") from error
    next_year = start.year + 1 if start.month == 12 else start.year
    next_month_number = 1 if start.month == 12 else start.month + 1
    next_month = date(next_year, next_month_number, 1)
    return start, next_month - timedelta(days=1)


def normalize_application_update(existing, payload):
    if not isinstance(payload, dict):
        raise ValueError("Application update must be an object")
    allowed = {
        "company",
        "job_title",
        "job_url",
        "job_source",
        "application_date",
        "status",
        "requested_for_sponsorship",
        "referral",
        "job_description",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown application fields: {', '.join(sorted(unknown))}")
    updated = {field: existing.get(field, "") for field in allowed}
    updated.update(payload)
    for field in ("company", "job_title", "job_url", "job_source"):
        updated[field] = str(updated.get(field, "")).strip()
        if not updated[field]:
            raise ValueError(f"{field} is required")
    updated["application_date"] = str(updated.get("application_date", "")).strip()
    try:
        datetime.strptime(updated["application_date"], "%Y-%m-%d")
    except ValueError as error:
        raise ValueError("Application date must use YYYY-MM-DD") from error
    updated["status"] = str(updated.get("status", "")).strip()
    if updated["status"] not in VALID_STATUSES:
        raise ValueError("Invalid application status")
    for field in ("requested_for_sponsorship", "referral"):
        if not isinstance(updated[field], bool):
            raise ValueError(f"{field} must be a boolean")
    updated["job_description"] = str(updated.get("job_description", "")).strip()
    updated["application_key"] = application_key(
        updated["company"], updated["job_title"], updated["job_source"]
    )
    return updated
