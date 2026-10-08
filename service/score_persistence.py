import json
import math
from contextlib import nullcontext

try:
    from service.scorer import score_band, score_cache_key, score_input_fingerprint
except ModuleNotFoundError:
    from scorer import score_band, score_cache_key, score_input_fingerprint


class ScorePersistenceError(RuntimeError):
    pass


def normalize_score_result(result):
    required = (
        "application_key", "input_fingerprint", "scorer_version", "embedding_model",
        "score", "band", "matched_terms", "category_breakdown",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise ScorePersistenceError(f"Missing score fields: {', '.join(missing)}")
    try:
        score = float(result["score"])
    except (TypeError, ValueError) as error:
        raise ScorePersistenceError("Score must be a number") from error
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ScorePersistenceError("Score must be between 0 and 100")
    band = str(result["band"])
    expected_band = score_band(score)
    if band != expected_band:
        raise ScorePersistenceError(f"Band must be {expected_band} for score {score:g}")
    if not isinstance(result["matched_terms"], list):
        raise ScorePersistenceError("Matched terms must be an array")
    if any(not isinstance(term, str) for term in result["matched_terms"]):
        raise ScorePersistenceError("Matched terms must contain only strings")
    matched_terms = sorted({term.strip() for term in result["matched_terms"] if term.strip()}, key=str.lower)
    if not isinstance(result["category_breakdown"], dict):
        raise ScorePersistenceError("Category breakdown must be an object")
    categories = {}
    for name, value in result["category_breakdown"].items():
        if not isinstance(name, str) or not name.strip():
            raise ScorePersistenceError("Category names must be non-empty strings")
        normalized_name = name.strip()
        if normalized_name in categories:
            raise ScorePersistenceError("Category names must be unique after trimming")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ScorePersistenceError(f"Category {name} must be a number") from error
        if not math.isfinite(numeric) or not 0 <= numeric <= 100:
            raise ScorePersistenceError(f"Category {name} must be between 0 and 100")
        categories[normalized_name] = numeric
    normalized = {
        "application_key": str(result["application_key"]).strip(),
        "input_fingerprint": str(result["input_fingerprint"]).strip(),
        "scorer_version": str(result["scorer_version"]).strip(),
        "embedding_model": str(result["embedding_model"]).strip(),
        "score": score,
        "band": band,
        "matched_terms": matched_terms,
        "category_breakdown": dict(sorted(categories.items())),
        "deployment_revision": str(result.get("deployment_revision", "")).strip(),
        "fit_label": str(result.get("fit_label", "")).strip(),
        "fit_probability": result.get("fit_probability"),
        "decision_threshold": result.get("decision_threshold"),
    }
    for name in ("application_key", "input_fingerprint", "scorer_version", "embedding_model"):
        if not normalized[name]:
            raise ScorePersistenceError(f"{name.replace('_', ' ').title()} is required")
    return normalized


def notion_properties(result):
    matched_json = json.dumps(result["matched_terms"], ensure_ascii=True, separators=(",", ":"))
    category_json = json.dumps(result["category_breakdown"], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if len(matched_json) > 2000 or len(category_json) > 2000:
        raise ScorePersistenceError("Structured score fields exceed Notion rich-text limits")
    return {
        "Application key": {"title": [{"text": {"content": result["application_key"]}}]},
        "Score": {"number": result["score"]},
        "Band": {"select": {"name": result["band"]}},
        "Matched terms": {"rich_text": [{"text": {"content": matched_json}}]},
        "Category breakdown": {"rich_text": [{"text": {"content": category_json}}]},
        "Cache key": {"rich_text": [{"text": {"content": result["cache_key"]}}]},
        "Fit label": {
            "rich_text": (
                [{"text": {"content": result["fit_label"]}}]
                if result["fit_label"] else []
            )
        },
        "Fit probability": {"number": result["fit_probability"]},
        "Decision threshold": {"number": result["decision_threshold"]},
    }


def persist_score_result(
    store,
    query_client,
    applications_source_id,
    scores_database_id,
    scores_source_id,
    result,
    write_client=None,
    use_lock=True,
):
    result = normalize_score_result(result)
    write_client = write_client or query_client
    lock = store.publication_lock() if use_lock else nullcontext()
    with lock:
        return _persist_score_result(
            store,
            query_client,
            applications_source_id,
            scores_database_id,
            scores_source_id,
            result,
            write_client,
        )


def _persist_score_result(
    store, query_client, applications_source_id, scores_database_id,
    scores_source_id, result, write_client,
):
    application_pages = query_client.query_data_source(
        applications_source_id,
        {"property": "Application key", "rich_text": {"equals": result["application_key"]}},
    )
    if not application_pages:
        raise ScorePersistenceError("Application key does not exist in Job Applications")
    if len(application_pages) > 1:
        raise ScorePersistenceError("Application key matches multiple Job Applications rows")
    snapshot = store.get_scoring_snapshot(result["input_fingerprint"])
    if not snapshot or snapshot["application_key"] != result["application_key"]:
        raise ScorePersistenceError("Input fingerprint does not match a stored scoring snapshot")
    expected_fingerprint = score_input_fingerprint(
        snapshot,
        result["scorer_version"],
        result["embedding_model"],
        result["deployment_revision"],
    )
    if expected_fingerprint != result["input_fingerprint"]:
        raise ScorePersistenceError("Score metadata does not match its stored scoring snapshot")
    result["cache_key"] = score_cache_key(
        snapshot,
        result["scorer_version"],
        result["embedding_model"],
        result["deployment_revision"],
    )
    score_pages = query_client.query_data_source(
        scores_source_id,
        {"property": "Application key", "title": {"equals": result["application_key"]}},
    )
    if len(score_pages) > 1:
        raise ScorePersistenceError("Application key matches multiple Resume Match Scores rows")

    run = store.save_score_run(result)
    if not run["is_application_time"]:
        return "stored_locally"
    properties = notion_properties(result)
    if score_pages:
        write_client.update_page(score_pages[0]["id"], properties)
        return "updated"
    write_client.create_page(scores_database_id, properties)
    return "created"
