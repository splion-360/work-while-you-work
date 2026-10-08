from collections import Counter
from pathlib import Path

from service.scoring_contract import file_hash, score_input_fingerprint


def _property_text(page, name):
    prop = page.get("properties", {}).get(name, {})
    values = prop.get("title") or prop.get("rich_text") or []
    return "".join(item.get("plain_text", item.get("text", {}).get("content", "")) for item in values).strip()


def _notion_key_counts(client, source_id):
    return Counter(
        key for page in client.query_data_source(source_id) if (key := _property_text(page, "Application key"))
    )


def reconcile(
    store, notion, applications_source_id, scores_source_id, scorer_version,
    embedding_model, min_description_chars=1, publication_uncertainty_seconds=900,
    enqueue_missing_scores=True, deployment_revision="",
):
    application_keys = _notion_key_counts(notion, applications_source_id)
    score_keys = _notion_key_counts(notion, scores_source_id)
    base_inputs = store.list_scoring_inputs()
    inputs_by_fingerprint = {
        item["input_fingerprint"]: item for item in store.list_scoring_snapshots()
    }
    for scoring_input in base_inputs:
        fingerprint = score_input_fingerprint(
            scoring_input, scorer_version, embedding_model, deployment_revision
        )
        store.save_scoring_snapshot(fingerprint, scoring_input)
        inputs_by_fingerprint.setdefault(fingerprint, scoring_input)
    inputs = list(inputs_by_fingerprint.values())
    jobs = store.list_score_jobs()
    existing_fingerprints = {job["input_fingerprint"] for job in jobs}
    activated = 0
    requeued = 0
    unavailable_inputs = 0

    for job in jobs:
        if job["status"] == "waiting_application" and application_keys[job["application_key"]] == 1:
            activated += int(store.activate_score_job(job["input_fingerprint"]))

    for scoring_input in inputs if enqueue_missing_scores else ():
        key = scoring_input["application_key"]
        if application_keys[key] != 1:
            continue
        resume_path = Path(scoring_input["resume_path"])
        if (
            len(scoring_input["job_description"].strip()) < min_description_chars
            or scoring_input["extraction_quality"] not in {"complete", "partial"}
            or not resume_path.is_file()
        ):
            unavailable_inputs += 1
            continue
        try:
            snapshot_valid = file_hash(resume_path) == scoring_input["resume_content_hash"]
        except OSError:
            snapshot_valid = False
        if not snapshot_valid:
            unavailable_inputs += 1
            continue
        fingerprint = score_input_fingerprint(
            scoring_input, scorer_version, embedding_model, deployment_revision
        )
        if fingerprint not in existing_fingerprints:
            store.enqueue_score_job(key, fingerprint, status="queued")
            existing_fingerprints.add(fingerprint)
            requeued += 1

    recovered_leases = store.recover_expired_leases()
    recovered_publications = 0
    retried_publications = 0
    uncertain_publications = 0
    for job in store.list_expired_publications():
        score_count = score_keys[job["application_key"]]
        if score_count == 1:
            recovered_publications += int(store.complete_score_job(job["id"]))
        elif score_count == 0:
            publication_requeued = store.requeue_expired_publication(
                job["id"], publication_uncertainty_seconds
            )
            retried_publications += int(publication_requeued)
            uncertain_publications += int(not publication_requeued)
    local_keys = {item["application_key"] for item in base_inputs}
    stale_versions = sum(
        run["scorer_version"] != scorer_version
        or run["embedding_model"] != embedding_model
        or run.get("deployment_revision", "") != deployment_revision
        for run in store.list_all_score_runs()
    )
    return {
        "activated": activated,
        "requeued": requeued,
        "recovered_leases": recovered_leases,
        "recovered_publications": recovered_publications,
        "retried_publications": retried_publications,
        "uncertain_publications": uncertain_publications,
        "unavailable_inputs": unavailable_inputs,
        "duplicate_application_keys": sum(count > 1 for count in application_keys.values()),
        "duplicate_score_keys": sum(count > 1 for count in score_keys.values()),
        "orphan_score_keys": sum(key not in application_keys for key in score_keys),
        "missing_local_inputs": sum(key not in local_keys for key in application_keys),
        "stale_score_runs": stale_versions,
    }
