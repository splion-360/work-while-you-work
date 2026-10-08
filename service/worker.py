#!/usr/bin/env python3
import os
import random
import socket
import time
import uuid
import json

from service.config import ScoringSettings
from service.notion import NotionClient
from service.reconciliation import reconcile
from service.score_persistence import persist_score_result
from service.scoring_client import create_scoring_client
from service.scoring_contract import score_cache_key
from service.scoring_state import ScoringStore


def process_one_job(
    store,
    scorer,
    persist,
    worker_id,
    lease_seconds=300,
    max_attempts=5,
    retry_delay_seconds=5,
    retry_jitter_seconds=2,
    min_description_chars=1,
    cache_ttl_seconds=300,
):
    job = store.claim_score_job(worker_id, lease_seconds=lease_seconds)
    if not job:
        return False
    try:
        scoring_input = store.get_scoring_snapshot(job["input_fingerprint"])
        if not scoring_input:
            raise RuntimeError("Scoring input no longer exists")
        if (
            len(scoring_input["job_description"].strip()) < min_description_chars
            or scoring_input["extraction_quality"] not in {"complete", "partial"}
        ):
            raise RuntimeError("Scoring input does not meet extraction quality requirements")
        result = store.get_score_result(job["input_fingerprint"])
        if result is None:
            cache_key = score_cache_key(
                scoring_input,
                getattr(scorer, "scorer_version", ""),
                getattr(scorer, "embedding_model", ""),
                getattr(scorer, "fingerprint_revision", ""),
            )
            cached = store.get_cached_score(cache_key)
            if cached:
                result = {
                    **cached,
                    "application_key": scoring_input["application_key"],
                    "input_fingerprint": job["input_fingerprint"],
                }
            else:
                result = scorer.score(scoring_input)
                store.save_cached_score(cache_key, result, cache_ttl_seconds)
            store.save_score_run(result)
        if result["input_fingerprint"] != job["input_fingerprint"]:
            raise RuntimeError("Scorer input fingerprint does not match queued job")
        with store.publication_lock():
            if not store.begin_score_publication(
                job["id"], worker_id, job["attempt_count"], lease_seconds
            ):
                return True
            persist(result)
            store.complete_score_job(job["id"], worker_id, job["attempt_count"])
    except Exception as error:
        delay = retry_delay_seconds * (2 ** max(0, job["attempt_count"] - 1))
        delay += random.uniform(0, retry_jitter_seconds)
        store.fail_score_job(
            job["id"], error, max_attempts, delay, worker_id, job["attempt_count"]
        )
    return True


def main():
    settings = ScoringSettings.from_env()
    store = ScoringStore(settings.database_path)
    store.initialize()
    scorer = create_scoring_client(settings)
    query_client = NotionClient(os.getenv("NOTION_TOKEN", ""))
    write_client = NotionClient(os.getenv("NOTION_TOKEN", ""), notion_version="2022-06-28")

    def persist(result):
        return persist_score_result(
            store,
            query_client,
            os.getenv("NOTION_DATA_SOURCE_ID", ""),
            os.getenv("NOTION_SCORE_DATABASE_ID", ""),
            os.getenv("NOTION_SCORE_DATA_SOURCE_ID", ""),
            result,
            write_client=write_client,
            use_lock=False,
        )

    worker_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    next_reconciliation = 0
    while True:
        store.record_worker_heartbeat(worker_id)
        now = time.monotonic()
        if now >= next_reconciliation:
            try:
                report = reconcile(
                    store,
                    query_client,
                    os.getenv("NOTION_DATA_SOURCE_ID", ""),
                    os.getenv("NOTION_SCORE_DATA_SOURCE_ID", ""),
                    settings.scorer_version,
                    settings.embedding_model,
                    settings.min_job_description_chars,
                    settings.publication_uncertainty_seconds,
                    settings.automatic_score_backfill,
                    settings.scoring_model_revision,
                )
                print(json.dumps({"event": "scoring_reconciliation", **report}, sort_keys=True))
            except Exception as error:
                print(json.dumps({"event": "scoring_reconciliation_failed", "error": str(error)[:500]}))
            next_reconciliation = now + max(60, settings.reconciliation_interval_seconds)
        worked = process_one_job(
            store,
            scorer,
            persist,
            worker_id,
            lease_seconds=settings.job_lease_seconds,
            max_attempts=settings.max_score_attempts,
            min_description_chars=settings.min_job_description_chars,
            cache_ttl_seconds=settings.score_cache_ttl_seconds,
        )
        if not worked:
            time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
