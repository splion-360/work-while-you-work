#!/usr/bin/env python3
import json
import os
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime, timezone

try:
    from service.notion import NotionClient
except ModuleNotFoundError:
    from notion import NotionClient


def request_json(url, payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def main():
    service_url = os.getenv("JOB_TRACKER_URL", "http://127.0.0.1:8765").rstrip("/")
    api_key = os.getenv("JOB_TRACKER_API_KEY", "")
    headers = {"X-Job-Tracker-Key": api_key} if api_key else {}
    health = request_json(f"{service_url}/healthz")
    if not health.get("ok"):
        raise RuntimeError("Job tracker is not ready")
    resumes = request_json(f"{service_url}/api/resumes", headers=headers)["resumes"]
    if not resumes:
        raise RuntimeError("No resume PDFs are available")
    suffix = uuid.uuid4().hex[:10]
    payload = {
        "company": f"Job Tracker Smoke {suffix}",
        "job_title": "Applied ML Engineer E2E",
        "job_url": f"https://www.linkedin.com/jobs/view/{suffix}",
        "job_source": "LinkedIn",
        "resume_version": resumes[0]["version"],
        "application_date": datetime.now(timezone.utc).date().isoformat(),
        "status": "Applied",
        "requested_for_sponsorship": False,
        "referral": False,
        "job_description": (
            "Build and deploy production machine learning systems using Python, PyTorch, "
            "FastAPI, Snowflake, Docker, Kubernetes, AWS, model evaluation, monitoring, "
            "computer vision, and recommendation systems across reliable data pipelines."
        ),
        "description_provenance": "original_saved",
        "extraction_quality": "complete",
    }
    notion = NotionClient(os.environ.get("NOTION_TOKEN", ""))
    application_page_id = None
    application_key = None
    score_pages = []
    started = time.monotonic()
    try:
        preview = request_json(f"{service_url}/api/applications/score", payload, headers)
        if preview.get("state") != "completed" or not preview.get("preview"):
            raise RuntimeError("On-demand preview score was not returned")
        preview_key = preview["application_key"]
        preview_application_pages = notion.query_data_source(
            os.environ.get("NOTION_DATA_SOURCE_ID", ""),
            {"property": "Application key", "rich_text": {"equals": preview_key}},
        )
        if preview_application_pages:
            raise RuntimeError("Preview scoring created a Notion application")
        logged = request_json(f"{service_url}/api/applications", payload, headers)
        elapsed = round(time.monotonic() - started, 3)
        application_page_id = logged["notion_page_id"]
        application_key = logged["application_key"]
        if application_key != preview_key:
            raise RuntimeError("Preview and logged application keys differ")
        application_pages = notion.query_data_source(
            os.environ.get("NOTION_DATA_SOURCE_ID", ""),
            {"property": "Application key", "rich_text": {"equals": application_key}},
        )
        if len(application_pages) != 1:
            raise RuntimeError(f"Expected one application row, found {len(application_pages)}")
        description_items = application_pages[0]["properties"]["Job description"]["rich_text"]
        saved_description = "".join(item.get("plain_text", "") for item in description_items)
        if saved_description != payload["job_description"]:
            raise RuntimeError("Notion did not preserve the captured job description")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            score_pages = notion.query_data_source(
                os.environ.get("NOTION_SCORE_DATA_SOURCE_ID", ""),
                {"property": "Application key", "title": {"equals": application_key}},
            )
            if score_pages:
                break
            time.sleep(2)
        if len(score_pages) != 1:
            raise RuntimeError(f"Expected one score row, found {len(score_pages)}")
        checked_score = request_json(
            f"{service_url}/api/applications/score", payload, headers
        )
        if checked_score.get("state") != "completed":
            raise RuntimeError(f"Score lookup returned {checked_score.get('state')}")
        print(json.dumps({
            "application_logged_seconds": elapsed,
            "scoring_status": logged.get("scoring_status"),
            "score_rows": len(score_pages),
            "score_lookup": checked_score["score"],
            "preview_score": preview["score"],
            "scoring_backend": health["scoring_backend"],
            "job_description_saved": True,
        }, sort_keys=True))
    finally:
        for page_id in [*(page["id"] for page in score_pages), application_page_id]:
            if page_id:
                notion.trash_page(page_id)
        if application_key:
            subprocess.run(
                [
                    "docker", "compose", "exec", "-T", "scoring-worker", "python", "-c",
                    (
                        "import sys; from service.config import ScoringSettings; "
                        "from service.scoring_state import ScoringStore; "
                        "s=ScoringStore(ScoringSettings.from_env().database_path); c=s.connect(); "
                        "c.execute('DELETE FROM scoring_inputs WHERE application_key = ?', (sys.argv[1],)); "
                        "c.commit(); c.close()"
                    ),
                    application_key,
                ],
                check=True,
                timeout=15,
            )


if __name__ == "__main__":
    main()
