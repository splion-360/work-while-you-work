import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from service import server


class ResumeCatalogTests(unittest.TestCase):
    def test_catalog_returns_all_variants_newest_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / "one-column" / "jake" / "applied-ml" / "pdf" / "applied-ml-resume-v1.pdf",
                root / "one-column" / "professional" / "applied-ml" / "pdf" / "applied-ml-resume-v3.pdf",
                root / "two-column" / "sample-keywords" / "applied-ml" / "pdf" / "applied-ml-resume.pdf",
            ]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                os.utime(path, (index, index))
            (root / "one-column" / "jake" / "applied-ml" / "main.pdf").touch()
            with patch.object(server, "RESUMES_DIR", root):
                catalog = server.resume_catalog()
                self.assertEqual([item["version"] for item in catalog], [
                    "two-column/applied-ml-resume",
                    "professional/applied-ml-resume-v3",
                    "jake/applied-ml-resume-v1",
                ])
                self.assertEqual(catalog[0]["label"], "Two Column | applied-ml-resume")


class ApplicationKeyTests(unittest.TestCase):
    def test_key_normalizes_case_and_whitespace(self):
        first = server.application_key(" Example AI ", "ML Engineer", "LinkedIn")
        second = server.application_key("example ai", "  ML   Engineer ", "linkedin")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_preview_key_does_not_require_extracted_company_or_title(self):
        key = server.preview_application_key({
            "job_url": "https://example.com/jobs/1",
            "canonical_url": "https://example.com/jobs/1",
            "source_job_id": "1",
            "job_source": "Example",
        })

        self.assertEqual(len(key), 64)


class DashboardTests(unittest.TestCase):
    def test_counts_current_calendar_periods_and_ignores_future_dates(self):
        class FakeNotion:
            def __init__(self):
                self.filter_value = None

            def query_data_source(self, _source_id, filter_value):
                self.filter_value = filter_value
                return [
                    {"properties": {"Application date": {"date": {"start": value}}}}
                    for value in ("2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "invalid")
                ]

        notion = FakeNotion()
        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(server, "NOTION_DATA_SOURCE_ID", "source"):
            counts = server.dashboard_counts(server.date(2026, 9, 2), client=notion)

        self.assertEqual(counts, {"current_day": 1, "current_week": 3, "current_month": 2})
        self.assertEqual(notion.filter_value, {
            "and": [
                {"property": "Application date", "date": {"on_or_after": "2026-08-31"}},
                {"property": "Application date", "date": {"on_or_before": "2026-09-02"}},
            ]
        })

    def test_counts_applications_for_a_selected_month(self):
        class FakeNotion:
            def __init__(self):
                self.filter_value = None

            def query_data_source(self, _source_id, filter_value):
                self.filter_value = filter_value
                return [
                    {"properties": {"Application date": {"date": {"start": value}}}}
                    for value in ("2026-08-01", "2026-08-15", "2026-08-31", "2026-09-01")
                ]

        notion = FakeNotion()
        start, end = server.month_bounds("2026-08")
        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(server, "NOTION_DATA_SOURCE_ID", "source"):
            count = server.monthly_application_count(start, end, client=notion)

        self.assertEqual((start.isoformat(), end.isoformat()), ("2026-08-01", "2026-08-31"))
        self.assertEqual(count, 3)
        self.assertEqual(notion.filter_value, {
            "and": [
                {"property": "Application date", "date": {"on_or_after": "2026-08-01"}},
                {"property": "Application date", "date": {"on_or_before": "2026-08-31"}},
            ]
        })

    def test_summarizes_selected_month_statuses(self):
        class FakeNotion:
            def query_data_source(self, _source_id, _filter_value):
                return [
                    {
                        "properties": {
                            "Application date": {"date": {"start": application_date}},
                            "Status": {"status": {"name": status}},
                        }
                    }
                    for application_date, status in (
                        ("2026-09-01", "Applied"),
                        ("2026-09-02", "In Progress"),
                        ("2026-09-03", "Rejected"),
                        ("2026-09-04", "Cancelled"),
                        ("2026-09-05", "Applied"),
                        ("2026-10-01", "Applied"),
                    )
                ]

        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(server, "NOTION_DATA_SOURCE_ID", "source"):
            summary = server.monthly_application_summary(
                server.date(2026, 9, 1), server.date(2026, 9, 30), client=FakeNotion()
            )

        self.assertEqual(summary, {
            "total": 5,
            "statuses": {
                "applied": 2,
                "in_progress": 1,
                "rejected": 1,
                "cancelled": 1,
            },
        })

    def test_rejects_invalid_selected_month(self):
        with self.assertRaisesRegex(ValueError, "YYYY-MM"):
            server.month_bounds("2026-13")

    def test_counts_all_application_rows(self):
        notion = unittest.mock.Mock()
        notion.query_data_source.return_value = [{"id": "one"}, {"id": "two"}, {"id": "three"}]

        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(server, "NOTION_DATA_SOURCE_ID", "source"):
            count = server.total_application_count(client=notion)

        self.assertEqual(count, 3)
        notion.query_data_source.assert_called_once_with("source")


class ApplicationSearchTests(unittest.TestCase):
    @staticmethod
    def application_page(key="app-1", company="Gray Swan", title="Machine Learning Engineer"):
        return {
            "id": "application-page-1",
            "created_time": "2026-09-01T10:00:00Z",
            "last_edited_time": "2026-09-03T12:00:00Z",
            "properties": {
                "Company": {"title": [{"plain_text": company}]},
                "Job title": {"rich_text": [{"plain_text": title}]},
                "Job URL": {"url": "https://example.com/jobs/1"},
                "Job source": {"select": {"name": "LinkedIn"}},
                "Application date": {"date": {"start": "2026-09-02"}},
                "Resume version": {"select": {"name": "Professional | applied-ml-resume-v10"}},
                "Status": {"status": {"name": "In Progress"}},
                "Requested for sponsorship": {"checkbox": True},
                "Referral": {"checkbox": False},
                "Application key": {"rich_text": [{"plain_text": key}]},
                "Job description": {"rich_text": [{"plain_text": "Build production ML systems."}]},
            },
        }

    @staticmethod
    def score_page(key="app-1"):
        return {
            "id": "score-page-1",
            "created_time": "2026-09-02T10:00:00Z",
            "last_edited_time": "2026-09-02T11:00:00Z",
            "properties": {
                "Application key": {"title": [{"plain_text": key}]},
                "Score": {"number": 78.5},
                "Band": {"select": {"name": "Strong"}},
                "Matched terms": {"rich_text": [{"plain_text": '["Python","PyTorch"]'}]},
                "Category breakdown": {"rich_text": [{"plain_text": '{"core":80,"required":77}'}]},
            },
        }

    def test_lists_lightweight_application_summaries_newest_first(self):
        notion = unittest.mock.Mock()
        older = self.application_page("app-older", "Older Co", "Data Scientist")
        older["properties"]["Application date"]["date"]["start"] = "2026-08-31"
        notion.query_data_source.return_value = [older, self.application_page()]

        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(
            server, "NOTION_DATA_SOURCE_ID", "applications-source"
        ):
            summaries = server.application_search_catalog(client=notion)

        self.assertEqual([item["application_key"] for item in summaries], ["app-1", "app-older"])
        self.assertEqual(summaries[0], {
            "application_key": "app-1",
            "company": "Gray Swan",
            "job_title": "Machine Learning Engineer",
            "status": "In Progress",
            "application_date": "2026-09-02",
            "job_source": "LinkedIn",
        })
        notion.query_data_source.assert_called_once_with("applications-source")

    def test_joins_application_with_its_persisted_score(self):
        class FakeNotion:
            def query_data_source(self, source_id, filter_value):
                if source_id == "applications-source":
                    self.application_filter = filter_value
                    return [ApplicationSearchTests.application_page()]
                self.score_filter = filter_value
                return [ApplicationSearchTests.score_page()]

        notion = FakeNotion()
        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(
            server, "NOTION_DATA_SOURCE_ID", "applications-source"
        ), patch.object(server, "NOTION_SCORE_DATA_SOURCE_ID", "scores-source"):
            result = server.application_insights("app-1", client=notion)

        self.assertEqual(result["application"]["resume_version"], "Professional | applied-ml-resume-v10")
        self.assertTrue(result["application"]["requested_for_sponsorship"])
        self.assertEqual(result["logged_score"], {
            "score": 78.5,
            "band": "Strong",
            "matched_terms": ["Python", "PyTorch"],
            "category_breakdown": {"core": 80, "required": 77},
            "created_at": "2026-09-02T10:00:00Z",
            "updated_at": "2026-09-02T11:00:00Z",
        })
        self.assertEqual(notion.application_filter, {
            "property": "Application key", "rich_text": {"equals": "app-1"}
        })
        self.assertEqual(notion.score_filter, {
            "property": "Application key", "title": {"equals": "app-1"}
        })

    def test_application_check_response_includes_existing_status(self):
        response = server.application_check_response([self.application_page()], "app-1")

        self.assertEqual(response, {
            "exists": True,
            "application_key": "app-1",
            "status": "In Progress",
        })

    def test_application_check_response_omits_status_when_not_found(self):
        response = server.application_check_response([], "app-1")

        self.assertEqual(response, {"exists": False, "application_key": "app-1"})

    def test_returns_application_when_score_is_not_available(self):
        notion = unittest.mock.Mock()
        notion.query_data_source.side_effect = [[self.application_page()], []]

        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(
            server, "NOTION_DATA_SOURCE_ID", "applications-source"
        ), patch.object(server, "NOTION_SCORE_DATA_SOURCE_ID", "scores-source"):
            result = server.application_insights("app-1", client=notion)

        self.assertIsNone(result["logged_score"])

    def test_updates_only_the_status_for_one_application(self):
        notion = unittest.mock.Mock()
        notion.query_data_source.return_value = [self.application_page()]
        notion.update_page.return_value = {"last_edited_time": "2026-09-07T15:00:00Z"}

        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(
            server, "NOTION_DATA_SOURCE_ID", "applications-source"
        ):
            result = server.update_application_status(
                "app-1", "Rejected", client=notion, write_client=notion
            )

        self.assertEqual(result, {
            "application_key": "app-1",
            "status": "Rejected",
            "updated_at": "2026-09-07T15:00:00Z",
        })
        notion.update_page.assert_called_once_with(
            "application-page-1", {"Status": {"status": {"name": "Rejected"}}}
        )

    def test_rejects_unknown_status_without_writing(self):
        notion = unittest.mock.Mock()

        with self.assertRaisesRegex(ValueError, "Invalid application status"):
            server.update_application_status(
                "app-1", "Interviewing", client=notion, write_client=notion
            )

        notion.query_data_source.assert_not_called()
        notion.update_page.assert_not_called()

    def test_updates_application_details_and_migrates_identity(self):
        class FakeNotion:
            def __init__(self):
                self.application_updates = []
                self.score_updates = []

            def query_data_source(self, source_id, filter_value):
                if source_id == "applications-source":
                    equals = filter_value["rich_text"]["equals"]
                    if equals == "app-1":
                        return [ApplicationSearchTests.application_page()]
                    return []
                self.score_filter = filter_value
                return [ApplicationSearchTests.score_page()]

            def update_page(self, page_id, properties):
                if page_id.startswith("score"):
                    self.score_updates.append((page_id, properties))
                else:
                    self.application_updates.append((page_id, properties))
                return {"last_edited_time": "2026-09-08T15:00:00Z"}

        notion = FakeNotion()
        store = unittest.mock.Mock()
        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(
            server, "NOTION_DATA_SOURCE_ID", "applications-source"
        ), patch.object(server, "NOTION_SCORE_DATA_SOURCE_ID", "scores-source"):
            result = server.update_application_details(
                "app-1",
                {
                    "company": "Example AI",
                    "job_title": "ML Engineer",
                    "job_source": "Greenhouse",
                    "job_url": "https://example.com/jobs/2",
                    "application_date": "2026-09-08",
                    "status": "In Progress",
                    "requested_for_sponsorship": False,
                    "referral": True,
                    "job_description": "Build reliable systems.",
                },
                client=notion,
                write_client=notion,
                store=store,
            )

        new_key = server.application_key("Example AI", "ML Engineer", "Greenhouse")
        self.assertEqual(result["application"]["application_key"], new_key)
        self.assertEqual(result["application"]["job_source"], "Greenhouse")
        self.assertEqual(result["application"]["referral"], True)
        self.assertEqual(notion.application_updates[0][0], "application-page-1")
        self.assertEqual(
            notion.application_updates[0][1]["Application key"],
            {"rich_text": [{"text": {"content": new_key}}]},
        )
        self.assertEqual(
            notion.score_updates[0],
            ("score-page-1", {"Application key": {"title": [{"text": {"content": new_key}}]}}),
        )
        store.rename_application.assert_called_once_with("app-1", new_key)

    def test_update_application_details_rejects_duplicate_identity(self):
        class FakeNotion:
            def query_data_source(self, _source_id, filter_value):
                equals = filter_value["rich_text"]["equals"]
                if equals == "app-1":
                    return [ApplicationSearchTests.application_page()]
                return [{"id": "other-page", "properties": {}}]

        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(
            server, "NOTION_DATA_SOURCE_ID", "applications-source"
        ):
            with self.assertRaisesRegex(ValueError, "matches another application"):
                server.update_application_details(
                    "app-1",
                    {"company": "Other", "job_title": "Role", "job_source": "LinkedIn"},
                    client=FakeNotion(),
                    write_client=unittest.mock.Mock(),
                    store=unittest.mock.Mock(),
                )

    def test_deletes_application_score_and_local_scoring_state(self):
        notion = unittest.mock.Mock()
        notion.query_data_source.side_effect = [
            [self.application_page()],
            [self.score_page()],
        ]
        store = unittest.mock.Mock()
        store.delete_application.return_value = True

        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(
            server, "NOTION_DATA_SOURCE_ID", "applications-source"
        ), patch.object(server, "NOTION_SCORE_DATA_SOURCE_ID", "scores-source"):
            result = server.delete_application(
                "app-1", client=notion, write_client=notion, store=store
            )

        self.assertEqual(result, {
            "application_key": "app-1",
            "scores_deleted": 1,
            "local_scoring_deleted": True,
        })
        self.assertEqual(
            [call.args[0] for call in notion.trash_page.call_args_list],
            ["score-page-1", "application-page-1"],
        )
        store.delete_application.assert_called_once_with("app-1")

    def test_delete_returns_none_when_application_does_not_exist(self):
        notion = unittest.mock.Mock()
        notion.query_data_source.return_value = []
        store = unittest.mock.Mock()

        with patch.object(server, "NOTION_TOKEN", "token"), patch.object(
            server, "NOTION_DATA_SOURCE_ID", "applications-source"
        ):
            result = server.delete_application("missing", client=notion, store=store)

        self.assertIsNone(result)
        notion.trash_page.assert_not_called()
        store.delete_application.assert_not_called()


class ScoringInputTests(unittest.TestCase):
    def test_notion_rich_text_chunks_long_descriptions(self):
        chunks = server.notion_rich_text("x" * 4500)

        self.assertEqual([len(item["text"]["content"]) for item in chunks], [2000, 2000, 500])
        self.assertEqual(server.notion_rich_text("  "), [])

    def test_notion_rich_text_respects_utf16_limit(self):
        text = ("x" * 1999) + "\U0001f680" + "tail"

        chunks = server.notion_rich_text(text)

        self.assertEqual("".join(item["text"]["content"] for item in chunks), text)
        self.assertTrue(all(
            len(item["text"]["content"].encode("utf-16-le")) // 2 <= 2000
            for item in chunks
        ))

    def test_prepares_hashes_for_the_exact_resume_and_description(self):
        with tempfile.TemporaryDirectory() as directory:
            resume = Path(directory) / "resume.pdf"
            resume.write_bytes(b"resume snapshot")
            application = {
                "application_key": "app-1",
                "canonical_url": "https://example.com/jobs/1",
                "source_job_id": "job-1",
                "job_description": "  Build ML systems.  ",
                "description_provenance": "original_saved",
                "extraction_quality": "complete",
                "resume_version": "professional/resume-v1",
            }

            captured = server.prepare_scoring_input(application, {"path": str(resume)})

        self.assertEqual(captured["job_description"], "Build ML systems.")
        self.assertEqual(captured["job_description_hash"], "540e3b8bb379472806b9da20a12e77c8ea40a56fab805980a1a2997e7a123ba9")
        self.assertEqual(captured["resume_content_hash"], "39a68a4be393922b54104ff4145e9aadb37be92c58d61c562aba620261d43fa6")
        self.assertEqual(captured["description_provenance"], "original_saved")

    def test_preserves_job_description_section_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            resume = Path(directory) / "resume.pdf"
            resume.write_bytes(b"resume snapshot")
            application = {
                "application_key": "app-1",
                "job_url": "https://example.com/jobs/1",
                "job_description": "About us\n\nWhat You'll Do\n  Build ML systems.  ",
                "extraction_quality": "complete",
                "resume_version": "professional/resume-v1",
            }

            captured = server.prepare_scoring_input(application, {"path": str(resume)})

        self.assertEqual(
            captured["job_description"],
            "About us\n\nWhat You'll Do\nBuild ML systems.",
        )

    def test_scoring_description_drops_compensation_and_disclaimer_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            resume = Path(directory) / "resume.pdf"
            resume.write_bytes(b"resume snapshot")
            application = {
                "application_key": "app-1",
                "job_url": "https://example.com/jobs/1",
                "job_description": (
                    "Responsibilities\nBuild ML systems.\n\n"
                    "Compensation\n$120k to $180k.\n\n"
                    "Disclaimer\nThis posting may change."
                ),
                "extraction_quality": "complete",
                "resume_version": "professional/resume-v1",
            }

            captured = server.prepare_scoring_input(application, {"path": str(resume)})

        self.assertEqual(captured["job_description"], "Responsibilities\nBuild ML systems.")

    def test_notion_logging_remains_successful_when_queue_activation_fails(self):
        with patch.object(server.ScoringSettings, "from_env") as settings:
            settings.return_value.database_path = Path("/unavailable/state.sqlite3")
            with patch.object(server.ScoringStore, "activate_score_job", side_effect=OSError("disk unavailable")):
                status = server.activate_captured_score("fingerprint-1")

        self.assertEqual(status, "pending_reconciliation")

    def test_queue_status_is_pending_when_no_staged_job_is_activated(self):
        with patch.object(server.ScoringSettings, "from_env") as settings:
            settings.return_value.database_path = Path("/tmp/state.sqlite3")
            with patch.object(server.ScoringStore, "activate_score_job", return_value=False):
                status = server.activate_captured_score("missing-fingerprint")

        self.assertEqual(status, "pending_reconciliation")

    def test_short_descriptions_are_not_queued(self):
        with patch.object(server.ScoringSettings, "from_env") as settings:
            settings.return_value.min_job_description_chars = 200
            status, fingerprint = server.capture_scoring_input(
                {"job_description": "Too short", "extraction_quality": "complete"},
                {"path": "/unused.pdf"},
            )

        self.assertEqual(status, "unavailable")
        self.assertIsNone(fingerprint)

    def test_log_without_score_does_not_capture_or_queue_scoring(self):
        application = {"score_application": False}
        with patch.object(server, "capture_scoring_input") as capture:
            status, fingerprint = server.capture_requested_score(
                application, {"path": "/unused.pdf"}
            )

        self.assertEqual(status, "not_requested")
        self.assertIsNone(fingerprint)
        capture.assert_not_called()

    def test_log_with_score_preserves_existing_capture_behavior(self):
        application = {"score_application": True}
        with patch.object(
            server, "capture_scoring_input", return_value=("captured", "fingerprint-1")
        ) as capture:
            result = server.capture_requested_score(application, {"path": "/resume.pdf"})

        self.assertEqual(result, ("captured", "fingerprint-1"))
        capture.assert_called_once()

    def test_rejects_non_boolean_score_choice(self):
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            server.capture_requested_score(
                {"score_application": "false"}, {"path": "/unused.pdf"}
            )


class OnDemandScoreTests(unittest.TestCase):
    def test_scores_preview_without_persisting_an_application_or_queue_job(self):
        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "resume.pdf"
            resume_path.write_bytes(b"resume")
            application = {
                "application_key": "app-1",
                "job_url": "https://example.com/jobs/1",
                "job_description": "Build production machine learning systems. " * 8,
                "extraction_quality": "complete",
                "resume_version": "professional/resume-v1",
            }
            settings = SimpleNamespace(
                database_path=Path(directory) / "state.sqlite3",
                min_job_description_chars=200,
                max_job_description_chars=50000,
                embedding_model="embedding-v1",
                scorer_version="v1",
                scoring_engine_url="http://scoring-engine:8770",
                scoring_engine_timeout_seconds=180,
                scoring_model_revision="",
            )
            store = unittest.mock.Mock()
            expected = {
                "score": 81.0,
                "band": "Strong",
                "matched_terms": ["Python"],
                "category_breakdown": {"requirements": 80.0},
            }
            with patch.object(server, "create_scoring_client") as scorer:
                store.get_cached_score.return_value = None
                scorer.return_value.score.return_value = expected
                result, source = server.score_application_now(
                    application, {"path": str(resume_path)}, store=store, settings=settings
                )

        self.assertEqual(result, expected)
        self.assertEqual(source, "computed")
        store.initialize.assert_called_once_with()
        self.assertFalse(store.save_scoring_input.called)
        self.assertFalse(store.enqueue_score_job.called)

    def test_reuses_only_an_exact_existing_score(self):
        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "resume.pdf"
            resume_path.write_bytes(b"resume")
            application = {
                "application_key": "app-1",
                "job_url": "https://example.com/jobs/1",
                "job_description": "Build production machine learning systems. " * 8,
                "extraction_quality": "complete",
                "resume_version": "professional/resume-v1",
            }
            settings = SimpleNamespace(
                database_path=Path(directory) / "state.sqlite3",
                min_job_description_chars=200,
                max_job_description_chars=50000,
                embedding_model="embedding-v1",
                scorer_version="v1",
                scoring_engine_url="http://scoring-engine:8770",
                scoring_engine_timeout_seconds=180,
                scoring_model_revision="",
            )
            cached = {"score": 72.0, "band": "Moderate"}
            store = unittest.mock.Mock()
            store.get_cached_score.return_value = cached
            with patch.object(server, "create_scoring_client") as scorer:
                result, source = server.score_application_now(
                    application, {"path": str(resume_path)}, store=store, settings=settings
                )

        self.assertEqual(result["score"], 72.0)
        self.assertEqual(result["application_key"], "app-1")
        self.assertEqual(source, "cached")
        scorer.assert_not_called()

    def test_fetches_exact_notion_score_and_warms_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "resume.pdf"
            resume_path.write_bytes(b"resume")
            application = {
                "application_key": "app-1",
                "job_url": "https://example.com/jobs/1",
                "job_description": "Build production machine learning systems. " * 8,
                "extraction_quality": "complete",
                "resume_version": "professional/resume-v1",
            }
            settings = SimpleNamespace(
                database_path=Path(directory) / "state.sqlite3",
                min_job_description_chars=200,
                max_job_description_chars=50000,
                embedding_model="embedding-v1",
                scorer_version="v1",
                scoring_model_revision="revision-1",
                score_cache_ttl_seconds=300,
            )
            store = unittest.mock.Mock()
            store.get_cached_score.return_value = None
            notion = unittest.mock.Mock()
            notion.query_data_source.return_value = [{
                "properties": {
                    "Score": {"number": 79.0},
                    "Band": {"select": {"name": "Strong"}},
                    "Matched terms": {"rich_text": []},
                    "Category breakdown": {"rich_text": []},
                    "Fit label": {"rich_text": [{"plain_text": "Good Fit"}]},
                    "Fit probability": {"number": 0.79},
                    "Decision threshold": {"number": 0.55},
                }
            }]
            with patch.object(server, "NOTION_SCORE_DATA_SOURCE_ID", "scores-source"), patch.object(
                server, "create_scoring_client"
            ) as scorer:
                result, source = server.score_application_now(
                    application,
                    {"path": str(resume_path)},
                    store=store,
                    settings=settings,
                    notion=notion,
                )

        self.assertEqual(source, "fetched")
        self.assertEqual(result["fit_probability"], 0.79)
        store.save_cached_score.assert_called_once()
        scorer.assert_not_called()

    def test_force_recompute_bypasses_sqlite_and_notion_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "resume.pdf"
            resume_path.write_bytes(b"resume")
            application = {
                "application_key": "app-1",
                "job_url": "https://example.com/jobs/1",
                "job_description": "Build production machine learning systems. " * 8,
                "extraction_quality": "complete",
                "resume_version": "professional/resume-v1",
            }
            settings = SimpleNamespace(
                database_path=Path(directory) / "state.sqlite3",
                min_job_description_chars=200,
                max_job_description_chars=50000,
                embedding_model="embedding-v1",
                scorer_version="v1",
                scoring_model_revision="revision-1",
                score_cache_ttl_seconds=300,
            )
            expected = {
                "score": 84.0,
                "band": "Strong",
                "matched_terms": ["Python"],
                "category_breakdown": {"requirements": 84.0},
            }
            store = unittest.mock.Mock()
            store.get_cached_score.return_value = {"score": 20.0, "band": "Weak"}
            notion = unittest.mock.Mock()
            with patch.object(server, "create_scoring_client") as scorer:
                scorer.return_value.score.return_value = expected
                result, source = server.score_application_now(
                    application,
                    {"path": str(resume_path)},
                    store=store,
                    settings=settings,
                    notion=notion,
                    force_recompute=True,
                )

        self.assertEqual(result, expected)
        self.assertEqual(source, "computed")
        store.get_cached_score.assert_not_called()
        notion.query_data_source.assert_not_called()
        scorer.return_value.score.assert_called_once()
        store.save_cached_score.assert_called_once()


if __name__ == "__main__":
    unittest.main()
