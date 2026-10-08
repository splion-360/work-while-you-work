import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from service.notion import NotionClient, NotionError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return io.BytesIO(json.dumps(self.payload).encode())

    def __exit__(self, *_args):
        return False


class NotionClientTests(unittest.TestCase):
    def test_queries_an_explicit_data_source(self):
        client = NotionClient("secret", notion_version="2026-03-11")
        with patch("urllib.request.urlopen", return_value=FakeResponse({"results": [{"id": "page-1"}]})) as urlopen:
            results = client.query_data_source("source-1", {"property": "Application key", "rich_text": {"equals": "abc"}})

        self.assertEqual(results, [{"id": "page-1"}])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.notion.com/v1/data_sources/source-1/query")
        self.assertEqual(json.loads(request.data), {
            "filter": {"property": "Application key", "rich_text": {"equals": "abc"}}
        })

    def test_paginates_unfiltered_data_source_queries(self):
        client = NotionClient("secret", notion_version="2026-03-11")
        with patch("urllib.request.urlopen", side_effect=[
            FakeResponse({"results": [{"id": "page-1"}], "has_more": True, "next_cursor": "cursor-1"}),
            FakeResponse({"results": [{"id": "page-2"}], "has_more": False}),
        ]) as urlopen:
            results = client.query_data_source("source-1")

        self.assertEqual([page["id"] for page in results], ["page-1", "page-2"])
        self.assertEqual(json.loads(urlopen.call_args_list[0].args[0].data), {})
        self.assertEqual(json.loads(urlopen.call_args_list[1].args[0].data), {"start_cursor": "cursor-1"})

    def test_creates_and_updates_pages(self):
        client = NotionClient("secret")
        with patch("urllib.request.urlopen", side_effect=[
            FakeResponse({"id": "created"}),
            FakeResponse({"id": "updated"}),
        ]) as urlopen:
            created = client.create_page("database-1", {"Company": {"title": []}})
            updated = client.update_page("page-1", {"Score": {"number": 80}})

        self.assertEqual(created["id"], "created")
        self.assertEqual(updated["id"], "updated")
        self.assertEqual(urlopen.call_args_list[0].args[0].get_method(), "POST")
        self.assertEqual(urlopen.call_args_list[1].args[0].get_method(), "PATCH")

    def test_updates_data_source_properties(self):
        client = NotionClient("secret")
        with patch("urllib.request.urlopen", return_value=FakeResponse({"id": "source-1"})) as urlopen:
            client.update_data_source("source-1", {"Job description": {"rich_text": {}}})

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.notion.com/v1/data_sources/source-1")
        self.assertEqual(request.get_method(), "PATCH")
        self.assertEqual(json.loads(request.data), {
            "properties": {"Job description": {"rich_text": {}}}
        })

    def test_trashes_pages_with_the_current_api_field(self):
        client = NotionClient("secret")
        with patch("urllib.request.urlopen", return_value=FakeResponse({"id": "trashed"})) as urlopen:
            client.trash_page("page-1")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "PATCH")
        self.assertEqual(json.loads(request.data), {"in_trash": True})

    def test_surfaces_notion_http_errors(self):
        error = urllib.error.HTTPError(
            "https://api.notion.com/v1/pages",
            429,
            "rate limited",
            {},
            io.BytesIO(b'{"message":"slow down"}'),
        )
        try:
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(NotionError, "Notion API returned 429"):
                    NotionClient("secret").create_page("database-1", {})
        finally:
            error.close()


if __name__ == "__main__":
    unittest.main()
