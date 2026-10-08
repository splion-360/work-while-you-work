import json
import urllib.error
import urllib.request


class NotionError(RuntimeError):
    pass


class NotionClient:
    def __init__(self, token, notion_version="2026-03-11", timeout=15):
        self.token = token
        self.notion_version = notion_version
        self.timeout = timeout

    def _request(self, path, method="GET", payload=None):
        if not self.token:
            raise NotionError("NOTION_TOKEN is required")
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"https://api.notion.com/v1/{path.lstrip('/')}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Notion-Version": self.notion_version,
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise NotionError(f"Notion API returned {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise NotionError(f"Could not reach Notion API: {error.reason}") from error

    def query_data_source(self, data_source_id, filter_value=None):
        if not data_source_id:
            raise NotionError("Notion data source ID is required")
        results = []
        cursor = None
        while True:
            payload = {}
            if filter_value is not None:
                payload["filter"] = filter_value
            if cursor:
                payload["start_cursor"] = cursor
            response = self._request(
                f"data_sources/{data_source_id}/query",
                method="POST",
                payload=payload,
            )
            results.extend(response.get("results", []))
            if not response.get("has_more"):
                return results
            cursor = response.get("next_cursor")
            if not cursor:
                raise NotionError("Notion pagination ended without a next cursor")

    def create_page(self, database_id, properties):
        if not database_id:
            raise NotionError("Notion database ID is required")
        return self._request(
            "pages",
            method="POST",
            payload={"parent": {"database_id": database_id}, "properties": properties},
        )

    def update_data_source(self, data_source_id, properties):
        if not data_source_id:
            raise NotionError("Notion data source ID is required")
        return self._request(
            f"data_sources/{data_source_id}",
            method="PATCH",
            payload={"properties": properties},
        )

    def update_page(self, page_id, properties):
        if not page_id:
            raise NotionError("Notion page ID is required")
        return self._request(
            f"pages/{page_id}",
            method="PATCH",
            payload={"properties": properties},
        )

    def trash_page(self, page_id):
        if not page_id:
            raise NotionError("Notion page ID is required")
        return self._request(
            f"pages/{page_id}",
            method="PATCH",
            payload={"in_trash": True},
        )
