from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8765"
BASE_URL_ENV_VAR = "POSTGRES_MCP_UI_BASE_URL"


class PostgresMcpUiClientError(RuntimeError):
    """Raised when the PostgreSQL MCP UI API request fails."""


class PostgresMcpUiClient:
    """
    Small HTTP client for the local PostgreSQL MCP web UI.

    The base URL defaults to http://127.0.0.1:8765 and can be overridden with
    POSTGRES_MCP_UI_BASE_URL.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        configured_base_url = base_url or os.getenv(BASE_URL_ENV_VAR) or DEFAULT_BASE_URL

        self.base_url = configured_base_url.rstrip("/") + "/"
        self.timeout = timeout

    def settings(self) -> dict[str, Any]:
        return self._request("GET", "api/settings")

    def tools(self) -> list[dict[str, Any]]:
        response = self._request("GET", "api/tools")
        tools = response.get("tools")

        if not isinstance(tools, list):
            raise PostgresMcpUiClientError("Unexpected /api/tools response.")

        return tools

    def ask(self, question: str, allow_destructive: bool = False) -> str:
        response = self._request(
            "POST",
            "api/ask",
            {
                "question": question,
                "allow_destructive": allow_destructive,
            },
        )
        answer = response.get("answer")

        if not isinstance(answer, str):
            raise PostgresMcpUiClientError("Unexpected /api/ask response.")

        return answer

    def read_query(self, sql: str, database_name: str | None = None) -> Any:
        payload: dict[str, Any] = {"sql": sql}

        if database_name:
            payload["database_name"] = database_name

        response = self._request("POST", "api/read-query", payload)
        return response.get("result")

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        response = self._request(
            "POST",
            "api/tool",
            {
                "name": name,
                "arguments": arguments or {},
            },
        )
        return response.get("result")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: bytes | None = None
        headers = {"Accept": "application/json"}

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            urljoin(self.base_url, path),
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return self._decode_json_response(response.read())
        except HTTPError as exc:
            detail = self._read_error_detail(exc)
            raise PostgresMcpUiClientError(
                f"PostgreSQL MCP UI returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise PostgresMcpUiClientError(
                f"Could not connect to PostgreSQL MCP UI at {self.base_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise PostgresMcpUiClientError(
                f"Timed out calling PostgreSQL MCP UI at {self.base_url}."
            ) from exc

    def _decode_json_response(self, raw_body: bytes) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise PostgresMcpUiClientError(
                "PostgreSQL MCP UI returned invalid JSON."
            ) from exc

        if not isinstance(parsed, dict):
            raise PostgresMcpUiClientError(
                "PostgreSQL MCP UI returned an unexpected JSON value."
            )

        if "error" in parsed:
            raise PostgresMcpUiClientError(str(parsed["error"]))

        return parsed

    def _read_error_detail(self, exc: HTTPError) -> str:
        try:
            parsed = self._decode_json_response(exc.read())
        except PostgresMcpUiClientError:
            return exc.reason

        return str(parsed.get("error") or parsed)
