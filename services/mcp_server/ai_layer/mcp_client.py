from __future__ import annotations

import json
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ai_layer.settings import get_ai_settings


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())

    if hasattr(value, "dict"):
        return _json_safe(value.dict())

    return str(value)


def _content_to_jsonable(content: Any) -> Any:
    if isinstance(content, list):
        values: list[Any] = []

        for item in content:
            if hasattr(item, "text"):
                text = item.text

                try:
                    values.append(json.loads(text))
                except Exception:
                    values.append(text)
            else:
                values.append(_json_safe(item))

        if len(values) == 1:
            return values[0]

        return values

    return _json_safe(content)


class McpToolClient:
    """
    Thin MCP stdio client for the local PostgreSQL MCP server.

    It starts server.py as a subprocess, lists tools, and can call tools by name.
    """

    def __init__(self) -> None:
        self.settings = get_ai_settings()

        self.server_params = StdioServerParameters(
            command=self.settings.resolved_mcp_python(),
            args=[self.settings.resolved_mcp_script()],
            env=None,
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                response = await session.list_tools()

                tools: list[dict[str, Any]] = []

                for tool in response.tools:
                    tools.append(
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": _json_safe(tool.inputSchema),
                        }
                    )

                return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    name,
                    arguments=arguments or {},
                )

                if getattr(result, "isError", False):
                    raise RuntimeError(_content_to_jsonable(result.content))

                return _content_to_jsonable(result.content)
