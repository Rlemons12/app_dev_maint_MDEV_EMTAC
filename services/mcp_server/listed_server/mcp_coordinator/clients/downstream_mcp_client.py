from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from listed_server.mcp_coordinator.settings import DownstreamServerConfig, get_settings


class DownstreamMCPClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def execute(
        self,
        target_server: str,
        target_tool: str | None,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if target_server == "postgres":
            return self._execute_postgres(target_tool, arguments)

        if target_server == "grafana":
            return asyncio.run(
                self.execute_async(
                    target_server=target_server,
                    target_tool=target_tool,
                    arguments=arguments,
                )
            )

        return {
            "status": "not_implemented",
            "message": "Downstream MCP execution is not wired yet.",
            "target_server": target_server,
            "target_tool": target_tool,
            "arguments": arguments,
        }

    async def execute_async(
        self,
        target_server: str,
        target_tool: str | None,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if target_server == "postgres":
            return self._execute_postgres(target_tool, arguments)

        if target_server == "grafana":
            return await self._execute_configured_downstream(
                target_server=target_server,
                config=self.settings.grafana_mcp,
                target_tool=target_tool,
                arguments=arguments,
            )

        return {
            "status": "not_implemented",
            "message": "Downstream MCP execution is not wired yet.",
            "target_server": target_server,
            "target_tool": target_tool,
            "arguments": arguments,
        }

    @staticmethod
    def _execute_postgres(
        target_tool: str | None,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not target_tool:
            return {
                "status": "error",
                "message": "No postgres target tool was selected.",
                "target_server": "postgres",
                "arguments": arguments,
            }

        from listed_server.mcp_postgres.tools import postgres_tools

        tool_fn = getattr(postgres_tools, target_tool, None)
        if tool_fn is None:
            return {
                "status": "not_implemented",
                "message": "Postgres target tool is not available.",
                "target_server": "postgres",
                "target_tool": target_tool,
                "arguments": arguments,
            }

        return {
            "status": "ok",
            "target_server": "postgres",
            "target_tool": target_tool,
            "result": tool_fn(**arguments),
        }

    async def _execute_configured_downstream(
        self,
        target_server: str,
        config: DownstreamServerConfig,
        target_tool: str | None,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not config.enabled:
            return {
                "status": "disabled",
                "message": f"{target_server} MCP is not enabled.",
                "target_server": target_server,
            }

        if not config.command:
            return {
                "status": "error",
                "message": f"{target_server} MCP command is not configured.",
                "target_server": target_server,
            }

        if not target_tool:
            return {
                "status": "tool_required",
                "message": "No target tool was selected. Call the Grafana MCP server directly or route a more specific request.",
                "target_server": target_server,
                "arguments": arguments,
            }

        return await self._call_stdio_tool(
            target_server=target_server,
            config=config,
            target_tool=target_tool,
            arguments=arguments,
        )

    @staticmethod
    async def _call_stdio_tool(
        target_server: str,
        config: DownstreamServerConfig,
        target_tool: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        env = os.environ.copy()
        env.update(config.env)

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(target_tool, arguments=arguments)

                if getattr(result, "isError", False):
                    return {
                        "status": "error",
                        "target_server": target_server,
                        "target_tool": target_tool,
                        "result": _content_to_jsonable(result.content),
                    }

                return {
                    "status": "ok",
                    "target_server": target_server,
                    "target_tool": target_tool,
                    "result": _content_to_jsonable(result.content),
                }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

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
                values.append(item.text)
            else:
                values.append(_json_safe(item))

        if len(values) == 1:
            return values[0]

        return values

    return _json_safe(content)
