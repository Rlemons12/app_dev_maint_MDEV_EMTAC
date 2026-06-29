from __future__ import annotations

import asyncio
import os
from typing import Any

from listed_server.mcp_coordinator.settings import DownstreamServerConfig, get_settings


class DownstreamMCPClient:
    """
    Executes routed coordinator requests against downstream capability providers.

    Current behavior:
    - postgres: direct local Python call into listed_server.mcp_postgres.tools.postgres_tools
    - grafana: stdio MCP client call using configured downstream MCP command
    - other capabilities: not implemented yet

    Important Windows note:
    MCP stdio imports can require pywin32/win32api. Those imports are intentionally
    lazy now, so Postgres execution does not require win32api.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def execute(
        self,
        target_server: str,
        target_tool: str | None,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        target_server = (target_server or "").strip().lower()

        if target_server == "postgres":
            return self._execute_postgres(target_tool, arguments)

        if target_server == "grafana":
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(
                    self.execute_async(
                        target_server=target_server,
                        target_tool=target_tool,
                        arguments=arguments,
                    )
                )

            return {
                "status": "error",
                "message": (
                    "Synchronous execute() cannot run Grafana while an event loop is already active. "
                    "Use execute_async() instead."
                ),
                "target_server": target_server,
                "target_tool": target_tool,
                "arguments": arguments,
            }

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
        target_server = (target_server or "").strip().lower()

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

        try:
            from listed_server.mcp_postgres.tools import postgres_tools
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Could not import postgres_tools: {exc}",
                "target_server": "postgres",
                "target_tool": target_tool,
                "arguments": arguments,
            }

        tool_fn = getattr(postgres_tools, target_tool, None)

        if tool_fn is None:
            return {
                "status": "not_implemented",
                "message": "Postgres target tool is not available.",
                "target_server": "postgres",
                "target_tool": target_tool,
                "arguments": arguments,
            }

        try:
            result = tool_fn(**arguments)
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Postgres tool execution failed: {exc}",
                "target_server": "postgres",
                "target_tool": target_tool,
                "arguments": arguments,
            }

        return {
            "status": "ok",
            "target_server": "postgres",
            "target_tool": target_tool,
            "result": result,
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
                "target_tool": target_tool,
                "arguments": arguments,
            }

        if not config.command:
            return {
                "status": "error",
                "message": f"{target_server} MCP command is not configured.",
                "target_server": target_server,
                "target_tool": target_tool,
                "arguments": arguments,
            }

        if not target_tool:
            return {
                "status": "tool_required",
                "message": (
                    "No target tool was selected. "
                    "Call the downstream MCP server directly or route a more specific request."
                ),
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
        """
        Calls a downstream MCP server over stdio.

        MCP stdio imports are intentionally inside this function so Postgres
        execution does not require win32api/pywin32.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ModuleNotFoundError as exc:
            return {
                "status": "error",
                "message": (
                    "Could not import MCP stdio dependencies. "
                    "On Windows this usually means pywin32/win32api is unavailable "
                    "or PATH/PYTHONPATH is not configured for pywin32."
                ),
                "missing_module": exc.name,
                "target_server": target_server,
                "target_tool": target_tool,
                "arguments": arguments,
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Could not import MCP stdio client: {exc}",
                "target_server": target_server,
                "target_tool": target_tool,
                "arguments": arguments,
            }

        env = os.environ.copy()
        env.update(getattr(config, "env", {}) or {})

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=env,
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        target_tool,
                        arguments=arguments,
                    )

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

        except Exception as exc:
            return {
                "status": "error",
                "message": f"Downstream stdio MCP call failed: {exc}",
                "target_server": target_server,
                "target_tool": target_tool,
                "arguments": arguments,
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