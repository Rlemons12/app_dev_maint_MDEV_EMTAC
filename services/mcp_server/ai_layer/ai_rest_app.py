from __future__ import annotations

"""
EMTAC MCP Server AI REST Layer

Path:
    E:\\emtac\\services\\mcp_server\\ai_layer\\ai_rest_app.py

Purpose:
    This service owns the AI-to-MCP interface for EMTAC.

Flow:
    Service Dashboard chat box
        -> POST http://127.0.0.1:9200/api/ai/chat
        -> ai_rest_app.py
        -> AI Gateway / local model at :9000
        -> tool calls executed through MCP-facing logic
        -> final assistant response
        -> dashboard displays result

Notes:
    This is the first migration step. It intentionally keeps the dashboard-style
    tool loop here so the Service Dashboard can stop owning the AI orchestration.

    Over time, _execute_tool should be refactored so every tool call goes through
    listed_server/mcp_coordinator and the listed MCP servers.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests as http_requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------
# Bootstrap paths
# ---------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
AI_LAYER_DIR = CURRENT_FILE.parent
MCP_SERVER_ROOT = AI_LAYER_DIR.parent
SERVICES_DIR = MCP_SERVER_ROOT.parent
EMTAC_ROOT = SERVICES_DIR.parent

for path in [
    str(MCP_SERVER_ROOT),
    str(SERVICES_DIR),
    str(EMTAC_ROOT),
]:
    if path not in sys.path:
        sys.path.insert(0, path)

from ai_layer.dashboard_tool_definitions import TOOLS, get_tool_names  # noqa: E402


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

DEFAULT_ENV_PATH = Path(r"E:\emtac\dev_env\.env")
ENV_PATH = Path(os.getenv("EMTAC_ENV_PATH", str(DEFAULT_ENV_PATH)))

if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

AI_GATEWAY_URL = os.getenv(
    "AI_LOCAL_ENDPOINT",
    "http://127.0.0.1:9000/v1/chat/completions",
)

GPU_SERVICE_URL = os.getenv(
    "GPU_SERVICE_URL",
    "http://127.0.0.1:5051",
)

MCP_COORDINATOR_BASE_URL = os.getenv(
    "SERVICE_MCP_COORDINATOR_BASE_URL",
    "http://127.0.0.1:9100",
).rstrip("/")

LOCAL_MODEL = os.getenv("AI_LOCAL_MODEL", "emtac-gpu-qwen")

MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("AI_TOP_P", "0.95"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "120"))
MAX_HISTORY_MESSAGES = int(os.getenv("AI_MAX_HISTORY_MESSAGES", "12"))

AI_ENABLE_TOOLS = os.getenv("AI_ENABLE_TOOLS", "true").lower() == "true"

# Keep this false by default for local gateways/models that do not reliably
# support OpenAI-style function/tool calling.
# Deterministic MCP tool routing still works through AI_ENABLE_TOOLS.
AI_SEND_OPENAI_TOOLS = os.getenv("AI_SEND_OPENAI_TOOLS", "false").lower() == "true"

MAX_TOOL_ROUNDS = int(os.getenv("AI_MAX_TOOL_ROUNDS", "5"))

SERVICE_MCP_AI_LAYER_HOST = os.getenv("SERVICE_MCP_AI_LAYER_HOST", "127.0.0.1")
SERVICE_MCP_AI_LAYER_PORT = int(os.getenv("SERVICE_MCP_AI_LAYER_PORT", "9200"))


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def _log_info(message: str) -> None:
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO ai_rest_app: {message}",
        flush=True,
    )


def _log_warning(message: str) -> None:
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} WARNING ai_rest_app: {message}",
        flush=True,
    )


def _log_error(message: str) -> None:
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR ai_rest_app: {message}",
        flush=True,
    )


def _log_debug(message: str) -> None:
    if os.getenv("AI_REST_DEBUG", "false").lower() == "true":
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} DEBUG ai_rest_app: {message}",
            flush=True,
        )


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(
    title="EMTAC MCP AI Layer",
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------
BASE_SYSTEM_PROMPT = """\
You are Status A.I., an intelligent operations assistant embedded in the EMTAC Service Dashboard.

EMTAC runs a local services ecosystem. The dashboard monitors:
- GPU Service       — local model execution backend
- AI Gateway        — OpenAI-compatible local model gateway
- MCP Coordinator   — tool and context orchestration layer
- PostgreSQL Server — local database service
- Grafana           — observability UI

The Service Dashboard is only a UI and service monitor. Tool orchestration belongs to the MCP Server.

You have access to live tools that let you query services and system context in real time.
Use tool results when current data is needed. If a deterministic tool result is present, treat it as the source of truth.

Behavior requirements:
- Be concise, technical, and direct.
- For service health, service status, or "what is running" questions, summarize in this order: overall health -> key metrics -> concerns -> actions.
- For specific database, SQL, tool, table, log, or metric questions, answer the specific question first and do not include a full service-health summary unless the operator asks for one.
- Use markers: ok / warning / error.
- Keep responses under 250 words unless the operator asks for more detail.
- Do not invent telemetry that is not present in tool results.
- If a tool returns an error, report it clearly and suggest next steps.
- query_postgres only accepts SELECT statements. Never attempt writes.
- Do not claim you are running a tool, query, command, or SQL statement unless a real tool result is already present in the prompt.
- If a deterministic tool result is present, answer from that result directly.
- If no tool result is present, do not say "Running this query" or "I'll use the tool"; instead answer only from the provided telemetry/context.
- Do not treat unknown PID or unknown uptime as a concern when a service is reachable and marked running.
- Do not treat short uptime as a concern immediately after a service restart.
- Only list concerns for stopped services, unreachable health checks, errors, failed probes, degraded statuses, or repeated timeouts.
- If all required services are running and reachable, state that no concerns were identified.
"""


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []

    cleaned: list[dict[str, str]] = []

    for item in raw_messages:
        if not isinstance(item, dict):
            continue

        role = _safe_text(item.get("role")).strip().lower()
        content = _safe_text(item.get("content")).strip()

        if role not in {"system", "user", "assistant"}:
            continue

        if not content:
            continue

        cleaned.append(
            {
                "role": role,
                "content": content,
            }
        )

    return cleaned


def _trim_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    non_system = [m for m in messages if m["role"] != "system"]

    if len(non_system) <= MAX_HISTORY_MESSAGES:
        return non_system

    return non_system[-MAX_HISTORY_MESSAGES:]


def _probe_health(url: str, timeout: int = 3) -> tuple[bool, dict[str, Any]]:
    try:
        response = http_requests.get(url, timeout=timeout)
        ok = 200 <= response.status_code < 500

        try:
            body = response.json()
        except Exception:
            body = {"text": response.text[:500]}

        return ok, {
            "status_code": response.status_code,
            "body": body,
        }

    except Exception as exc:
        return False, {"error": str(exc)}


def _guess_gateway_health_url() -> str:
    if "/v1/chat/completions" in AI_GATEWAY_URL:
        return AI_GATEWAY_URL.replace("/v1/chat/completions", "/health")
    return AI_GATEWAY_URL.rstrip("/") + "/health"


def _dashboard_base_url() -> str:
    return os.getenv("SERVICE_DASHBOARD_BASE_URL", "http://127.0.0.1:5100").rstrip("/")


def _dashboard_get(path: str, timeout: int = 10) -> dict[str, Any]:
    url = f"{_dashboard_base_url()}{path}"
    response = http_requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _dashboard_post(path: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    url = f"{_dashboard_base_url()}{path}"
    response = http_requests.post(url, json=payload or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _mcp_coordinator_get(path: str, timeout: int = 10) -> dict[str, Any]:
    """
    Basic GET helper for coordinator-adjacent HTTP checks.

    FastMCP streamable-http does not behave like a normal REST API at '/',
    so a 404 can still prove the server is running.
    """

    url = f"{MCP_COORDINATOR_BASE_URL}{path}"
    response = http_requests.get(url, timeout=timeout)

    try:
        body = response.json()
    except Exception:
        body = {"text": response.text[:500]}

    return {
        "status_code": response.status_code,
        "reachable": response.status_code < 500,
        "body": body,
    }


def _maybe_call_mcp_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """
    Execute MCP-backed dashboard tools through the local coordinator/downstream layer.

    This intentionally uses the in-process coordinator client instead of trying to
    fake a normal REST call to /mcp. The coordinator is FastMCP streamable-http, but
    this AI layer lives in the same mcp_server project and can safely call the same
    coordinator execution path we already tested.

    Safety:
    - query_postgres is restricted to SELECT statements here.
    - The Postgres tool layer still performs its own read/write safety checks.
    - Database permissions still block writes to emtac.public.
    """

    tool_args = tool_args if isinstance(tool_args, dict) else {}

    def _clean_args(payload: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in payload.items() if v is not None and v != ""}

    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _run_postgres(target_tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from listed_server.mcp_coordinator.clients.downstream_mcp_client import (
            DownstreamMCPClient,
        )

        client = DownstreamMCPClient()
        result = client.execute(
            target_server="postgres",
            target_tool=target_tool,
            arguments=_clean_args(arguments),
        )

        return {
            "success": result.get("status") == "ok",
            "source": "mcp_coordinator_direct",
            "target_server": "postgres",
            "target_tool": target_tool,
            "arguments": _clean_args(arguments),
            "result": result,
        }

    def _route_and_run(request_text: str) -> dict[str, Any]:
        from dataclasses import asdict

        from listed_server.mcp_coordinator.clients.downstream_mcp_client import (
            DownstreamMCPClient,
        )
        from listed_server.mcp_coordinator.routing.router import CoordinatorRouter

        router = CoordinatorRouter()
        client = DownstreamMCPClient()

        decision = router.route(request_text)

        response: dict[str, Any] = {
            "success": False,
            "source": "mcp_coordinator_router",
            "request": request_text,
            "decision": asdict(decision),
        }

        if decision.needs_confirmation:
            response["message"] = (
                "The coordinator marked this request as confirmation-required. "
                "AI layer will not auto-execute confirmation-required tools."
            )
            return response

        if not decision.safe_to_execute:
            response["message"] = "The coordinator marked this request as unsafe to auto-execute."
            return response

        result = client.execute(
            target_server=decision.target_capability,
            target_tool=decision.target_tool,
            arguments=decision.suggested_arguments,
        )

        response["success"] = result.get("status") == "ok"
        response["execution_result"] = result
        return response

    try:
        # -------------------------------------------------------------
        # PostgreSQL tools
        # -------------------------------------------------------------

        if tool_name == "get_postgres_insights":
            database_name = (
                tool_args.get("database_name")
                or tool_args.get("database")
                or None
            )
            schema_name = (
                tool_args.get("schema_name")
                or tool_args.get("schema")
                or "public"
            )

            health = _run_postgres(
                "postgres_health_check",
                {"database_name": database_name},
            )
            whoami = _run_postgres(
                "postgres_whoami",
                {"database_name": database_name},
            )
            tables = _run_postgres(
                "postgres_list_tables",
                {
                    "database_name": database_name,
                    "schema_name": schema_name,
                    "include_views": True,
                },
            )

            return {
                "success": (
                    health.get("success")
                    and whoami.get("success")
                    and tables.get("success")
                ),
                "tool": tool_name,
                "database_name": database_name or "default",
                "schema_name": schema_name,
                "health": health,
                "whoami": whoami,
                "tables": tables,
            }

        if tool_name == "list_postgres_tables":
            database_name = (
                tool_args.get("database_name")
                or tool_args.get("database")
                or None
            )
            schema_name = (
                tool_args.get("schema_name")
                or tool_args.get("schema")
                or None
            )
            include_views = _as_bool(tool_args.get("include_views"), True)

            return _run_postgres(
                "postgres_list_tables",
                {
                    "database_name": database_name,
                    "schema_name": schema_name,
                    "include_views": include_views,
                },
            )

        if tool_name == "describe_postgres_table":
            table_name = (
                tool_args.get("table_name")
                or tool_args.get("table")
                or tool_args.get("name")
            )

            if not table_name:
                return {
                    "success": False,
                    "tool": tool_name,
                    "message": "table_name is required.",
                    "arguments": tool_args,
                }

            database_name = (
                tool_args.get("database_name")
                or tool_args.get("database")
                or None
            )
            schema_name = (
                tool_args.get("schema_name")
                or tool_args.get("schema")
                or None
            )

            return _run_postgres(
                "postgres_describe_table",
                {
                    "database_name": database_name,
                    "schema_name": schema_name,
                    "table_name": table_name,
                },
            )

        if tool_name == "query_postgres":
            sql = str(tool_args.get("sql") or tool_args.get("query") or "").strip()

            if not sql:
                return {
                    "success": False,
                    "tool": tool_name,
                    "message": "sql is required.",
                    "arguments": tool_args,
                }

            if not sql.lower().startswith("select"):
                return {
                    "success": False,
                    "tool": tool_name,
                    "message": "query_postgres only accepts SELECT statements.",
                    "sql_rejected": sql,
                }

            database_name = (
                tool_args.get("database_name")
                or tool_args.get("database")
                or None
            )

            return _run_postgres(
                "postgres_read_query",
                {
                    "database_name": database_name,
                    "sql": sql,
                },
            )

        # -------------------------------------------------------------
        # Grafana tools
        # -------------------------------------------------------------

        if tool_name == "get_grafana_health":
            return _route_and_run("grafana health check")

        if tool_name == "list_grafana_dashboards":
            return _route_and_run("list grafana dashboards")

        if tool_name == "list_grafana_datasources":
            return _route_and_run("list grafana datasources")

        if tool_name == "get_grafana_alert_rules":
            return _route_and_run("list grafana alert rules")

        # -------------------------------------------------------------
        # GPU insights
        # -------------------------------------------------------------

        if tool_name == "get_gpu_insights":
            ok, detail = _probe_health(
                f"{GPU_SERVICE_URL.rstrip('/')}/health",
                timeout=5,
            )

            return {
                "success": ok,
                "tool": tool_name,
                "source": "gpu_service_health",
                "url": f"{GPU_SERVICE_URL.rstrip('/')}/health",
                "detail": detail,
            }

        return {
            "success": False,
            "tool": tool_name,
            "arguments": tool_args,
            "message": f"No MCP mapping exists for tool '{tool_name}'.",
        }

    except Exception as exc:
        _log_error(
            f"MCP tool execution failed tool='{tool_name}' "
            f"args={tool_args} error={exc}"
        )

        return {
            "success": False,
            "tool": tool_name,
            "arguments": tool_args,
            "error": str(exc),
        }


# ---------------------------------------------------------------------
# Telemetry and prompt building
# ---------------------------------------------------------------------

def build_telemetry() -> str:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    lines: list[str] = [
        f"[EMTAC Live Telemetry Snapshot - {ts}]",
        f"AI Gateway      : {AI_GATEWAY_URL}",
        f"GPU Service     : {GPU_SERVICE_URL}",
        f"MCP Coordinator : {MCP_COORDINATOR_BASE_URL}",
        f"Tool Calling    : {'enabled' if AI_ENABLE_TOOLS else 'disabled'}",
    ]

    try:
        services_payload = _dashboard_get("/api/services", timeout=5)
        services = services_payload.get("services", [])

        running = [s for s in services if s.get("status") == "running"]
        stopped = [s for s in services if s.get("status") != "running"]

        lines.append(
            f"Services        : {len(services)} total - "
            f"{len(running)} running - {len(stopped)} stopped"
        )

        for svc in services:
            name = _safe_text(svc.get("name", "?"))
            status = _safe_text(svc.get("status", "unknown"))
            pid = svc.get("pid")
            uptime = svc.get("uptime_seconds")
            service_type = _safe_text(svc.get("service_type", "?"))

            icon = "ok" if status == "running" else "stopped"
            uptime_str = f"{uptime}s" if uptime is not None else "-"
            pid_str = str(pid) if pid else "-"

            lines.append(
                f"  {icon} {name}  "
                f"type={service_type}  status={status}  "
                f"pid={pid_str}  uptime={uptime_str}"
            )

    except Exception as exc:
        lines.append(f"Services        : dashboard unavailable - {exc}")

    lines.append(
        "\nUse tools when current detail is needed. "
        "The MCP Server is responsible for AI/tool orchestration."
    )

    return "\n".join(lines)


def build_system_prompt(page_context: str) -> str:
    parts = [BASE_SYSTEM_PROMPT]
    page_context = _safe_text(page_context).strip()

    if page_context:
        parts.append(f"[Page Context]\n{page_context}")

    return "\n\n".join(parts)


def inject_telemetry(
    messages: list[dict[str, str]],
    telemetry: str,
) -> list[dict[str, str]]:
    cloned = [
        {
            "role": m["role"],
            "content": m["content"],
        }
        for m in messages
    ]

    if cloned and cloned[-1]["role"] == "user":
        cloned[-1]["content"] = (
            f"[Live Telemetry Snapshot]\n{telemetry}\n\n"
            f"[Operator Query]\n{cloned[-1]['content']}"
        )

    return cloned

# ---------------------------------------------------------------------
# Deterministic tool pre-router
# ---------------------------------------------------------------------

def _latest_operator_query(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        content = _safe_text(message.get("content")).strip()

        if "[Operator Query]" in content:
            content = content.split("[Operator Query]", 1)[1].strip()

        return content

    return ""


def _extract_limit(text: str, default: int = 5, max_limit: int = 25) -> int:
    matches = re.findall(r"\bfirst\s+(\d+)\b|\btop\s+(\d+)\b|\blimit\s+(\d+)\b", text, flags=re.IGNORECASE)

    for match in matches:
        for value in match:
            if value:
                try:
                    return max(1, min(int(value), max_limit))
                except Exception:
                    return default

    return default


def _trim_tool_json_for_prompt(
    *,
    tool_name: str,
    tool_json: str,
    limit: int,
) -> str:
    try:
        parsed = json.loads(tool_json)
    except Exception:
        return tool_json[:8000]

    try:
        if tool_name == "list_postgres_tables":
            nested_result = parsed.get("result", {})
            table_rows = nested_result.get("result", [])

            if isinstance(table_rows, list):
                nested_result["total_rows_returned_by_tool"] = len(table_rows)
                nested_result["display_limit"] = limit
                nested_result["result"] = table_rows[:limit]

        if tool_name == "query_postgres":
            nested_result = parsed.get("result", {})
            read_result = nested_result.get("result", {})

            rows = read_result.get("rows")
            if isinstance(rows, list):
                read_result["total_rows_returned_by_tool"] = len(rows)
                read_result["display_limit"] = limit
                read_result["rows"] = rows[:limit]

    except Exception:
        pass

    return json.dumps(parsed, indent=2, default=str)[:8000]


def _deterministic_tool_match(operator_query: str) -> dict[str, Any] | None:
    query = _safe_text(operator_query).strip()
    query_l = query.lower()

    if not query:
        return None

    limit = _extract_limit(query, default=5, max_limit=25)

    # SELECT queries should go straight to read-only Postgres.
    if query_l.startswith("select "):
        return {
            "tool_name": "query_postgres",
            "tool_args": {
                "sql": query,
            },
            "limit": limit,
            "reason": "Operator query starts with SELECT.",
        }

    # Common wording: "run SELECT 1", "query postgres SELECT ...".
    select_match = re.search(r"\bselect\b\s+.+", query, flags=re.IGNORECASE | re.DOTALL)
    if select_match and any(word in query_l for word in {"postgres", "database", "sql", "query"}):
        return {
            "tool_name": "query_postgres",
            "tool_args": {
                "sql": select_match.group(0).strip(),
            },
            "limit": limit,
            "reason": "Operator query contains a SELECT request.",
        }

    # Count Postgres tables.
    wants_table_count = any(
        phrase in query_l
        for phrase in {
            "how many tables",
            "number of tables",
            "count tables",
            "count postgres tables",
            "postgres table count",
            "database table count",
            "tables are in",
            "tables in the database",
            "tables in emtac",
        }
    )

    if wants_table_count:
        return {
            "tool_name": "query_postgres",
            "tool_args": {
                "sql": (
                    "SELECT COUNT(*) AS table_count "
                    "FROM information_schema.tables "
                    "WHERE table_schema = 'public' "
                    "AND table_type = 'BASE TABLE';"
                ),
            },
            "limit": 1,
            "reason": "Operator query asks for a count of Postgres public tables.",
        }

    # List/show Postgres tables.
    wants_tables = any(
        phrase in query_l
        for phrase in {
            "list postgres tables",
            "show postgres tables",
            "postgres table names",
            "postgres tables",
            "database tables",
            "list tables",
            "show tables",
        }
    )

    if wants_tables:
        return {
            "tool_name": "list_postgres_tables",
            "tool_args": {
                "include_views": True,
            },
            "limit": limit,
            "reason": "Operator query asks for Postgres table names.",
        }

    if wants_tables:
        return {
            "tool_name": "list_postgres_tables",
            "tool_args": {
                "include_views": True,
            },
            "limit": limit,
            "reason": "Operator query asks for Postgres table names.",
        }

    # Describe table requests.
    describe_match = re.search(
        r"\bdescribe\s+(?:the\s+)?(?:postgres\s+)?(?:table\s+)?([a-zA-Z_][a-zA-Z0-9_\.]*)",
        query,
        flags=re.IGNORECASE,
    )

    if describe_match:
        table_name = describe_match.group(1).strip().strip(".")

        if table_name.lower() not in {"table", "postgres", "database"}:
            return {
                "tool_name": "describe_postgres_table",
                "tool_args": {
                    "table_name": table_name,
                },
                "limit": limit,
                "reason": f"Operator query asks to describe table '{table_name}'.",
            }

    return None


def _inject_deterministic_tool_result(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    operator_query = _latest_operator_query(messages)
    match = _deterministic_tool_match(operator_query)

    if not match:
        return messages, None

    tool_name = match["tool_name"]
    tool_args = match["tool_args"]
    limit = int(match.get("limit") or 5)

    _log_info(
        "deterministic tool match | "
        f"tool='{tool_name}' reason='{match.get('reason')}' args={tool_args}"
    )

    tool_json = _execute_tool(tool_name, tool_args)
    trimmed_tool_json = _trim_tool_json_for_prompt(
        tool_name=tool_name,
        tool_json=tool_json,
        limit=limit,
    )

    tool_context = (
        "[Deterministic Tool Result]\n"
        f"Tool: {tool_name}\n"
        f"Reason: {match.get('reason')}\n"
        f"Display limit: {limit}\n"
        f"Result JSON:\n{trimmed_tool_json}\n\n"
        "Use the tool result above as the source of truth. "
        "Do not invent rows, tables, services, metrics, or statuses not present in the result."
    )

    updated_messages = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in messages
    ]

    for index in range(len(updated_messages) - 1, -1, -1):
        if updated_messages[index]["role"] == "user":
            updated_messages[index]["content"] = (
                f"{updated_messages[index]['content']}\n\n"
                f"{tool_context}"
            )
            break

    return updated_messages, {
        "tool_name": tool_name,
        "tool_args": tool_args,
        "reason": match.get("reason"),
        "limit": limit,
    }

# ---------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------

def _execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    _log_info(f"executing tool='{tool_name}' args={tool_args}")

    try:
        if tool_name == "mcp_coordinator_status":
            result = _mcp_coordinator_get("/", timeout=5)
            return json.dumps(result, default=str)

        if tool_name == "ai_gateway_health":
            ok, detail = _probe_health(_guess_gateway_health_url(), timeout=5)
            return json.dumps(
                {
                    "reachable": ok,
                    "detail": detail,
                },
                default=str,
            )

        if tool_name == "gpu_service_health":
            ok, detail = _probe_health(f"{GPU_SERVICE_URL.rstrip('/')}/health", timeout=5)
            return json.dumps(
                {
                    "reachable": ok,
                    "detail": detail,
                },
                default=str,
            )

        if tool_name == "list_services":
            result = _dashboard_get("/api/services", timeout=10)
            return json.dumps(result, default=str)

        if tool_name == "get_service_logs":
            name = _safe_text(tool_args.get("name")).strip()
            lines = min(int(tool_args.get("lines", 50)), 200)

            result = _dashboard_get("/api/services", timeout=10)
            services = result.get("services", [])

            for svc in services:
                if _safe_text(svc.get("name")) == name:
                    output = svc.get("output") or []
                    return json.dumps(
                        {
                            "name": name,
                            "status": svc.get("status"),
                            "total_lines_buffered": len(output),
                            "lines_returned": min(lines, len(output)),
                            "output": output[-lines:],
                        },
                        default=str,
                    )

            return json.dumps(
                {
                    "error": f"Service '{name}' not found.",
                    "known_services": [s.get("name") for s in services],
                },
                default=str,
            )

        if tool_name == "start_service":
            name = _safe_text(tool_args.get("name")).strip()
            if not name:
                return json.dumps({"error": "name is required"}, default=str)
            return json.dumps(
                _dashboard_post(f"/api/services/{name}/start", timeout=60),
                default=str,
            )

        if tool_name == "stop_service":
            name = _safe_text(tool_args.get("name")).strip()
            if not name:
                return json.dumps({"error": "name is required"}, default=str)
            return json.dumps(
                _dashboard_post(f"/api/services/{name}/stop", timeout=60),
                default=str,
            )

        if tool_name == "restart_service":
            name = _safe_text(tool_args.get("name")).strip()
            if not name:
                return json.dumps({"error": "name is required"}, default=str)
            return json.dumps(
                _dashboard_post(f"/api/services/{name}/restart", timeout=90),
                default=str,
            )

        if tool_name == "clear_service_logs":
            name = _safe_text(tool_args.get("name")).strip()
            if not name:
                return json.dumps({"error": "name is required"}, default=str)
            return json.dumps(
                _dashboard_post(f"/api/services/{name}/clear-output", timeout=30),
                default=str,
            )

        if tool_name in {
            "get_postgres_insights",
            "list_postgres_tables",
            "describe_postgres_table",
            "query_postgres",
            "get_grafana_health",
            "list_grafana_dashboards",
            "list_grafana_datasources",
            "get_grafana_alert_rules",
            "get_gpu_insights",
        }:
            result = _maybe_call_mcp_tool(tool_name, tool_args)
            return json.dumps(result, default=str)

        return json.dumps({"error": f"Unknown tool: '{tool_name}'"}, default=str)

    except Exception as exc:
        _log_error(f"tool execution error tool='{tool_name}' error={exc}")
        return json.dumps({"error": str(exc)}, default=str)


# ---------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------

def call_local(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> tuple[str, str]:
    full_messages: list[dict[str, Any]] = (
        [{"role": "system", "content": system_prompt}] + messages
    )

    model_used = LOCAL_MODEL

    # AI_ENABLE_TOOLS controls deterministic MCP pre-routing.
    # AI_SEND_OPENAI_TOOLS controls whether we send OpenAI tool schemas
    # to the local AI gateway. Keep this false unless the gateway/model
    # reliably supports tool_calls.
    tools_active = AI_ENABLE_TOOLS and AI_SEND_OPENAI_TOOLS

    for round_num in range(MAX_TOOL_ROUNDS + 1):
        payload: dict[str, Any] = {
            "model": LOCAL_MODEL,
            "messages": full_messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "stream": False,
        }

        if tools_active:
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"

        _log_info(
            f"gateway call round={round_num} "
            f"endpoint={AI_GATEWAY_URL} model={LOCAL_MODEL} "
            f"messages={len(full_messages)} "
            f"tools={'on' if tools_active else 'off'}"
        )

        response = http_requests.post(
            AI_GATEWAY_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code >= 400 and tools_active:
            response_text = ""

            try:
                response_text = response.text[:500]
            except Exception:
                response_text = ""

            _log_warning(
                "gateway rejected or failed tool payload "
                f"status={response.status_code}; retrying without tools; "
                f"body={response_text}"
            )

            tools_active = False
            continue

        response.raise_for_status()

        try:
            data = response.json()
        except Exception as exc:
            if tools_active:
                _log_warning(
                    "gateway returned non-JSON while tools were enabled; "
                    f"retrying without tools error={exc}"
                )
                tools_active = False
                continue

            raise

        if not isinstance(data, dict):
            if tools_active:
                _log_warning(
                    "gateway returned non-dict JSON while tools were enabled; "
                    "retrying without tools"
                )
                tools_active = False
                continue

            raise ValueError(f"AI Gateway returned unexpected JSON type: {type(data)}")

        model_used = data.get("model", LOCAL_MODEL)
        choice = (data.get("choices") or [{}])[0]

        if not isinstance(choice, dict):
            raise ValueError("AI Gateway returned invalid choices payload.")

        message = choice.get("message", {})

        if not isinstance(message, dict):
            raise ValueError("AI Gateway returned invalid message payload.")

        tool_calls = message.get("tool_calls") or []

        if tool_calls and tools_active and round_num < MAX_TOOL_ROUNDS:
            full_messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )

            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{round_num}")
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")

                try:
                    fn_args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    fn_args = {}

                tool_result = _execute_tool(fn_name, fn_args)

                _log_debug(
                    f"tool result tool='{fn_name}' result_chars={len(tool_result)}"
                )

                full_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_result,
                    }
                )

            continue

        if round_num >= MAX_TOOL_ROUNDS and tool_calls:
            _log_warning(
                f"hit MAX_TOOL_ROUNDS={MAX_TOOL_ROUNDS}; forcing final answer"
            )

            payload["messages"] = full_messages
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

            response = http_requests.post(
                AI_GATEWAY_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            model_used = data.get("model", LOCAL_MODEL)
            message = (data.get("choices") or [{}])[0].get("message", {})

        reply = message.get("content") or "No response received."

        _log_info(
            f"final reply reply_chars={len(reply)} "
            f"model_used={model_used} rounds_used={round_num + 1}"
        )

        return reply, model_used

    return "Max tool rounds exceeded with no final answer.", model_used


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/api/ai/health")
def ai_health() -> dict[str, Any]:
    gateway_ok, gateway_status = _probe_health(_guess_gateway_health_url(), timeout=3)
    gpu_ok, gpu_status = _probe_health(f"{GPU_SERVICE_URL.rstrip('/')}/health", timeout=3)
    mcp_ok, mcp_status = _probe_health(MCP_COORDINATOR_BASE_URL, timeout=3)

    overall = "ok" if gateway_ok and gpu_ok and mcp_ok else "degraded"

    return {
        "status": overall,
        "env_path": str(ENV_PATH),
        "ai_gateway": {
            "url": _guess_gateway_health_url(),
            "reachable": gateway_ok,
            "detail": gateway_status,
        },
        "gpu": {
            "url": f"{GPU_SERVICE_URL.rstrip('/')}/health",
            "reachable": gpu_ok,
            "detail": gpu_status,
        },
        "mcp_coordinator": {
            "url": MCP_COORDINATOR_BASE_URL,
            "reachable": mcp_ok,
            "detail": mcp_status,
        },
        "model": LOCAL_MODEL,
        "tools_enabled": AI_ENABLE_TOOLS,
        "openai_tool_payload_enabled": AI_SEND_OPENAI_TOOLS,
        "available_tools": get_tool_names() if AI_ENABLE_TOOLS else [],
        "max_tool_rounds": MAX_TOOL_ROUNDS,
    }


@app.get("/api/ai/tools")
def ai_tools() -> dict[str, Any]:
    return {
        "tools_enabled": AI_ENABLE_TOOLS,
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "tools": TOOLS if AI_ENABLE_TOOLS else [],
    }


@app.post("/api/ai/chat")
async def ai_chat(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    raw_messages = body.get("messages", [])
    page_context = body.get("page_context", "")

    messages = _normalize_messages(raw_messages)
    messages = _trim_history(messages)

    if not messages:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid messages provided"},
        )

    telemetry = build_telemetry()
    system_prompt = build_system_prompt(page_context)
    messages = inject_telemetry(messages, telemetry)

    deterministic_tool: dict[str, Any] | None = None

    if AI_ENABLE_TOOLS:
        messages, deterministic_tool = _inject_deterministic_tool_result(messages)

    _log_info(
        "request | "
        f"messages={len(messages)} "
        f"telemetry_chars={len(telemetry)} "
        f"tools_enabled={AI_ENABLE_TOOLS} "
        f"deterministic_tool={deterministic_tool.get('tool_name') if deterministic_tool else 'none'}"
    )

    try:
        reply_text, model_used = call_local(messages, system_prompt)

    except http_requests.exceptions.ConnectionError as exc:
        _log_error(f"cannot reach AI gateway error={exc}")
        return JSONResponse(
            status_code=503,
            content={
                "error": (
                    f"AI Gateway unreachable at {AI_GATEWAY_URL}. "
                    "Is the AI Gateway service running?"
                )
            },
        )

    except http_requests.exceptions.Timeout:
        _log_error("AI gateway timed out")
        return JSONResponse(
            status_code=504,
            content={
                "error": (
                    f"AI Gateway timed out ({REQUEST_TIMEOUT_SECONDS}s). "
                    "GPU may be busy."
                )
            },
        )

    except http_requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502

        try:
            response_text = exc.response.text[:500] if exc.response is not None else ""
        except Exception:
            response_text = ""

        _log_error(
            "AI gateway HTTP error "
            f"status={status_code} body={response_text}"
        )

        return JSONResponse(
            status_code=502,
            content={
                "error": f"AI Gateway returned HTTP {status_code}",
                "detail": response_text,
            },
        )

    except Exception as exc:
        _log_error(f"unexpected error error={exc}")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )

    response_content: dict[str, Any] = {
        "id": "chatcmpl-statusai",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply_text,
                },
                "finish_reason": "stop",
            }
        ],
    }

    if deterministic_tool:
        response_content["deterministic_tool"] = deterministic_tool

    return JSONResponse(content=response_content)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:
    _log_info(
        "Starting EMTAC MCP AI Layer REST API on "
        f"{SERVICE_MCP_AI_LAYER_HOST}:{SERVICE_MCP_AI_LAYER_PORT}"
    )
    _log_info(f"Environment path: {ENV_PATH}")
    _log_info(f"AI Gateway URL: {AI_GATEWAY_URL}")
    _log_info(f"MCP Coordinator URL: {MCP_COORDINATOR_BASE_URL}")

    uvicorn.run(
        "ai_layer.ai_rest_app:app",
        host=SERVICE_MCP_AI_LAYER_HOST,
        port=SERVICE_MCP_AI_LAYER_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()