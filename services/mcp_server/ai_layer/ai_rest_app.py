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
Always use tools to get current data before answering operational questions when a tool can provide fresher information.

Behavior requirements:
- Be concise, technical, and direct.
- Lead summaries with: overall health -> key metrics -> concerns -> actions.
- Use markers:  ok / warning / error.
- Keep responses under 250 words unless the operator asks for more detail.
- Do not invent telemetry that is not present in tool results.
- If a tool returns an error, report it clearly and suggest next steps.
- query_postgres only accepts SELECT statements. Never attempt writes.
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
    Placeholder for the future true MCP client call.

    The coordinator is running as FastMCP streamable-http. The next refactor
    should replace this function with a real MCP client invocation.

    For now, this returns a clear message instead of silently pretending.
    """

    return {
        "success": False,
        "tool": tool_name,
        "arguments": tool_args,
        "message": (
            "MCP tool execution is not fully wired yet in ai_rest_app.py. "
            "The AI layer is now in mcp_server, but this tool still needs to be "
            "connected to listed_server/mcp_coordinator or a REST wrapper."
        ),
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
    tools_active = AI_ENABLE_TOOLS

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
            f"messages={len(full_messages)} tools={'on' if tools_active else 'off'}"
        )

        response = http_requests.post(
            AI_GATEWAY_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code in (400, 422) and tools_active:
            _log_warning(
                "gateway rejected tool payload "
                f"status={response.status_code}; retrying without tools"
            )
            tools_active = False
            continue

        response.raise_for_status()
        data = response.json()

        model_used = data.get("model", LOCAL_MODEL)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})

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

    _log_info(
        "request | "
        f"messages={len(messages)} "
        f"telemetry_chars={len(telemetry)} "
        f"tools_enabled={AI_ENABLE_TOOLS}"
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

    return JSONResponse(
        content={
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
    )


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