# services/service_dashboard/app/api/ai_chat_route.py

from __future__ import annotations

"""
Status A.I. — Flask Blueprint (local LLM via MCP/OpenAI-compatible gateway)
Path: services/service_dashboard/app/api/ai_chat_route.py

Flow:
Dashboard UI
    -> /api/ai/chat
    -> MCP / OpenAI-compatible gateway (:9000)  [with tool calling / agentic loop]
    -> GPU service (:5051)
    -> Local Qwen model

Agentic tool loop:
    1. Send messages + tool definitions to gateway
    2. If model returns tool_calls  → execute against ServiceManager → append results → repeat
    3. If model returns text        → return to caller
    4. Cap at MAX_TOOL_ROUNDS to prevent runaway loops

Register in main.py:

    from app.api.ai_chat_route import ai_bp, init_ai_blueprint
    init_ai_blueprint(service_manager)
    app.register_blueprint(ai_bp)
"""

import json
import os
import time
from typing import Any, Optional

import requests as http_requests
from flask import Blueprint, jsonify, request

from app.services.service_dashboard_logger import (
    dash_debug,
    dash_error,
    dash_info,
    dash_warning,
)

ai_bp = Blueprint("ai", __name__, url_prefix="/api/ai")

# -------------------------------------------------------------------
# Injected at startup
# -------------------------------------------------------------------
_service_manager = None


def init_ai_blueprint(service_manager) -> None:
    global _service_manager
    _service_manager = service_manager
    dash_info("ai_chat_route: ServiceManager injected into AI blueprint")


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

MCP_GATEWAY_URL = os.getenv(
    "AI_LOCAL_ENDPOINT",
    "http://127.0.0.1:9000/v1/chat/completions",
)
GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL", "http://127.0.0.1:5051")
LOCAL_MODEL = os.getenv("AI_LOCAL_MODEL", "emtac-gpu-qwen")

MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("AI_TOP_P", "0.95"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "120"))
MAX_HISTORY_MESSAGES = int(os.getenv("AI_MAX_HISTORY_MESSAGES", "12"))

# Set to "false" to disable tool calling (e.g. if your gateway doesn't support it)
AI_ENABLE_TOOLS = os.getenv("AI_ENABLE_TOOLS", "true").lower() == "true"

# Max tool-call → result → re-inference rounds before we force a text reply
MAX_TOOL_ROUNDS = int(os.getenv("AI_MAX_TOOL_ROUNDS", "5"))

BASE_SYSTEM_PROMPT = """\
You are Status A.I., an intelligent operations assistant embedded in the EMTAC Service Dashboard.

EMTAC runs a service dashboard that manages local infrastructure. The dashboard monitors and controls:
- GPU Service       — local model execution backend
- AI Gateway        — OpenAI-compatible MCP/gateway proxy
- PostgreSQL Server — local database service
- Grafana           — observability UI (dashboards, datasources, alerts)
- Other managed EMTAC services started by the ServiceManager

You have access to live tools that let you query and control services in real time.
Always use tools to get current data before answering operational questions — do not rely on
the telemetry snapshot alone when a tool can give you fresher or more detailed information.

Tool use guidelines:
- To answer "what services are running?" → call list_services
- To investigate a specific service → call get_service_logs with that service name
- To check GPU state → call get_gpu_insights
- To answer Grafana questions (health, dashboards, datasources, alerts)
  → call the matching get_grafana_* / list_grafana_* tool
- To start / stop / restart a service → call the appropriate control tool
- Chain tools when needed: e.g. check status, then act, then verify

Behavior requirements:
- Be concise, technical, and direct.
- Lead summaries with: overall health → key metrics → concerns → actions.
- Use markers:  ✓ ok   ⚠ warning   ✗ error
- Keep responses under 250 words unless the operator asks for more detail.
- Do not invent telemetry that is not present in tool results.
- If a tool returns an error, report it clearly and suggest next steps.
- Always confirm the result of a control action (start/stop/restart) by calling list_services
  or get_service_logs after acting.
"""


# -------------------------------------------------------------------
# Tool definitions  (OpenAI function-calling format)
# -------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    # ── Service management ─────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "list_services",
            "description": (
                "Return the current status of all managed services: name, type, "
                "status (running/stopped), PID, and uptime in seconds."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_logs",
            "description": (
                "Fetch the recent output / log lines for a named service. "
                "Use this to diagnose errors, check startup output, or read recent activity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the service as registered in ServiceManager.",
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of most-recent lines to return (default 50, max 200).",
                        "default": 50,
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_service",
            "description": "Start a stopped managed service by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the service to start.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_service",
            "description": "Stop a running managed service by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the service to stop.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "Restart a managed service (stop then start) by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the service to restart.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_service_logs",
            "description": "Clear the in-memory output buffer for a named service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the service whose logs to clear.",
                    }
                },
                "required": ["name"],
            },
        },
    },

    # ── GPU ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_gpu_insights",
            "description": (
                "Fetch detailed GPU service insights: GPU hardware metrics (VRAM, utilisation, "
                "temperature), process info, and model status from the GPU service API."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },

    # ── Grafana ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_grafana_health",
            "description": (
                "Fetch Grafana's /api/health endpoint: database status, "
                "version, and commit. Use this to confirm Grafana is reachable "
                "and healthy, not just that the process is running."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_grafana_dashboards",
            "description": (
                "List Grafana dashboards via /api/search. Use to answer "
                "'what dashboards do we have?' or to find a dashboard by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional title substring to filter by.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 50, max 200).",
                        "default": 50,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_grafana_datasources",
            "description": (
                "List configured Grafana datasources (Prometheus, Postgres, "
                "Loki, etc). Requires Grafana admin auth."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grafana_alert_rules",
            "description": (
                "Return configured Grafana alert rules via the provisioning "
                "API. Use to answer 'what alerts are set up?'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# -------------------------------------------------------------------
# Service lookup helpers
# -------------------------------------------------------------------

def _get_gpu_service():
    """Locate the managed GPU service regardless of exact registration name."""
    if _service_manager is None:
        return None
    for candidate in ("GPU Service", "gpu_service", "gpu"):
        svc = _service_manager.get_service(candidate)
        if svc:
            return svc
    for svc in _service_manager.services.values():
        if "gpu" in svc.name.lower():
            return svc
    return None


def _get_grafana_service():
    """Locate the managed Grafana service regardless of exact registration name."""
    if _service_manager is None:
        return None
    for candidate in ("Grafana", "grafana"):
        svc = _service_manager.get_service(candidate)
        if svc:
            return svc
    for svc in _service_manager.services.values():
        if "grafana" in svc.name.lower():
            return svc
    return None


# -------------------------------------------------------------------
# Tool executor  — maps model tool calls → ServiceManager actions
# -------------------------------------------------------------------

def _execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    """
    Dispatch a tool call from the model to the appropriate ServiceManager method
    or helper function. Always returns a JSON string so it can be fed back to
    the model as a tool result message.
    """
    dash_info(f"ai_chat_route: executing tool='{tool_name}' args={tool_args}")

    try:
        # ── list_services ──────────────────────────────────────────
        if tool_name == "list_services":
            if _service_manager is None:
                return json.dumps({"error": "ServiceManager not initialised."})
            services = _service_manager.get_service_data()
            summary = [
                {
                    "name": s["name"],
                    "service_type": s["service_type"],
                    "status": s["status"],
                    "pid": s["pid"],
                    "uptime_seconds": s["uptime_seconds"],
                }
                for s in services
            ]
            return json.dumps({"services": summary, "count": len(summary)})

        # ── get_service_logs ────────────────────────────────────────
        if tool_name == "get_service_logs":
            if _service_manager is None:
                return json.dumps({"error": "ServiceManager not initialised."})
            name = tool_args.get("name", "")
            lines = min(int(tool_args.get("lines", 50)), 200)
            service = _service_manager.get_service(name)
            if service is None:
                known = list(_service_manager.services.keys())
                return json.dumps({
                    "error": f"Service '{name}' not found.",
                    "known_services": known,
                })
            output = service.get_output()
            return json.dumps({
                "name": name,
                "status": service.get_status(),
                "total_lines_buffered": len(output),
                "lines_returned": min(lines, len(output)),
                "output": output[-lines:],
            })

        # ── start_service ───────────────────────────────────────────
        if tool_name == "start_service":
            if _service_manager is None:
                return json.dumps({"error": "ServiceManager not initialised."})
            name = tool_args.get("name", "")
            result = _service_manager.start_service(name)
            return json.dumps(result)

        # ── stop_service ────────────────────────────────────────────
        if tool_name == "stop_service":
            if _service_manager is None:
                return json.dumps({"error": "ServiceManager not initialised."})
            name = tool_args.get("name", "")
            result = _service_manager.stop_service(name)
            return json.dumps(result)

        # ── restart_service ─────────────────────────────────────────
        if tool_name == "restart_service":
            if _service_manager is None:
                return json.dumps({"error": "ServiceManager not initialised."})
            name = tool_args.get("name", "")
            result = _service_manager.restart_service(name)
            return json.dumps(result)

        # ── clear_service_logs ──────────────────────────────────────
        if tool_name == "clear_service_logs":
            if _service_manager is None:
                return json.dumps({"error": "ServiceManager not initialised."})
            name = tool_args.get("name", "")
            service = _service_manager.get_service(name)
            if service is None:
                return json.dumps({"error": f"Service '{name}' not found."})
            service.clear_output()
            return json.dumps({"success": True, "message": f"Logs cleared for '{name}'."})

        # ── get_gpu_insights ────────────────────────────────────────
        if tool_name == "get_gpu_insights":
            try:
                from app.services.gpu_insights_service import get_gpu_service_insights
            except ImportError as exc:
                return json.dumps({
                    "error": f"gpu_insights_service module not available: {exc}"
                })

            result = get_gpu_service_insights(
                _get_gpu_service(), base_url=GPU_SERVICE_URL
            )
            return json.dumps(result)

        # ── get_grafana_health ──────────────────────────────────────
        if tool_name == "get_grafana_health":
            try:
                from app.services.grafana_insights_service import get_grafana_health
            except ImportError as exc:
                return json.dumps({
                    "error": f"grafana_insights_service module not available: {exc}"
                })
            return json.dumps(get_grafana_health(_get_grafana_service()))

        # ── list_grafana_dashboards ─────────────────────────────────
        if tool_name == "list_grafana_dashboards":
            try:
                from app.services.grafana_insights_service import list_grafana_dashboards
            except ImportError as exc:
                return json.dumps({
                    "error": f"grafana_insights_service module not available: {exc}"
                })
            return json.dumps(list_grafana_dashboards(
                _get_grafana_service(),
                query=tool_args.get("query"),
                limit=int(tool_args.get("limit", 50)),
            ))

        # ── list_grafana_datasources ────────────────────────────────
        if tool_name == "list_grafana_datasources":
            try:
                from app.services.grafana_insights_service import list_grafana_datasources
            except ImportError as exc:
                return json.dumps({
                    "error": f"grafana_insights_service module not available: {exc}"
                })
            return json.dumps(list_grafana_datasources(_get_grafana_service()))

        # ── get_grafana_alert_rules ─────────────────────────────────
        if tool_name == "get_grafana_alert_rules":
            try:
                from app.services.grafana_insights_service import get_grafana_alert_rules
            except ImportError as exc:
                return json.dumps({
                    "error": f"grafana_insights_service module not available: {exc}"
                })
            return json.dumps(get_grafana_alert_rules(_get_grafana_service()))

        # ── unknown tool ────────────────────────────────────────────
        return json.dumps({"error": f"Unknown tool: '{tool_name}'"})

    except Exception as exc:
        dash_error(f"ai_chat_route: tool execution error tool='{tool_name}' error={exc}")
        return json.dumps({"error": str(exc)})


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

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
        cleaned.append({"role": role, "content": content})
    return cleaned


def _trim_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) <= MAX_HISTORY_MESSAGES:
        return non_system
    return non_system[-MAX_HISTORY_MESSAGES:]


def build_telemetry() -> str:
    """
    Build a lightweight telemetry snapshot that is prepended to the first
    user message.  The model can drill into any service using tools.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines: list[str] = [
        f"[EMTAC Live Telemetry Snapshot — {ts}]",
        f"MCP Gateway : {MCP_GATEWAY_URL}",
        f"GPU Service : {GPU_SERVICE_URL}",
        f"Tool Calling: {'enabled' if AI_ENABLE_TOOLS else 'disabled'}",
    ]

    if _service_manager is None:
        lines.append("Services    : ServiceManager not initialised")
        return "\n".join(lines)

    try:
        services: list[dict[str, Any]] = _service_manager.get_service_data()
    except Exception as exc:
        dash_warning(f"ai_chat_route: failed to collect service data error={exc}")
        lines.append(f"Services    : error — {exc}")
        return "\n".join(lines)

    running = [s for s in services if s.get("status") == "running"]
    stopped = [s for s in services if s.get("status") != "running"]

    lines.append(
        f"Services    : {len(services)} total · "
        f"{len(running)} running · {len(stopped)} stopped"
    )

    for svc in services:
        name = _safe_text(svc.get("name", "?"))
        status = _safe_text(svc.get("status", "unknown"))
        pid = svc.get("pid")
        uptime = svc.get("uptime_seconds")
        service_type = _safe_text(svc.get("service_type", "?"))

        icon = "✓" if status == "running" else "✗"
        uptime_str = f"{uptime}s" if uptime is not None else "—"
        pid_str = str(pid) if pid else "—"

        lines.append(
            f"  {icon} {name}  "
            f"type={service_type}  status={status}  "
            f"pid={pid_str}  uptime={uptime_str}"
        )

    lines.append(
        "\nUse tools (list_services, get_service_logs, get_gpu_insights, "
        "get_grafana_health, list_grafana_dashboards, etc.) "
        "to get full detail on any service."
    )

    dash_debug(
        f"ai_chat_route: telemetry built "
        f"services={len(services)} chars={sum(len(l) for l in lines)}"
    )
    return "\n".join(lines)


def build_system_prompt(page_context: str) -> str:
    parts = [BASE_SYSTEM_PROMPT]
    page_context = _safe_text(page_context).strip()
    if page_context:
        parts.append(f"[Page Context]\n{page_context}")
    return "\n\n".join(parts)


def inject_telemetry(
    messages: list[dict[str, str]], telemetry: str
) -> list[dict[str, str]]:
    cloned = [{"role": m["role"], "content": m["content"]} for m in messages]
    if cloned and cloned[-1]["role"] == "user":
        cloned[-1]["content"] = (
            f"[Live Telemetry Snapshot]\n{telemetry}\n\n"
            f"[Operator Query]\n{cloned[-1]['content']}"
        )
    return cloned


# -------------------------------------------------------------------
# Core inference  —  agentic tool loop
# -------------------------------------------------------------------

def call_local(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> tuple[str, str]:
    """
    POST to the MCP / OpenAI-compatible gateway with optional tool calling.

    If AI_ENABLE_TOOLS is True:
      - Attaches TOOLS to the request.
      - After each response, if the model emits tool_calls, executes them
        and feeds results back as tool-role messages, then re-infers.
      - Loops up to MAX_TOOL_ROUNDS times.
      - Falls back to plain text mode if the gateway rejects tool params.

    Returns (reply_text, model_used).
    """
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

        dash_info(
            f"ai_chat_route: gateway call round={round_num} "
            f"endpoint={MCP_GATEWAY_URL} model={LOCAL_MODEL} "
            f"messages={len(full_messages)} tools={'on' if tools_active else 'off'}"
        )

        response = http_requests.post(
            MCP_GATEWAY_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        # If the gateway doesn't support tools, retry without them
        if response.status_code in (400, 422) and tools_active:
            dash_warning(
                "ai_chat_route: gateway rejected tool payload "
                f"status={response.status_code} — retrying without tools"
            )
            tools_active = False
            continue

        response.raise_for_status()
        data = response.json()

        model_used = data.get("model", LOCAL_MODEL)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")

        # ── Tool calls requested by model ──────────────────────────
        tool_calls = message.get("tool_calls") or []

        if tool_calls and tools_active and round_num < MAX_TOOL_ROUNDS:
            # Append the assistant's tool-call message to history
            full_messages.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            })

            # Execute every requested tool and append results
            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{round_num}")
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                try:
                    fn_args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    fn_args = {}

                tool_result = _execute_tool(fn_name, fn_args)

                dash_debug(
                    f"ai_chat_route: tool result tool='{fn_name}' "
                    f"result_chars={len(tool_result)}"
                )

                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result,
                })

            # Continue the loop — re-infer with tool results in context
            continue

        # ── Final text response ────────────────────────────────────
        if round_num >= MAX_TOOL_ROUNDS and tool_calls:
            dash_warning(
                f"ai_chat_route: hit MAX_TOOL_ROUNDS={MAX_TOOL_ROUNDS}, "
                "forcing final answer"
            )
            # One last call without tools so the model gives a text summary
            payload["messages"] = full_messages
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
            response = http_requests.post(
                MCP_GATEWAY_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            model_used = data.get("model", LOCAL_MODEL)
            message = (data.get("choices") or [{}])[0].get("message", {})

        reply = message.get("content") or "No response received."

        dash_info(
            f"ai_chat_route: final reply "
            f"reply_chars={len(reply)} model_used={model_used} "
            f"rounds_used={round_num + 1}"
        )
        return reply, model_used

    # Should not be reached
    return "Max tool rounds exceeded with no final answer.", model_used


# -------------------------------------------------------------------
# Health probe helpers
# -------------------------------------------------------------------

def _probe_health(url: str, timeout: int = 3) -> tuple[bool, dict[str, Any]]:
    try:
        r = http_requests.get(url, timeout=timeout)
        ok = r.status_code == 200
        try:
            body = r.json()
        except Exception:
            body = {"text": r.text[:300]}
        return ok, body if ok else {"http": r.status_code, "detail": body}
    except Exception as exc:
        return False, {"error": str(exc)}


def _guess_gateway_health_url() -> str:
    if "/v1/chat/completions" in MCP_GATEWAY_URL:
        return MCP_GATEWAY_URL.replace("/v1/chat/completions", "/health")
    return MCP_GATEWAY_URL.rstrip("/") + "/health"


# -------------------------------------------------------------------
# POST /api/ai/chat
# -------------------------------------------------------------------

@ai_bp.route("/chat", methods=["POST"])
def ai_chat():
    """
    Status A.I. proxy → MCP Gateway → GPU → Qwen.

    Injects a live telemetry snapshot and runs an agentic tool loop so the
    model can query and control every registered service before answering.
    """
    body = request.get_json(force=True, silent=True) or {}
    raw_messages = body.get("messages", [])
    page_context = body.get("page_context", "")

    messages = _normalize_messages(raw_messages)
    messages = _trim_history(messages)

    if not messages:
        return jsonify({"error": "No valid messages provided"}), 400

    telemetry = build_telemetry()
    system_prompt = build_system_prompt(page_context)
    messages = inject_telemetry(messages, telemetry)

    dash_info(
        "ai_chat_route: request | "
        f"messages={len(messages)} "
        f"telemetry_chars={len(telemetry)} "
        f"tools_enabled={AI_ENABLE_TOOLS}"
    )

    try:
        reply_text, model_used = call_local(messages, system_prompt)

    except http_requests.exceptions.ConnectionError as exc:
        dash_error(f"ai_chat_route: cannot reach MCP gateway error={exc}")
        return jsonify({
            "error": (
                f"MCP Gateway unreachable at {MCP_GATEWAY_URL}. "
                "Is the AI Gateway service running?"
            )
        }), 503

    except http_requests.exceptions.Timeout:
        dash_error("ai_chat_route: MCP gateway timed out")
        return jsonify({
            "error": (
                f"MCP Gateway timed out ({REQUEST_TIMEOUT_SECONDS}s). "
                "GPU may be busy."
            )
        }), 504

    except http_requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        response_text = ""
        try:
            response_text = exc.response.text[:500] if exc.response is not None else ""
        except Exception:
            response_text = ""
        dash_error(
            "ai_chat_route: MCP gateway HTTP error "
            f"status={status_code} body={response_text}"
        )
        return jsonify({
            "error": f"MCP Gateway returned HTTP {status_code}",
            "detail": response_text,
        }), 502

    except Exception as exc:
        dash_error(f"ai_chat_route: unexpected error error={exc}")
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "id": "chatcmpl-statusai",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop",
            }
        ],
    })


# -------------------------------------------------------------------
# GET /api/ai/health
# -------------------------------------------------------------------

@ai_bp.route("/health", methods=["GET"])
def ai_health():
    """
    Probe the MCP gateway and GPU service so the frontend can display state.
    Also reports available tools and ServiceManager status.
    """
    gateway_health_url = _guess_gateway_health_url()
    gateway_ok, gateway_status = _probe_health(gateway_health_url, timeout=3)
    gpu_ok, gpu_status = _probe_health(f"{GPU_SERVICE_URL}/health", timeout=3)

    overall = "ok" if (gateway_ok and gpu_ok) else "degraded"

    available_tools = [t["function"]["name"] for t in TOOLS] if AI_ENABLE_TOOLS else []

    return jsonify({
        "status": overall,
        "gateway": {
            "url": gateway_health_url,
            "reachable": gateway_ok,
            "detail": gateway_status,
        },
        "gpu": {
            "url": f"{GPU_SERVICE_URL}/health",
            "reachable": gpu_ok,
            "detail": gpu_status,
        },
        "model": LOCAL_MODEL,
        "service_manager": (
            "available" if _service_manager is not None else "unavailable"
        ),
        "tools_enabled": AI_ENABLE_TOOLS,
        "available_tools": available_tools,
        "max_tool_rounds": MAX_TOOL_ROUNDS,
    })


# -------------------------------------------------------------------
# GET /api/ai/tools
# -------------------------------------------------------------------

@ai_bp.route("/tools", methods=["GET"])
def ai_tools():
    """
    Return the full tool schema so the frontend can display what the AI can do.
    """
    return jsonify({
        "tools_enabled": AI_ENABLE_TOOLS,
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "tools": TOOLS if AI_ENABLE_TOOLS else [],
    })