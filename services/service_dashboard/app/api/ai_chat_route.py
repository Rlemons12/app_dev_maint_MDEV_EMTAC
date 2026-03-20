from __future__ import annotations

"""
Status A.I. — Flask Blueprint (local LLM via MCP/OpenAI-compatible gateway)
Path: services/service_dashboard/app/api/ai_chat_route.py

Flow:
Dashboard UI
    -> /api/ai/chat
    -> MCP / OpenAI-compatible gateway (:9000)
    -> GPU service (:5051)
    -> Local Qwen model

Register in main.py:

    from app.api.ai_chat_route import ai_bp, init_ai_blueprint
    init_ai_blueprint(service_manager)
    app.register_blueprint(ai_bp)
"""

import os
import time
from typing import Any

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

# This is your OpenAI-compatible MCP/gateway endpoint
MCP_GATEWAY_URL = os.getenv(
    "AI_LOCAL_ENDPOINT",
    "http://127.0.0.1:9000/v1/chat/completions",
)

# Match your dashboard display and actual GPU service port
GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL", "http://127.0.0.1:5051")

# The model name the gateway expects
LOCAL_MODEL = os.getenv("AI_LOCAL_MODEL", "emtac-gpu-qwen")

MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("AI_TOP_P", "0.95"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "120"))
MAX_HISTORY_MESSAGES = int(os.getenv("AI_MAX_HISTORY_MESSAGES", "12"))

BASE_SYSTEM_PROMPT = """\
You are Status A.I., an intelligent operations assistant embedded in the EMTAC Service Dashboard.

EMTAC runs a service dashboard that manages local infrastructure. The dashboard commonly monitors:
- GPU Service       — local model execution backend
- AI Gateway        — OpenAI-compatible MCP/gateway proxy
- PostgreSQL Server — local database service
- Other managed EMTAC services started by the ServiceManager

You receive a live telemetry snapshot with every query, built directly from the ServiceManager.
Real service names, statuses, PIDs, uptimes, and recent output lines may be included.

Behavior requirements:
- Be concise, technical, and direct.
- Lead summaries with: overall health → key metrics → concerns → actions.
- Use markers:
  ✓ ok
  ⚠ warning
  ✗ error
- Keep responses under 200 words unless the operator asks for more detail.
- Do not invent telemetry that is not present.
- If service data is missing, say so clearly.
"""


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    """
    Accept only OpenAI-style messages:
    [{"role": "...", "content": "..."}]
    """
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
    """
    Keep only the most recent conversation turns.
    We do not preserve client-supplied system messages because this route
    owns the system prompt.
    """
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) <= MAX_HISTORY_MESSAGES:
        return non_system
    return non_system[-MAX_HISTORY_MESSAGES:]


def build_telemetry() -> str:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines: list[str] = [
        f"[EMTAC Live Telemetry — {ts}]",
        f"MCP Gateway : {MCP_GATEWAY_URL}",
        f"GPU Service : {GPU_SERVICE_URL}",
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
        output = svc.get("output", [])

        icon = "✓" if status == "running" else "✗"
        uptime_str = f"{uptime}s" if uptime is not None else "—"
        pid_str = str(pid) if pid else "—"

        lines.append("")
        lines.append(f"  {icon} {name}")
        lines.append(
            f"      type={service_type}  status={status}  "
            f"pid={pid_str}  uptime={uptime_str}"
        )

        if isinstance(output, list) and output:
            lines.append("      recent output:")
            for line in output[-5:]:
                lines.append(f"        {_safe_text(line)}")

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


def inject_telemetry(messages: list[dict[str, str]], telemetry: str) -> list[dict[str, str]]:
    """
    Prepend live telemetry to the last user message only.
    We clone the message dicts so we do not mutate the normalized history in-place.
    """
    cloned = [{"role": m["role"], "content": m["content"]} for m in messages]

    if cloned and cloned[-1]["role"] == "user":
        cloned[-1]["content"] = (
            f"[Live Telemetry]\n{telemetry}\n\n"
            f"[Operator Query]\n{cloned[-1]['content']}"
        )

    return cloned


def call_local(messages: list[dict[str, str]], system_prompt: str) -> tuple[str, str]:
    """
    POST to MCP/OpenAI-compatible gateway.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    dash_info(
        "ai_chat_route: calling MCP gateway | "
        f"endpoint={MCP_GATEWAY_URL} model={LOCAL_MODEL} "
        f"messages={len(full_messages)}"
    )

    response = http_requests.post(
        MCP_GATEWAY_URL,
        json={
            "model": LOCAL_MODEL,
            "messages": full_messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "stream": False,
        },
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()

    reply = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        or "No response received."
    )

    model_used = data.get("model", LOCAL_MODEL)

    dash_info(
        f"ai_chat_route: MCP gateway responded | "
        f"reply_chars={len(reply)} model_used={model_used}"
    )
    return reply, model_used


def _probe_health(url: str, timeout: int = 3) -> tuple[bool, dict[str, Any]]:
    """
    Simple helper for health probing.
    """
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
    """
    Convert:
      http://host:9000/v1/chat/completions
    into:
      http://host:9000/health
    """
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
    Injects live ServiceManager telemetry into every request.
    Returns OpenAI-compatible response shape.
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
        f"messages={len(messages)} telemetry_chars={len(telemetry)}"
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
            "error": f"MCP Gateway timed out ({REQUEST_TIMEOUT_SECONDS}s). GPU may be busy."
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
    """
    gateway_health_url = _guess_gateway_health_url()

    gateway_ok, gateway_status = _probe_health(gateway_health_url, timeout=3)
    gpu_ok, gpu_status = _probe_health(f"{GPU_SERVICE_URL}/health", timeout=3)

    overall = "ok" if (gateway_ok and gpu_ok) else "degraded"

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
        "service_manager": "available" if _service_manager is not None else "unavailable",
    })