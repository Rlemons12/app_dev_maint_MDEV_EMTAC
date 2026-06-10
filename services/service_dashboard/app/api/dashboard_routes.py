from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from flask import Blueprint, jsonify, redirect, render_template, request

from configuration.config import GRAFANA_URL


dashboard_bp = Blueprint("dashboard_bp", __name__)


LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _clean_url(value: str | None, default: str = "http://127.0.0.1:3000") -> str:
    value = (value or "").strip() or default
    return value.rstrip("/")


def _get_request_hostname() -> str:
    host = request.host or ""
    if ":" in host:
        return host.split(":", 1)[0].strip()
    return host.strip()


def _rewrite_loopback_for_browser(base_url: str) -> str:
    parsed = urlparse(_clean_url(base_url))

    scheme = parsed.scheme or "http"
    hostname = parsed.hostname or "127.0.0.1"
    port = parsed.port
    path = parsed.path or ""

    if hostname.lower() in LOOPBACK_HOSTS:
        browser_host = _get_request_hostname()
        if browser_host and browser_host.lower() not in LOOPBACK_HOSTS:
            hostname = browser_host

    netloc = hostname
    if port:
        netloc = f"{hostname}:{port}"

    return urlunparse((scheme, netloc, path.rstrip("/"), "", "", ""))


def _build_public_grafana_url() -> str:
    explicit_public_url = os.getenv("GRAFANA_PUBLIC_URL", "").strip()
    if explicit_public_url:
        return _clean_url(explicit_public_url)

    return _rewrite_loopback_for_browser(_clean_url(GRAFANA_URL))


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []

    for url in urls:
        normalized = _clean_url(url, default="")
        if not normalized:
            continue
        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(normalized)

    return cleaned


def _build_grafana_health_candidates() -> list[str]:
    return _dedupe_urls(
        [
            _clean_url(GRAFANA_URL),
            _clean_url(os.getenv("GRAFANA_PUBLIC_URL", ""), default=""),
            _build_public_grafana_url(),
        ]
    )


def _request_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    req = Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )

    with urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}


def _try_health(base_url: str, path: str = "/health", timeout: float = 3.0) -> dict[str, Any]:
    clean_base = _clean_url(base_url, default="")
    health_url = f"{clean_base}{path}"

    try:
        data = _request_json(health_url, timeout=timeout)

        return {
            "ok": True,
            "base_url": clean_base,
            "health_url": health_url,
            "data": data,
            "error": None,
        }

    except HTTPError as exc:
        return {
            "ok": False,
            "base_url": clean_base,
            "health_url": health_url,
            "data": None,
            "error": f"HTTP {exc.code}",
        }

    except URLError as exc:
        return {
            "ok": False,
            "base_url": clean_base,
            "health_url": health_url,
            "data": None,
            "error": f"Connection error: {exc.reason}",
        }

    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "base_url": clean_base,
            "health_url": health_url,
            "data": None,
            "error": f"Invalid JSON: {exc}",
        }

    except Exception as exc:
        return {
            "ok": False,
            "base_url": clean_base,
            "health_url": health_url,
            "data": None,
            "error": str(exc),
        }


def _grafana_health_payload() -> dict[str, Any]:
    public_url = _build_public_grafana_url()
    internal_url = _clean_url(GRAFANA_URL)
    attempts: list[dict[str, Any]] = []

    for base_url in _build_grafana_health_candidates():
        result = _try_health(base_url, path="/api/health")
        attempts.append(result)

        if result["ok"]:
            return {
                "success": True,
                "service": {
                    "status": "running",
                    "source": "grafana_api_health",
                    "url": public_url,
                    "internal_url": internal_url,
                    "checked_url": result["base_url"],
                    "health_url": result["health_url"],
                },
                "grafana": result["data"],
                "attempts": attempts,
            }

    return {
        "success": False,
        "service": {
            "status": "stopped",
            "source": "grafana_api_health",
            "url": public_url,
            "internal_url": internal_url,
            "checked_url": None,
        },
        "grafana": {
            "error": "Grafana /api/health was not reachable from any configured URL.",
        },
        "attempts": attempts,
    }


def _service_health_payload(
    *,
    service_name: str,
    env_key: str,
    default_url: str,
    health_path: str = "/health",
) -> dict[str, Any]:
    internal_url = _clean_url(os.getenv(env_key, default_url))
    public_url = _rewrite_loopback_for_browser(internal_url)

    result = _try_health(internal_url, path=health_path)

    return {
        "success": result["ok"],
        "service": {
            "name": service_name,
            "status": "running" if result["ok"] else "stopped",
            "source": "direct_health_check",
            "url": public_url,
            "internal_url": internal_url,
            "health_url": result["health_url"],
        },
        "data": result["data"],
        "error": result["error"],
    }


def _build_dashboard_context(active_dashboard: str = "services") -> dict[str, str]:
    return {
        "active_dashboard": active_dashboard,
        "grafana_url": _build_public_grafana_url(),
        "grafana_internal_url": _clean_url(GRAFANA_URL),
        "trace_url": "/dashboards/trace",
        "services_url": "/dashboards/services",
        "gpu_url": "/dashboards/gpu",
        "ai_url": "/dashboards/ai",
        "grafana_dashboard_url": "/dashboards/grafana",
        "grafana_health_url": "/api/grafana-health",
        "ai_health_url": "/api/ai-service-health",
        "gpu_health_url": "/api/gpu-service-health",
        "mcp_health_url": "/api/mcp-coordinator-health",
        "ai_tools_url": "/api/ai/tools",
    }


@dashboard_bp.route("/")
@dashboard_bp.route("/dashboards/services")
def dashboard_services():
    return render_template(
        "service_dashboard.html",
        **_build_dashboard_context("services"),
    )


@dashboard_bp.route("/dashboards/gpu")
def dashboard_gpu():
    return render_template(
        "service_dashboard.html",
        **_build_dashboard_context("gpu"),
    )


@dashboard_bp.route("/dashboards/ai")
def dashboard_ai():
    return render_template(
        "service_dashboard.html",
        **_build_dashboard_context("ai"),
    )


@dashboard_bp.route("/dashboards/trace")
def dashboard_trace():
    return render_template(
        "service_dashboard.html",
        **_build_dashboard_context("trace"),
    )


@dashboard_bp.route("/dashboards/grafana")
def dashboard_grafana():
    return redirect(_build_public_grafana_url(), code=302)


@dashboard_bp.route("/api/grafana-health")
def api_grafana_health():
    return jsonify(_grafana_health_payload())


@dashboard_bp.route("/api/ai-service-health")
def api_ai_service_health():
    return jsonify(
        _service_health_payload(
            service_name="AI Gateway",
            env_key="AI_GATEWAY_URL",
            default_url="http://127.0.0.1:9000",
            health_path="/health",
        )
    )


@dashboard_bp.route("/api/gpu-service-health")
def api_gpu_service_health():
    return jsonify(
        _service_health_payload(
            service_name="GPU Service",
            env_key="GPU_SERVICE_URL",
            default_url="http://127.0.0.1:5051",
            health_path="/health",
        )
    )


@dashboard_bp.route("/api/mcp-coordinator-health")
def api_mcp_coordinator_health():
    return jsonify(
        _service_health_payload(
            service_name="MCP Coordinator",
            env_key="MCP_COORDINATOR_URL",
            default_url="http://127.0.0.1:9100",
            health_path="/health",
        )
    )