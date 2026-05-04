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


def _build_public_grafana_url() -> str:
    """
    Build the browser-facing Grafana URL.

    Priority:
    1. GRAFANA_PUBLIC_URL from .env
    2. GRAFANA_URL rewritten from localhost to the current Flask host/IP
    """
    explicit_public_url = os.getenv("GRAFANA_PUBLIC_URL", "").strip()
    if explicit_public_url:
        return _clean_url(explicit_public_url)

    configured = _clean_url(GRAFANA_URL)
    parsed = urlparse(configured)

    scheme = parsed.scheme or "http"
    hostname = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3000
    path = parsed.path or ""

    if hostname.lower() in LOOPBACK_HOSTS:
        browser_host = _get_request_hostname()
        if browser_host and browser_host.lower() not in LOOPBACK_HOSTS:
            hostname = browser_host

    netloc = f"{hostname}:{port}"
    return urlunparse((scheme, netloc, path.rstrip("/"), "", "", ""))


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []

    for url in urls:
        normalized = _clean_url(url)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)

    return cleaned


def _build_grafana_health_candidates() -> list[str]:
    """
    Build every Grafana base URL worth testing.

    This fixes the case where:
    - browser link works through public/server IP
    - internal GRAFANA_URL health check fails
    """
    candidates = [
        _clean_url(GRAFANA_URL),
        _clean_url(os.getenv("GRAFANA_PUBLIC_URL", "")),
        _build_public_grafana_url(),
    ]

    return _dedupe_urls(candidates)


def _try_grafana_health(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    health_url = f"{_clean_url(base_url)}/api/health"

    try:
        req = Request(
            health_url,
            headers={"Accept": "application/json"},
            method="GET",
        )

        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}

        return {
            "ok": True,
            "base_url": _clean_url(base_url),
            "health_url": health_url,
            "data": data,
            "error": None,
        }

    except HTTPError as exc:
        return {
            "ok": False,
            "base_url": _clean_url(base_url),
            "health_url": health_url,
            "data": None,
            "error": f"HTTP {exc.code}",
        }

    except URLError as exc:
        return {
            "ok": False,
            "base_url": _clean_url(base_url),
            "health_url": health_url,
            "data": None,
            "error": f"Connection error: {exc.reason}",
        }

    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "base_url": _clean_url(base_url),
            "health_url": health_url,
            "data": None,
            "error": f"Invalid JSON: {exc}",
        }

    except Exception as exc:
        return {
            "ok": False,
            "base_url": _clean_url(base_url),
            "health_url": health_url,
            "data": None,
            "error": str(exc),
        }


def _grafana_health_payload() -> dict[str, Any]:
    """
    Real Grafana health check.

    It does not trust ServiceManager status.
    It tests actual Grafana /api/health reachability.
    """
    public_url = _build_public_grafana_url()
    internal_url = _clean_url(GRAFANA_URL)
    candidates = _build_grafana_health_candidates()

    attempts: list[dict[str, Any]] = []

    for base_url in candidates:
        result = _try_grafana_health(base_url)
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


def _build_dashboard_context() -> dict[str, str]:
    public_grafana_url = _build_public_grafana_url()

    return {
        "grafana_url": public_grafana_url,
        "grafana_internal_url": _clean_url(GRAFANA_URL),
        "trace_url": "/dashboards/trace",
        "services_url": "/dashboards/services",
        "gpu_url": "/dashboards/gpu",
        "ai_url": "/dashboards/ai",
        "grafana_dashboard_url": "/dashboards/grafana",
    }


@dashboard_bp.route("/")
@dashboard_bp.route("/dashboards/services")
def dashboard_services():
    context = _build_dashboard_context()
    return render_template("service_dashboard.html", **context)


@dashboard_bp.route("/dashboards/gpu")
def dashboard_gpu():
    context = _build_dashboard_context()
    return render_template("service_dashboard.html", **context)


@dashboard_bp.route("/dashboards/ai")
def dashboard_ai():
    context = _build_dashboard_context()
    return render_template("service_dashboard.html", **context)


@dashboard_bp.route("/dashboards/trace")
def dashboard_trace():
    context = _build_dashboard_context()
    return render_template("service_dashboard.html", **context)


@dashboard_bp.route("/dashboards/grafana")
def dashboard_grafana():
    return redirect(_build_public_grafana_url(), code=302)


@dashboard_bp.route("/api/grafana-health")
def api_grafana_health():
    return jsonify(_grafana_health_payload())