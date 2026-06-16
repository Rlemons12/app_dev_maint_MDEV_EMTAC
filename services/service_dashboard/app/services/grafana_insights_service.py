from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from configuration.config import (
    GRAFANA_URL,
    GRAFANA_API_KEY,
    GRAFANA_USER,
    GRAFANA_PASSWORD,
)
from app.services.service_dashboard_logger import (
    dash_debug,
    dash_error,
    dash_info,
    dash_warning,
)


def _auth_headers() -> Dict[str, str]:
    """
    Prefer API token; fall back to basic auth if user/pass are configured.
    /api/health works without auth, but /api/search, /api/datasources,
    and /api/v1/provisioning/* all require it.
    """
    headers = {"Accept": "application/json"}
    if GRAFANA_API_KEY:
        headers["Authorization"] = f"Bearer {GRAFANA_API_KEY}"
    elif GRAFANA_USER and GRAFANA_PASSWORD:
        token = base64.b64encode(
            f"{GRAFANA_USER}:{GRAFANA_PASSWORD}".encode("utf-8")
        ).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    return headers


def _grafana_get(path: str, timeout: float = 3.0) -> Dict[str, Any]:
    url = f"{GRAFANA_URL.rstrip('/')}{path}"
    dash_debug(f"Grafana GET url='{url}'")
    try:
        req = Request(url, headers=_auth_headers())
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return {"ok": True, "data": json.loads(raw)}
    except HTTPError as exc:
        dash_warning(f"Grafana HTTP error url='{url}' status={exc.code}")
        return {"ok": False, "error": f"HTTP {exc.code} from {path}"}
    except URLError as exc:
        dash_warning(f"Grafana connection error url='{url}' reason={exc.reason}")
        return {"ok": False, "error": f"Connection error: {exc.reason}"}
    except json.JSONDecodeError as exc:
        dash_error(f"Grafana invalid JSON url='{url}' error={exc}")
        return {"ok": False, "error": f"Invalid JSON: {exc}"}
    except Exception as exc:
        dash_error(f"Grafana unexpected error url='{url}' error={exc}")
        return {"ok": False, "error": str(exc)}


def _snapshot(grafana_service: Any) -> Dict[str, Any]:
    if not grafana_service:
        return {"status": "unregistered", "url": GRAFANA_URL}
    return {
        "status": grafana_service.get_status(),
        "pid": grafana_service.get_pid(),
        "uptime_seconds": grafana_service.get_uptime_seconds(),
        "url": GRAFANA_URL,
    }


def _guard(grafana_service: Any) -> Optional[Dict[str, Any]]:
    """Return an error payload if Grafana isn't usable, else None."""
    if not grafana_service:
        return {"success": False, "service": _snapshot(None),
                "error": "Grafana is not registered."}
    if grafana_service.get_status() != "running":
        return {"success": False, "service": _snapshot(grafana_service),
                "error": "Grafana is not running."}
    return None


def get_grafana_health(grafana_service: Any) -> Dict[str, Any]:
    dash_info("Building Grafana health payload")
    guard = _guard(grafana_service)
    if guard:
        return guard

    result = _grafana_get("/api/health")
    return {
        "success": result["ok"],
        "service": _snapshot(grafana_service),
        "grafana": result.get("data") if result["ok"] else {"error": result["error"]},
    }


def list_grafana_dashboards(
    grafana_service: Any,
    query: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    dash_info(f"Listing Grafana dashboards query='{query}' limit={limit}")
    guard = _guard(grafana_service)
    if guard:
        return guard

    path = f"/api/search?type=dash-db&limit={max(1, min(limit, 200))}"
    if query:
        path += f"&query={query}"

    result = _grafana_get(path)
    if not result["ok"]:
        return {"success": False, "service": _snapshot(grafana_service),
                "error": result["error"]}

    items = result["data"] or []
    return {
        "success": True,
        "service": _snapshot(grafana_service),
        "count": len(items),
        "dashboards": [
            {
                "uid": it.get("uid"),
                "title": it.get("title"),
                "folder": it.get("folderTitle"),
                "url": it.get("url"),
                "tags": it.get("tags", []),
            }
            for it in items
        ],
    }


def list_grafana_datasources(grafana_service: Any) -> Dict[str, Any]:
    dash_info("Listing Grafana datasources")
    guard = _guard(grafana_service)
    if guard:
        return guard

    result = _grafana_get("/api/datasources")
    if not result["ok"]:
        return {
            "success": False,
            "service": _snapshot(grafana_service),
            "error": result["error"],
            "hint": (
                "Datasource listing requires admin auth. Set GRAFANA_API_KEY "
                "in your .env, or GRAFANA_USER / GRAFANA_PASSWORD."
            ),
        }

    items = result["data"] or []
    return {
        "success": True,
        "service": _snapshot(grafana_service),
        "count": len(items),
        "datasources": [
            {
                "uid": it.get("uid"),
                "name": it.get("name"),
                "type": it.get("type"),
                "url": it.get("url"),
                "is_default": it.get("isDefault"),
            }
            for it in items
        ],
    }


def get_grafana_alert_rules(grafana_service: Any) -> Dict[str, Any]:
    dash_info("Fetching Grafana alert rules")
    guard = _guard(grafana_service)
    if guard:
        return guard

    result = _grafana_get("/api/v1/provisioning/alert-rules")
    if not result["ok"]:
        return {"success": False, "service": _snapshot(grafana_service),
                "error": result["error"]}

    items = result["data"] or []
    return {
        "success": True,
        "service": _snapshot(grafana_service),
        "count": len(items),
        "alert_rules": [
            {
                "uid": it.get("uid"),
                "title": it.get("title"),
                "folder_uid": it.get("folderUID"),
                "rule_group": it.get("ruleGroup"),
                "condition": it.get("condition"),
            }
            for it in items
        ],
    }