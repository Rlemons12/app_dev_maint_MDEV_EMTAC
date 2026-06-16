from __future__ import annotations

import re

from listed_server.mcp_coordinator.routing.models import RouteDecision


POSTGRES_TERMS = {
    "postgres",
    "database",
    "sql",
    "table",
    "schema",
    "qanda",
    "chat",
    "chats",
    "audit",
    "users",
    "login",
    "tablet_user_session",
    "search_audit",
}
GRAFANA_TERMS = {
    "grafana",
    "dashboard",
    "dashboards",
    "panel",
    "panels",
    "datasource",
    "datasources",
    "prometheus",
    "loki",
    "metrics",
    "logs",
    "alert rule",
    "alerts",
    "annotation",
    "annotations",
    "oncall",
    "sift",
    "deeplink",
}
FILESYSTEM_TERMS = {"file", "folder", "directory", "manual", "pdf", "log", "config", "path"}
GIT_TERMS = {"git", "commit", "branch", "diff", "status", "repo"}
BROWSER_TERMS = {"browser", "web ui", "click", "open page", "login page", "button", "screenshot", "webview"}
DANGEROUS_TERMS = {
    "drop",
    "delete",
    "truncate",
    "alter",
    "grant",
    "revoke",
    "create database",
    "drop database",
    "reset password",
    "update password",
    "insert",
    "update",
}


class CoordinatorRouter:
    def route(self, request: str) -> RouteDecision:
        text = request.lower().strip()
        needs_confirmation = any(term in text for term in DANGEROUS_TERMS)

        capability, reason, confidence = self._capability_for_text(text)
        target_tool = self._tool_for(capability, text, needs_confirmation)
        suggested_arguments = self._suggested_arguments(
            request=request,
            capability=capability,
            target_tool=target_tool,
        )

        return RouteDecision(
            target_capability=capability,
            target_tool=target_tool,
            confidence=confidence,
            reason=reason,
            safe_to_execute=capability == "postgres" and not needs_confirmation and target_tool == "postgres_read_query",
            needs_confirmation=needs_confirmation,
            suggested_arguments=suggested_arguments,
        )

    def _capability_for_text(self, text: str) -> tuple[str, str, float]:
        if any(term in text for term in GRAFANA_TERMS):
            return "grafana", "Matched Grafana keywords.", 0.92
        if any(term in text for term in POSTGRES_TERMS):
            return "postgres", "Matched PostgreSQL keywords.", 0.92
        if any(term in text for term in FILESYSTEM_TERMS):
            return "filesystem", "Matched filesystem keywords.", 0.9
        if any(term in text for term in GIT_TERMS):
            return "git", "Matched git keywords.", 0.9
        if any(term in text for term in BROWSER_TERMS):
            return "browser", "Matched browser keywords.", 0.88
        return "memory", "No strong keyword match, defaulting to memory placeholder.", 0.35

    @staticmethod
    def _tool_for(capability: str, text: str, needs_confirmation: bool) -> str | None:
        if capability == "grafana":
            return CoordinatorRouter._grafana_tool_for(text)
        if capability != "postgres":
            return None
        if needs_confirmation:
            if "admin" in text or "create database" in text or "drop database" in text:
                return "postgres_admin_execute"
            return "postgres_write_execute"
        return "postgres_read_query"

    @staticmethod
    def _grafana_tool_for(text: str) -> str | None:
        if "datasource" in text or "datasources" in text:
            return "list_datasources"
        if "folder" in text or "folders" in text:
            return "search_folders"
        if "dashboard" in text or "dashboards" in text:
            if any(term in text for term in {"create", "new", "blank", "update"}):
                return "update_dashboard"
            if "summary" in text:
                return "get_dashboard_summary"
            return "search_dashboards"
        if "prometheus" in text or "metric" in text or "metrics" in text:
            return "query_prometheus"
        if "deeplink" in text or "link" in text:
            return "generate_deeplink"
        if "annotation" in text or "annotations" in text:
            return "get_annotations"
        return None

    @staticmethod
    def _suggested_arguments(
        request: str,
        capability: str,
        target_tool: str | None,
    ) -> dict[str, object]:
        if capability == "postgres" and target_tool == "postgres_read_query":
            return {"sql": request}

        if capability == "grafana":
            if target_tool == "update_dashboard":
                title = _extract_dashboard_title(request)
                uid = _uid_from_title(title)
                return {
                    "dashboard": {
                        "id": None,
                        "uid": uid,
                        "title": title,
                        "tags": ["mcp"],
                        "timezone": "browser",
                        "schemaVersion": 40,
                        "version": 0,
                        "refresh": "",
                        "panels": [],
                        "templating": {"list": []},
                        "annotations": {"list": []},
                    },
                    "overwrite": True,
                    "message": f"Create blank {title} dashboard from MCP UI",
                }
            if target_tool in {"search_dashboards", "search_folders"}:
                return {"query": request}
            if target_tool == "list_datasources":
                return {}
            return {"request": request}

        return {"request": request}


def _extract_dashboard_title(request: str) -> str:
    quoted = re.search(r'["“](.+?)["”]', request)
    if quoted:
        return quoted.group(1).strip() or "MCP"

    named = re.search(
        r"\b(?:called|named|title|titled)\s+(.+?)(?:\s+in\s+grafana|\s+with\b|$)",
        request,
        flags=re.IGNORECASE,
    )
    if named:
        return named.group(1).strip(" .!?\"'") or "MCP"

    return "MCP"


def _uid_from_title(title: str) -> str:
    uid = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return uid or "mcp"
