from __future__ import annotations

import re
from typing import Any

from listed_server.mcp_coordinator.routing.models import RouteDecision


# ============================================================
# Capability keyword groups
# ============================================================

POSTGRES_TERMS = {
    "postgres",
    "postgresql",
    "database",
    "databases",
    "sql",
    "query",
    "table",
    "tables",
    "schema",
    "schemas",
    "column",
    "columns",
    "index",
    "indexes",
    "constraint",
    "constraints",
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
    "metric",
    "logs",
    "alert rule",
    "alerts",
    "alert",
    "annotation",
    "annotations",
    "oncall",
    "sift",
    "deeplink",
}

FILESYSTEM_TERMS = {
    "file",
    "files",
    "folder",
    "folders",
    "directory",
    "directories",
    "manual",
    "pdf",
    "log file",
    "config",
    "path",
}

GIT_TERMS = {
    "git",
    "commit",
    "commits",
    "branch",
    "branches",
    "diff",
    "status",
    "repo",
    "repository",
}

BROWSER_TERMS = {
    "browser",
    "web ui",
    "click",
    "open page",
    "login page",
    "button",
    "screenshot",
    "webview",
}

# Natural-language and SQL terms that should require confirmation.
DANGEROUS_TERMS = {
    "drop",
    "delete",
    "truncate",
    "alter",
    "grant",
    "revoke",
    "create database",
    "drop database",
    "drop schema",
    "drop table",
    "reset password",
    "update password",
    "insert",
    "update",
}

READ_ONLY_SQL_START_RE = re.compile(
    r"^\s*(select|with|show|explain)\b",
    flags=re.IGNORECASE,
)

WRITE_SQL_START_RE = re.compile(
    r"^\s*(insert|update|delete)\b",
    flags=re.IGNORECASE,
)

ADMIN_SQL_START_RE = re.compile(
    r"^\s*(create|drop|alter|grant|revoke|truncate)\b",
    flags=re.IGNORECASE,
)

POSTGRES_SAFE_TO_EXECUTE_TOOLS = {
    "postgres_health_check",
    "postgres_whoami",
    "postgres_list_databases",
    "postgres_list_schemas",
    "postgres_list_tables",
    "postgres_describe_table",
    "postgres_read_query",
}

TOOLS_REQUIRING_CONFIRMATION = {
    "postgres_create_database",
    "postgres_drop_database",
    "postgres_create_schema",
    "postgres_drop_schema",
    "postgres_create_table",
    "postgres_drop_table",
    "postgres_grant_standard_database_permissions",
    "postgres_write_execute",
    "postgres_admin_execute",
    "postgres_insert_row",
    "postgres_update_rows",
    "postgres_delete_rows",
    "update_dashboard",
}


# ============================================================
# Router
# ============================================================

class CoordinatorRouter:
    def route(self, request: str) -> RouteDecision:
        original_request = request or ""
        text = original_request.lower().strip()

        capability, reason, confidence = self._capability_for_text(text)
        target_tool = self._tool_for(
            capability=capability,
            request=original_request,
            text=text,
        )

        needs_confirmation = self._needs_confirmation(
            text=text,
            target_tool=target_tool,
        )

        suggested_arguments = self._suggested_arguments(
            request=original_request,
            text=text,
            capability=capability,
            target_tool=target_tool,
        )

        safe_to_execute = self._safe_to_execute(
            capability=capability,
            target_tool=target_tool,
            needs_confirmation=needs_confirmation,
        )

        return RouteDecision(
            target_capability=capability,
            target_tool=target_tool,
            confidence=confidence,
            reason=reason,
            safe_to_execute=safe_to_execute,
            needs_confirmation=needs_confirmation,
            suggested_arguments=suggested_arguments,
        )

    # --------------------------------------------------------
    # Capability selection
    # --------------------------------------------------------

    def _capability_for_text(self, text: str) -> tuple[str, str, float]:
        if self._looks_like_sql(text):
            return "postgres", "Request starts with SQL-like syntax.", 0.96

        if any(term in text for term in GRAFANA_TERMS):
            return "grafana", "Matched Grafana keywords.", 0.92

        if any(term in text for term in POSTGRES_TERMS):
            return "postgres", "Matched PostgreSQL keywords.", 0.92

        if any(term in text for term in FILESYSTEM_TERMS):
            return "filesystem", "Matched filesystem keywords.", 0.90

        if any(term in text for term in GIT_TERMS):
            return "git", "Matched git keywords.", 0.90

        if any(term in text for term in BROWSER_TERMS):
            return "browser", "Matched browser keywords.", 0.88

        return "memory", "No strong keyword match, defaulting to memory placeholder.", 0.35

    # --------------------------------------------------------
    # Tool selection
    # --------------------------------------------------------

    def _tool_for(
        self,
        *,
        capability: str,
        request: str,
        text: str,
    ) -> str | None:
        if capability == "grafana":
            return self._grafana_tool_for(text)

        if capability != "postgres":
            return None

        return self._postgres_tool_for(request=request, text=text)

    def _postgres_tool_for(self, *, request: str, text: str) -> str | None:
        # Health / role inspection
        if "health" in text or "connection" in text or "connectivity" in text:
            return "postgres_health_check"

        if "whoami" in text or "who am i" in text or "current user" in text:
            return "postgres_whoami"

        # Cluster metadata
        if "list database" in text or "list databases" in text or "show databases" in text:
            return "postgres_list_databases"

        # Schema metadata
        if "list schema" in text or "list schemas" in text or "show schemas" in text:
            return "postgres_list_schemas"

        # Table metadata
        if (
            "list table" in text
            or "list tables" in text
            or "show tables" in text
            or "list views" in text
            or "show views" in text
        ):
            return "postgres_list_tables"

        if (
            "describe table" in text
            or "describe the table" in text
            or "table detail" in text
            or "table details" in text
            or "show columns" in text
            or "list columns" in text
            or "columns for" in text
            or "indexes for" in text
            or "constraints for" in text
        ):
            table_name = _extract_table_name(request)
            if table_name:
                return "postgres_describe_table"
            return None

        # Structured database/schema operations when the name is obvious
        if "create database" in text:
            database_name = _extract_named_identifier(
                request,
                patterns=[
                    r"\bcreate\s+database\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                    r"\bdatabase\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                ],
            )
            if database_name:
                return "postgres_create_database"

        if "drop database" in text:
            database_name = _extract_named_identifier(
                request,
                patterns=[
                    r"\bdrop\s+database\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                    r"\bdatabase\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                ],
            )
            if database_name:
                return "postgres_drop_database"

        if "create schema" in text:
            schema_name = _extract_named_identifier(
                request,
                patterns=[
                    r"\bcreate\s+schema\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                    r"\bschema\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                ],
            )
            if schema_name:
                return "postgres_create_schema"

        if "drop schema" in text:
            schema_name = _extract_named_identifier(
                request,
                patterns=[
                    r"\bdrop\s+schema\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                    r"\bschema\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                ],
            )
            if schema_name:
                return "postgres_drop_schema"

        # Raw SQL routing
        if READ_ONLY_SQL_START_RE.search(request):
            return "postgres_read_query"

        if WRITE_SQL_START_RE.search(request):
            return "postgres_write_execute"

        if ADMIN_SQL_START_RE.search(request):
            return "postgres_admin_execute"

        # Do not send natural language into postgres_read_query.
        # The AI layer should convert natural language into SQL first.
        return None

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

    # --------------------------------------------------------
    # Confirmation / safety
    # --------------------------------------------------------

    @staticmethod
    def _needs_confirmation(text: str, target_tool: str | None) -> bool:
        if target_tool in TOOLS_REQUIRING_CONFIRMATION:
            return True

        if any(term in text for term in DANGEROUS_TERMS):
            return True

        return False

    @staticmethod
    def _safe_to_execute(
        *,
        capability: str,
        target_tool: str | None,
        needs_confirmation: bool,
    ) -> bool:
        if needs_confirmation:
            return False

        if capability == "postgres" and target_tool in POSTGRES_SAFE_TO_EXECUTE_TOOLS:
            return True

        return False

    # --------------------------------------------------------
    # Suggested arguments
    # --------------------------------------------------------

    def _suggested_arguments(
        self,
        *,
        request: str,
        text: str,
        capability: str,
        target_tool: str | None,
    ) -> dict[str, object]:
        if capability == "postgres":
            return self._postgres_arguments(
                request=request,
                text=text,
                target_tool=target_tool,
            )

        if capability == "grafana":
            return self._grafana_arguments(
                request=request,
                target_tool=target_tool,
            )

        return {"request": request}

    @staticmethod
    def _postgres_arguments(
        *,
        request: str,
        text: str,
        target_tool: str | None,
    ) -> dict[str, object]:
        if not target_tool:
            return {
                "request": request,
                "message": (
                    "No concrete PostgreSQL tool was selected. "
                    "Use direct SQL or ask for a supported metadata action such as "
                    "'list tables', 'list schemas', or 'describe table <name>'."
                ),
            }

        if target_tool == "postgres_health_check":
            database_name = _extract_database_name(request)
            return _remove_none_values(
                {
                    "database_name": database_name,
                }
            )

        if target_tool == "postgres_whoami":
            database_name = _extract_database_name(request)
            return _remove_none_values(
                {
                    "database_name": database_name,
                }
            )

        if target_tool == "postgres_list_databases":
            include_templates = (
                "template" in text
                or "templates" in text
                or "system database" in text
                or "system databases" in text
            )
            return {
                "include_templates": include_templates,
            }

        if target_tool == "postgres_list_schemas":
            database_name = _extract_database_name(request)
            include_system_schemas = (
                "system schema" in text
                or "system schemas" in text
                or "pg_catalog" in text
                or "information_schema" in text
            )
            return _remove_none_values(
                {
                    "database_name": database_name,
                    "include_system_schemas": include_system_schemas,
                }
            )

        if target_tool == "postgres_list_tables":
            database_name = _extract_database_name(request)
            schema_name = _extract_schema_name(request)

            include_views = not (
                "tables only" in text
                or "no views" in text
                or "exclude views" in text
            )

            return _remove_none_values(
                {
                    "database_name": database_name,
                    "schema_name": schema_name,
                    "include_views": include_views,
                }
            )

        if target_tool == "postgres_describe_table":
            table_name = _extract_table_name(request)
            database_name = _extract_database_name(request)
            schema_name = _extract_schema_name(request)

            return _remove_none_values(
                {
                    "table_name": table_name,
                    "database_name": database_name,
                    "schema_name": schema_name,
                }
            )

        if target_tool == "postgres_create_database":
            database_name = _extract_named_identifier(
                request,
                patterns=[
                    r"\bcreate\s+database\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                    r"\bdatabase\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                ],
            )

            return _remove_none_values(
                {
                    "database_name": database_name,
                    "grant_standard_permissions": True,
                }
            )

        if target_tool == "postgres_drop_database":
            database_name = _extract_named_identifier(
                request,
                patterns=[
                    r"\bdrop\s+database\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                    r"\bdatabase\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                ],
            )

            return _remove_none_values(
                {
                    "database_name": database_name,
                    "force": "force" in text,
                    "allow_system_database": False,
                }
            )

        if target_tool == "postgres_create_schema":
            schema_name = _extract_named_identifier(
                request,
                patterns=[
                    r"\bcreate\s+schema\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                    r"\bschema\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                ],
            )
            database_name = _extract_database_name(request)

            return _remove_none_values(
                {
                    "schema_name": schema_name,
                    "database_name": database_name,
                    "if_not_exists": True,
                    "grant_standard_permissions": True,
                }
            )

        if target_tool == "postgres_drop_schema":
            schema_name = _extract_named_identifier(
                request,
                patterns=[
                    r"\bdrop\s+schema\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                    r"\bschema\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                ],
            )
            database_name = _extract_database_name(request)

            return _remove_none_values(
                {
                    "schema_name": schema_name,
                    "database_name": database_name,
                    "cascade": "cascade" in text,
                    "allow_public_schema": False,
                }
            )

        if target_tool in {
            "postgres_read_query",
            "postgres_write_execute",
            "postgres_admin_execute",
        }:
            database_name = _extract_database_name(request)
            args: dict[str, object] = {
                "sql": request,
            }

            if database_name:
                args["database_name"] = database_name

            if target_tool == "postgres_admin_execute":
                args["autocommit"] = True

            return args

        return {"request": request}

    @staticmethod
    def _grafana_arguments(
        *,
        request: str,
        target_tool: str | None,
    ) -> dict[str, object]:
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

    # --------------------------------------------------------
    # SQL detection
    # --------------------------------------------------------

    @staticmethod
    def _looks_like_sql(text: str) -> bool:
        return bool(
            READ_ONLY_SQL_START_RE.search(text)
            or WRITE_SQL_START_RE.search(text)
            or ADMIN_SQL_START_RE.search(text)
        )


# ============================================================
# Extraction helpers
# ============================================================

def _extract_dashboard_title(request: str) -> str:
    quoted = re.search(r'["“”](.+?)["“”]', request)
    if quoted:
        return quoted.group(1).strip() or "MCP"

    named = re.search(
        r"\b(?:called|named|title|titled)\s+(.+?)(?:\s+in\s+grafana|\s+with\b|$)",
        request,
        flags=re.IGNORECASE,
    )
    if named:
        return named.group(1).strip(" .!?\"'“”") or "MCP"

    return "MCP"


def _uid_from_title(title: str) -> str:
    uid = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return uid or "mcp"


def _extract_database_name(request: str) -> str | None:
    return _extract_named_identifier(
        request,
        patterns=[
            r"\bdatabase\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
            r"\bdb\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
            r"\bin\s+database\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bfrom\s+database\s+([A-Za-z_][A-Za-z0-9_]*)",
        ],
    )


def _extract_schema_name(request: str) -> str | None:
    return _extract_named_identifier(
        request,
        patterns=[
            r"\bschema\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
            r"\bin\s+schema\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bfrom\s+schema\s+([A-Za-z_][A-Za-z0-9_]*)",
        ],
    )


def _extract_table_name(request: str) -> str | None:
    return _extract_named_identifier(
        request,
        patterns=[
            r"\bdescribe\s+(?:the\s+)?table\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\btable\s+(?:called\s+|named\s+)?([A-Za-z_][A-Za-z0-9_]*)",
            r"\bcolumns\s+(?:for|in|of)\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bindexes\s+(?:for|in|of)\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bconstraints\s+(?:for|in|of)\s+([A-Za-z_][A-Za-z0-9_]*)",
        ],
    )


def _extract_named_identifier(
    request: str,
    *,
    patterns: list[str],
) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, request, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" .!?\"'“”")
            if _is_identifier(value):
                return value

    return None


def _is_identifier(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$", value or ""))


def _remove_none_values(values: dict[str, object | None]) -> dict[str, object]:
    return {
        key: value
        for key, value in values.items()
        if value is not None
    }