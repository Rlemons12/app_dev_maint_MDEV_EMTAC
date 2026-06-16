from __future__ import annotations

"""
Dashboard-facing AI tool definitions for the EMTAC MCP AI Layer.

Path:
    E:\\emtac\\services\\mcp_server\\ai_layer\\dashboard_tool_definitions.py

Purpose:
    Defines the OpenAI-compatible tool schemas exposed to the local model
    when the Service Dashboard chat box sends a request to the MCP AI layer.

Important:
    This file defines schemas only.
    It does not execute tools.

Execution should be handled by:
    ai_layer.ai_rest_app
        for the first migration step

Eventually:
    ai_layer tool execution should route into:
        listed_server.mcp_coordinator
        listed_server.mcp_postgres
        listed_server.mcp_grafana
        listed_server.mcp_emtac
"""

from typing import Any


# ---------------------------------------------------------------------
# Dashboard / service tools
# ---------------------------------------------------------------------

SERVICE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_services",
            "description": (
                "Return the current status of all managed EMTAC services, including "
                "service name, service type, running/stopped status, PID, uptime, "
                "base URL, and recent output summary when available."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_logs",
            "description": (
                "Fetch recent output or log lines for a named service. Use this to "
                "diagnose startup failures, import errors, service crashes, or recent "
                "activity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Exact name of the service as shown by list_services, "
                            "for example 'GPU Service', 'AI Gateway', "
                            "'EMTAC MCP Coordinator', 'PostgreSQL Server', or 'Grafana'."
                        ),
                    },
                    "lines": {
                        "type": "integer",
                        "description": (
                            "Number of most-recent output lines to return. "
                            "Default is 50. Maximum should be treated as 200."
                        ),
                        "default": 50,
                    },
                },
                "required": ["name"],
            },
        },
    },
]


# ---------------------------------------------------------------------
# Health / connectivity tools
# ---------------------------------------------------------------------

HEALTH_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "mcp_coordinator_status",
            "description": (
                "Check whether the EMTAC MCP Coordinator is reachable. "
                "Use this when diagnosing MCP server startup, tool routing, or "
                "AI-to-MCP connectivity."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ai_gateway_health",
            "description": (
                "Check whether the local AI Gateway is reachable and healthy. "
                "The AI Gateway is the OpenAI-compatible local model endpoint "
                "used by the MCP AI layer."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gpu_service_health",
            "description": (
                "Check whether the local GPU service is reachable. "
                "The GPU service owns local model execution and hardware/model runtime state."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------
# PostgreSQL tools
# ---------------------------------------------------------------------

POSTGRES_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_postgres_insights",
            "description": (
                "Fetch a high-level PostgreSQL health and status summary. "
                "Use this for database service health, database size/version, "
                "connection counts, and table overview."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_postgres_tables",
            "description": (
                "List PostgreSQL tables in the configured schema, optionally filtered "
                "by a case-insensitive table-name substring. Use this before querying "
                "unknown database structures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional substring filter on table name. "
                            "Use an empty string to return the available table list."
                        ),
                        "default": "",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_postgres_table",
            "description": (
                "Return column definitions, indexes, and foreign keys for a named "
                "PostgreSQL table. Use this before writing a SELECT query against "
                "an unfamiliar table."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": (
                            "Exact table name in the configured PostgreSQL schema."
                        ),
                    },
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_postgres",
            "description": (
                "Execute a read-only SELECT query against PostgreSQL and return rows "
                "up to the configured row cap. Only SELECT statements are allowed. "
                "Never use this for INSERT, UPDATE, DELETE, ALTER, DROP, CREATE, "
                "TRUNCATE, GRANT, or other write/schema operations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "A single read-only SELECT statement. Must begin with SELECT "
                            "after stripping whitespace."
                        ),
                    },
                },
                "required": ["sql"],
            },
        },
    },
]


# ---------------------------------------------------------------------
# Grafana tools
# ---------------------------------------------------------------------

GRAFANA_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_grafana_health",
            "description": (
                "Fetch Grafana health information such as database status, version, "
                "and basic API reachability."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_grafana_dashboards",
            "description": (
                "List Grafana dashboards. Use this to answer questions like "
                "'what dashboards do we have?' or to find a dashboard by title."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional dashboard-title substring filter.",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of dashboards to return.",
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
                "List configured Grafana datasources such as PostgreSQL, Prometheus, "
                "Loki, or other configured sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grafana_alert_rules",
            "description": (
                "Return configured Grafana alert rules. Use this to answer questions "
                "about active observability alerting configuration."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------
# GPU tools
# ---------------------------------------------------------------------

GPU_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_gpu_insights",
            "description": (
                "Fetch detailed GPU service insights, including GPU hardware metrics, "
                "VRAM use, model/runtime status, and GPU service process information "
                "when available."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------
# Service control tools
# ---------------------------------------------------------------------

SERVICE_CONTROL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "start_service",
            "description": (
                "Start a stopped managed EMTAC service by exact service name. "
                "After using this tool, verify the result with list_services or "
                "get_service_logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the service to start.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_service",
            "description": (
                "Stop a running managed EMTAC service by exact service name. "
                "After using this tool, verify the result with list_services."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the service to stop.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": (
                "Restart a managed EMTAC service by exact service name. "
                "Use this when logs or status suggest a service is stuck or unhealthy. "
                "After using this tool, verify the result with list_services or logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the service to restart.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_service_logs",
            "description": (
                "Clear the in-memory output buffer for a named service. "
                "This does not delete persisted log files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the service whose output buffer to clear.",
                    },
                },
                "required": ["name"],
            },
        },
    },
]


# ---------------------------------------------------------------------
# Combined tool list
# ---------------------------------------------------------------------

DASHBOARD_TOOLS: list[dict[str, Any]] = (
    HEALTH_TOOLS
    + SERVICE_TOOLS
    + GPU_TOOLS
    + POSTGRES_TOOLS
    + GRAFANA_TOOLS
    + SERVICE_CONTROL_TOOLS
)


# Backward-compatible alias for older code that expects TOOLS.
TOOLS: list[dict[str, Any]] = DASHBOARD_TOOLS


def get_dashboard_tools(*, include_control_tools: bool = True) -> list[dict[str, Any]]:
    """
    Return dashboard-facing tool schemas.

    Args:
        include_control_tools:
            If False, excludes start_service, stop_service, restart_service,
            and clear_service_logs.

    This is useful if the AI layer should run in read-only mode.
    """

    if include_control_tools:
        return DASHBOARD_TOOLS

    return (
        HEALTH_TOOLS
        + SERVICE_TOOLS
        + GPU_TOOLS
        + POSTGRES_TOOLS
        + GRAFANA_TOOLS
    )


def get_tool_names(*, include_control_tools: bool = True) -> list[str]:
    """
    Return the registered tool names in order.
    """

    return [
        tool["function"]["name"]
        for tool in get_dashboard_tools(include_control_tools=include_control_tools)
    ]