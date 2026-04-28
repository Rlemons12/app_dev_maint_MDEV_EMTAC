from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.services.service_dashboard_logger import (
    dash_debug,
    dash_error,
    dash_info,
    dash_warning,
)


# ---------------------------------------------------------
# Normalized snapshot of the managed service object
# ---------------------------------------------------------

@dataclass
class PostgresServiceSnapshot:
    status: str
    pid: Optional[int]
    uptime_seconds: Optional[int]
    cwd: Optional[str]
    command: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "pid": self.pid,
            "uptime_seconds": self.uptime_seconds,
            "cwd": self.cwd,
            "command": self.command,
        }


def build_service_snapshot(pg_service: Any) -> PostgresServiceSnapshot:
    """
    Build a normalized snapshot of the managed PostgreSQL service object
    from ServiceManager.get_service("PostgreSQL Server").

    PostgreSQL is registered as service_type='command' so its primary
    command is .start_command, not .command.
    """
    start_cmd = getattr(pg_service, "start_command", None) or []
    snapshot = PostgresServiceSnapshot(
        status=pg_service.get_status(),
        pid=pg_service.get_pid(),
        uptime_seconds=pg_service.get_uptime_seconds(),
        cwd=getattr(pg_service, "cwd", None),
        command=" ".join(start_cmd) if start_cmd else "",
    )
    dash_debug(
        "Built Postgres service snapshot "
        f"status='{snapshot.status}' pid='{snapshot.pid}' uptime='{snapshot.uptime_seconds}'"
    )
    return snapshot


# ---------------------------------------------------------
# Lazy client access
# ---------------------------------------------------------

def _get_client() -> Any:
    """
    Lazily import the module-level PostgresServiceClient singleton so this
    module doesn't hard-depend on psycopg2 at import time. If the client
    module is missing or psycopg2 isn't installed, return None and callers
    degrade gracefully.

    Adjust the import path below if you place postgres_service_client.py
    somewhere other than app/services/.
    """
    try:
        from app.services.postgres_service_client import postgres_service_client
        return postgres_service_client
    except Exception as exc:
        dash_warning(f"Postgres client unavailable: {exc}")
        return None


# ---------------------------------------------------------
# DB-level queries (read-only, routed through execute_query)
# ---------------------------------------------------------

def _db_overview(client: Any) -> Dict[str, Any]:
    """Database name, size, server version, server uptime."""
    try:
        rows = client.execute_query(
            """
            SELECT
                current_database()                                     AS name,
                pg_size_pretty(pg_database_size(current_database()))   AS size,
                pg_database_size(current_database())                   AS size_bytes,
                version()                                              AS version,
                EXTRACT(EPOCH FROM (now() - pg_postmaster_start_time()))::bigint
                                                                       AS uptime_seconds
            """
        )
        return dict(rows[0]) if rows else {"error": "no rows returned"}
    except Exception as exc:
        dash_warning(f"Postgres _db_overview failed: {exc}")
        return {"error": str(exc)}


def _connections_overview(client: Any) -> Dict[str, Any]:
    """Active connections broken down by state, plus max_connections."""
    try:
        state_rows = client.execute_query(
            """
            SELECT
                COALESCE(state, 'unknown') AS state,
                count(*)::int              AS n
            FROM pg_stat_activity
            WHERE datname = current_database()
            GROUP BY state
            """
        )
        by_state = {r["state"]: r["n"] for r in state_rows}
        total = sum(by_state.values())

        max_rows = client.execute_query(
            """
            SELECT setting::int AS max_connections
            FROM pg_settings
            WHERE name = 'max_connections'
            """
        )
        max_conn = max_rows[0]["max_connections"] if max_rows else None

        return {
            "total": total,
            "active": by_state.get("active", 0),
            "idle": by_state.get("idle", 0),
            "idle_in_transaction": by_state.get("idle in transaction", 0),
            "by_state": by_state,
            "max_connections": max_conn,
        }
    except Exception as exc:
        dash_warning(f"Postgres _connections_overview failed: {exc}")
        return {"error": str(exc)}


# ---------------------------------------------------------
# Main insights payload
# ---------------------------------------------------------

def get_postgres_service_insights(
    pg_service: Any,
    *,
    include_tables: bool = True,
    table_query: str = "",
) -> Dict[str, Any]:
    """
    Return a single structured payload describing the PostgreSQL service
    for the dashboard. Mirrors get_gpu_service_insights() in shape.

    Structure:
        {
          "success": bool,
          "service":     {status, pid, uptime_seconds, cwd, command},
          "health":      {reachable: bool, error?: str},
          "database":    {name, size, size_bytes, version, uptime_seconds} | {error},
          "connections": {total, active, idle, idle_in_transaction, by_state, max_connections} | {error},
          "tables":      [ {name, schema, row_estimate, comment}, ... ],
          "host":        "host:port",
          "schema":      "public",
        }
    """
    dash_info("Building Postgres service insights payload")

    if not pg_service:
        dash_warning("Postgres insights requested but PostgreSQL Server is not registered")
        not_registered = {"error": "PostgreSQL Server is not registered."}
        return {
            "success": False,
            "message": "PostgreSQL Server is not registered.",
            "service": None,
            "health": not_registered,
            "database": not_registered,
            "connections": not_registered,
            "tables": [],
        }

    snapshot = build_service_snapshot(pg_service).to_dict()

    if pg_service.get_status() != "running":
        dash_info("Postgres insights requested while PostgreSQL Server is not running")
        not_running = {"error": "PostgreSQL Server is not running."}
        return {
            "success": True,
            "service": snapshot,
            "health": {"reachable": False, "error": "PostgreSQL Server is not running."},
            "database": not_running,
            "connections": not_running,
            "tables": [],
        }

    client = _get_client()
    if client is None:
        err = {"error": "Postgres client not available (is psycopg2 installed?)."}
        return {
            "success": True,
            "service": snapshot,
            "health": {"reachable": False, "error": err["error"]},
            "database": err,
            "connections": err,
            "tables": [],
        }

    reachable = client.health()
    if not reachable:
        dash_warning("Postgres health check failed while process appears to be running")
        return {
            "success": True,
            "service": snapshot,
            "health": {"reachable": False, "error": "Health check failed (connection refused or auth error)."},
            "database": {"error": "Unreachable"},
            "connections": {"error": "Unreachable"},
            "tables": [],
            "host": f"{client.host}:{client.port}",
            "schema": client.schema,
        }

    db = _db_overview(client)
    conn = _connections_overview(client)
    tables: List[Dict[str, Any]] = client.get_tables(query=table_query) if include_tables else []

    dash_info(
        "Postgres insights payload built successfully "
        f"tables={len(tables)} connections_total={conn.get('total')}"
    )

    return {
        "success": True,
        "service": snapshot,
        "health": {"reachable": True},
        "database": db,
        "connections": conn,
        "tables": tables,
        "host": f"{client.host}:{client.port}",
        "schema": client.schema,
    }


# ---------------------------------------------------------
# Optional narrower helpers (useful for future dashboard widgets)
# ---------------------------------------------------------

def get_postgres_health_only(pg_service: Any) -> Dict[str, Any]:
    """Just the service snapshot + reachability check."""
    if not pg_service:
        return {
            "success": False,
            "service": None,
            "health": {"reachable": False, "error": "PostgreSQL Server is not registered."},
        }

    snapshot = build_service_snapshot(pg_service).to_dict()

    if pg_service.get_status() != "running":
        return {
            "success": True,
            "service": snapshot,
            "health": {"reachable": False, "error": "PostgreSQL Server is not running."},
        }

    client = _get_client()
    if client is None:
        return {
            "success": True,
            "service": snapshot,
            "health": {"reachable": False, "error": "Postgres client not available."},
        }

    return {
        "success": True,
        "service": snapshot,
        "health": {"reachable": client.health()},
    }