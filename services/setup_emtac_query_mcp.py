from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def write_file(path: Path, content: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        print(f"[SKIP] {path}")
        return

    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    print(f"[WRITE] {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create EMTAC Query MCP server package."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated files.",
    )

    args = parser.parse_args()

    services_root = Path.cwd()

    if services_root.name.lower() != "services":
        print(f"[WARN] This script is intended to run from E:\\emtac\\services")
        print(f"[WARN] Current folder: {services_root}")

    package_dir = services_root / "emtac_query_mcp"
    logs_dir = services_root / "logs"

    package_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    write_file(
        package_dir / "__init__.py",
        r'''
        """
        EMTAC Query MCP package.

        Read-only MCP-style query tools for EMTAC data.
        """
        ''',
        args.overwrite,
    )

    write_file(
        package_dir / "config.py",
        r'''
        from __future__ import annotations

        import os
        from dataclasses import dataclass, field
        from pathlib import Path


        def _int_env(name: str, default: int) -> int:
            value = os.getenv(name)

            if value is None or not str(value).strip():
                return default

            try:
                return int(value)
            except ValueError:
                return default


        @dataclass
        class EmtacQueryMcpConfig:
            """
            Configuration for the EMTAC Query MCP server.

            Environment variables are loaded by launcher.py before this class is created.
            """

            host: str = field(
                default_factory=lambda: os.getenv("EMTAC_QUERY_MCP_HOST", "127.0.0.1")
            )
            port: int = field(
                default_factory=lambda: _int_env("EMTAC_QUERY_MCP_PORT", 8071)
            )
            log_level: str = field(
                default_factory=lambda: os.getenv("EMTAC_QUERY_MCP_LOG_LEVEL", "INFO")
            )

            services_root: Path = field(
                default_factory=lambda: Path(__file__).resolve().parents[1]
            )

            database_url: str = field(
                default_factory=lambda: os.getenv("DATABASE_URL", "")
            )

            postgres_host: str = field(
                default_factory=lambda: os.getenv("POSTGRES_HOST", "127.0.0.1")
            )
            postgres_port: int = field(
                default_factory=lambda: _int_env("POSTGRES_PORT", 5432)
            )
            postgres_db: str = field(
                default_factory=lambda: (
                    os.getenv("POSTGRES_DB")
                    or os.getenv("PGDATABASE")
                    or "emtac"
                )
            )
            postgres_user: str = field(
                default_factory=lambda: (
                    os.getenv("POSTGRES_USER_READ_ONLY")
                    or os.getenv("POSTGRES_USER")
                    or os.getenv("PGUSER")
                    or "postgres"
                )
            )
            postgres_password: str = field(
                default_factory=lambda: (
                    os.getenv("POSTGRES_PASSWORD_READ_ONLY")
                    or os.getenv("POSTGRES_PASSWORD")
                    or os.getenv("PGPASSWORD")
                    or ""
                )
            )

            postgres_timeout: int = field(
                default_factory=lambda: _int_env("EMTAC_QUERY_MCP_TIMEOUT", 10)
            )
            max_rows: int = field(
                default_factory=lambda: _int_env("EMTAC_QUERY_MCP_MAX_ROWS", 250)
            )
            tablet_schema: str = field(
                default_factory=lambda: os.getenv("EMTAC_TABLET_SCHEMA", "tablet_edge")
            )

            @property
            def log_dir(self) -> Path:
                return self.services_root / "logs"

            @property
            def log_file(self) -> Path:
                return self.log_dir / "emtac_query_mcp.log"

            def psycopg2_dsn(self) -> str:
                """
                Build a psycopg2-compatible DSN.

                Handles SQLAlchemy-style DATABASE_URL such as:
                    postgresql+psycopg2://postgres:password@127.0.0.1:5432/emtac
                """
                if self.database_url:
                    return (
                        self.database_url
                        .replace("postgresql+psycopg2://", "postgresql://")
                        .replace("postgres+psycopg2://", "postgresql://")
                    )

                return (
                    f"host={self.postgres_host} "
                    f"port={self.postgres_port} "
                    f"dbname={self.postgres_db} "
                    f"user={self.postgres_user} "
                    f"password={self.postgres_password} "
                    f"connect_timeout={self.postgres_timeout}"
                )

            def safe_dict(self) -> dict:
                return {
                    "host": self.host,
                    "port": self.port,
                    "log_level": self.log_level,
                    "services_root": str(self.services_root),
                    "log_file": str(self.log_file),
                    "database_url_present": bool(self.database_url),
                    "postgres_host": self.postgres_host,
                    "postgres_port": self.postgres_port,
                    "postgres_db": self.postgres_db,
                    "postgres_user": self.postgres_user,
                    "postgres_password": "***" if self.postgres_password else "",
                    "postgres_timeout": self.postgres_timeout,
                    "max_rows": self.max_rows,
                    "tablet_schema": self.tablet_schema,
                }


        def get_config() -> EmtacQueryMcpConfig:
            return EmtacQueryMcpConfig()
        ''',
        args.overwrite,
    )

    write_file(
        package_dir / "logger.py",
        r'''
        from __future__ import annotations

        import logging
        import sys
        from logging.handlers import RotatingFileHandler
        from pathlib import Path
        from typing import Optional


        _LOGGING_CONFIGURED = False


        def configure_logging(
            log_file: Optional[Path] = None,
            level: str = "INFO",
        ) -> None:
            """
            Configure console and rotating file logging.
            """
            global _LOGGING_CONFIGURED

            if _LOGGING_CONFIGURED:
                return

            numeric_level = getattr(logging, str(level).upper(), logging.INFO)

            handlers: list[logging.Handler] = []

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            handlers.append(console_handler)

            if log_file is not None:
                log_file.parent.mkdir(parents=True, exist_ok=True)

                file_handler = RotatingFileHandler(
                    filename=str(log_file),
                    maxBytes=5_000_000,
                    backupCount=5,
                    encoding="utf-8",
                )
                file_handler.setLevel(numeric_level)
                handlers.append(file_handler)

            logging.basicConfig(
                level=numeric_level,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                handlers=handlers,
            )

            _LOGGING_CONFIGURED = True


        def get_logger(name: str = "emtac_query_mcp") -> logging.Logger:
            return logging.getLogger(name)
        ''',
        args.overwrite,
    )

    write_file(
        package_dir / "postgres_client.py",
        r'''
        from __future__ import annotations

        import re
        from contextlib import contextmanager
        from datetime import date, datetime
        from decimal import Decimal
        from typing import Any, Generator, Iterable, Optional

        import psycopg2
        import psycopg2.extras

        from .config import EmtacQueryMcpConfig
        from .logger import get_logger


        logger = get_logger("emtac_query_mcp.postgres_client")


        BLOCKED_SQL_WORDS = {
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "TRUNCATE",
            "CREATE",
            "GRANT",
            "REVOKE",
            "VACUUM",
            "ANALYZE",
            "COPY",
            "CALL",
            "DO",
            "EXECUTE",
        }


        class PostgresReadOnlyClient:
            """
            Read-only PostgreSQL client for MCP-style query tools.
            """

            def __init__(self, config: EmtacQueryMcpConfig):
                self.config = config
                self._dsn = config.psycopg2_dsn()

                logger.info(
                    "PostgresReadOnlyClient initialized | config=%s",
                    config.safe_dict(),
                )

            @contextmanager
            def connect(self) -> Generator[psycopg2.extensions.connection, None, None]:
                conn = psycopg2.connect(self._dsn)
                try:
                    conn.set_session(readonly=True, autocommit=True)
                    yield conn
                finally:
                    conn.close()

            def health(self) -> bool:
                try:
                    rows = self.fetch_all("SELECT 1 AS alive")
                    return bool(rows and rows[0].get("alive") == 1)
                except Exception as exc:
                    logger.warning("Postgres health check failed: %s", exc)
                    return False

            def fetch_all(
                self,
                sql: str,
                params: Optional[tuple[Any, ...]] = None,
                *,
                max_rows: Optional[int] = None,
            ) -> list[dict[str, Any]]:
                limit = max_rows or self.config.max_rows

                with self.connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql, params or ())
                        rows = cur.fetchmany(limit)

                return normalize_rows([dict(row) for row in rows])

            def execute_select(
                self,
                sql: str,
                params: Optional[tuple[Any, ...]] = None,
            ) -> list[dict[str, Any]]:
                validate_read_only_sql(sql)
                return self.fetch_all(sql, params=params)

            def server_info(self) -> dict[str, Any]:
                rows = self.fetch_all(
                    """
                    SELECT
                        current_database() AS current_database,
                        current_user AS current_user,
                        inet_server_addr() AS server_addr,
                        inet_server_port() AS server_port,
                        now() AS server_time,
                        version() AS server_version
                    """
                )

                return rows[0] if rows else {}


        def normalize_value(value: Any) -> Any:
            if isinstance(value, (datetime, date)):
                return value.isoformat()

            if isinstance(value, Decimal):
                return float(value)

            return value


        def normalize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {
                    key: normalize_value(value)
                    for key, value in row.items()
                }
                for row in rows
            ]


        def strip_sql_comments(sql: str) -> str:
            without_line_comments = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
            without_block_comments = re.sub(
                r"/\*.*?\*/",
                "",
                without_line_comments,
                flags=re.DOTALL,
            )
            return without_block_comments.strip()


        def validate_read_only_sql(sql: str) -> None:
            cleaned = strip_sql_comments(sql)
            upper = cleaned.upper().strip()

            if not upper:
                raise ValueError("SQL query is empty.")

            if not (upper.startswith("SELECT") or upper.startswith("WITH")):
                raise ValueError("Only SELECT or WITH queries are allowed.")

            tokens = set(re.findall(r"\b[A-Z_]+\b", upper))
            blocked = sorted(tokens.intersection(BLOCKED_SQL_WORDS))

            if blocked:
                raise ValueError(
                    "SQL contains blocked write/DDL keywords: "
                    + ", ".join(blocked)
                )
        ''',
        args.overwrite,
    )

    write_file(
        package_dir / "tablet_edge_tools.py",
        r'''
        from __future__ import annotations

        from typing import Any, Callable, Optional

        from .config import EmtacQueryMcpConfig
        from .logger import get_logger
        from .postgres_client import PostgresReadOnlyClient


        logger = get_logger("emtac_query_mcp.tablet_edge_tools")


        ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


        class TabletEdgeTools:
            """
            MCP-style read-only tools for the tablet_edge schema.
            """

            def __init__(
                self,
                db: PostgresReadOnlyClient,
                config: EmtacQueryMcpConfig,
            ) -> None:
                self.db = db
                self.config = config
                self.schema = config.tablet_schema

                self._handlers: dict[str, ToolHandler] = {
                    "tablet.table_counts": self.table_counts,
                    "tablet.list_devices": self.list_devices,
                    "tablet.latest_device_status": self.latest_device_status,
                    "tablet.recent_sync_events": self.recent_sync_events,
                    "tablet.recent_network_events": self.recent_network_events,
                    "tablet.recent_health_samples": self.recent_health_samples,
                    "tablet.recent_offline_events": self.recent_offline_events,
                    "tablet.recent_app_logs": self.recent_app_logs,
                    "tablet.sync_failures": self.sync_failures,
                    "tablet.inactive_devices": self.inactive_devices,
                    "postgres.server_info": self.postgres_server_info,
                }

            def list_tools(self) -> list[dict[str, Any]]:
                return [
                    {
                        "name": "tablet.table_counts",
                        "description": "Return row counts for tablet_edge tables.",
                        "args": {},
                    },
                    {
                        "name": "tablet.list_devices",
                        "description": "List registered tablets.",
                        "args": {"limit": "optional int"},
                    },
                    {
                        "name": "tablet.latest_device_status",
                        "description": "Show latest status for each tablet.",
                        "args": {"limit": "optional int"},
                    },
                    {
                        "name": "tablet.recent_sync_events",
                        "description": "Show recent sync events. Optional tablet_uid filter.",
                        "args": {"tablet_uid": "optional UUID", "limit": "optional int"},
                    },
                    {
                        "name": "tablet.recent_network_events",
                        "description": "Show recent network events. Optional tablet_uid filter.",
                        "args": {"tablet_uid": "optional UUID", "limit": "optional int"},
                    },
                    {
                        "name": "tablet.recent_health_samples",
                        "description": "Show recent health samples. Optional tablet_uid filter.",
                        "args": {"tablet_uid": "optional UUID", "limit": "optional int"},
                    },
                    {
                        "name": "tablet.recent_offline_events",
                        "description": "Show recent offline events. Optional tablet_uid filter.",
                        "args": {"tablet_uid": "optional UUID", "limit": "optional int"},
                    },
                    {
                        "name": "tablet.recent_app_logs",
                        "description": "Show recent app logs. Optional tablet_uid filter.",
                        "args": {"tablet_uid": "optional UUID", "limit": "optional int"},
                    },
                    {
                        "name": "tablet.sync_failures",
                        "description": "Show recent failed or partial sync events.",
                        "args": {"limit": "optional int"},
                    },
                    {
                        "name": "tablet.inactive_devices",
                        "description": "Show active tablets not seen in N minutes.",
                        "args": {"minutes": "optional int default 5", "limit": "optional int"},
                    },
                    {
                        "name": "postgres.server_info",
                        "description": "Return PostgreSQL server/database connection info.",
                        "args": {},
                    },
                ]

            def call_tool(
                self,
                name: str,
                args: Optional[dict[str, Any]] = None,
            ) -> dict[str, Any]:
                args = args or {}
                handler = self._handlers.get(name)

                if handler is None:
                    return {
                        "ok": False,
                        "error": f"Unknown tool: {name}",
                        "available_tools": sorted(self._handlers.keys()),
                    }

                logger.info("Tool call | name=%s | args=%s", name, args)

                try:
                    result = handler(args)
                    result.setdefault("ok", True)
                    result.setdefault("tool", name)
                    return result
                except Exception as exc:
                    logger.exception("Tool failed | name=%s | error=%s", name, exc)
                    return {
                        "ok": False,
                        "tool": name,
                        "error": str(exc),
                    }

            def _limit(
                self,
                args: dict[str, Any],
                default: int = 50,
                maximum: int = 500,
            ) -> int:
                raw = args.get("limit", default)

                try:
                    value = int(raw)
                except Exception:
                    value = default

                return max(1, min(value, maximum, self.config.max_rows))

            def _tablet_uid(self, args: dict[str, Any]) -> str | None:
                value = args.get("tablet_uid")

                if value is None:
                    return None

                value = str(value).strip()

                return value or None

            def table_counts(self, args: dict[str, Any]) -> dict[str, Any]:
                sql = f"""
                    SELECT 'tablet_device' AS table_name, COUNT(*)::int AS row_count
                    FROM {self.schema}.tablet_device

                    UNION ALL

                    SELECT 'tablet_network_event', COUNT(*)::int
                    FROM {self.schema}.tablet_network_event

                    UNION ALL

                    SELECT 'tablet_health_sample', COUNT(*)::int
                    FROM {self.schema}.tablet_health_sample

                    UNION ALL

                    SELECT 'tablet_dropdown_cache_manifest', COUNT(*)::int
                    FROM {self.schema}.tablet_dropdown_cache_manifest

                    UNION ALL

                    SELECT 'tablet_sync_event', COUNT(*)::int
                    FROM {self.schema}.tablet_sync_event

                    UNION ALL

                    SELECT 'tablet_offline_event', COUNT(*)::int
                    FROM {self.schema}.tablet_offline_event

                    UNION ALL

                    SELECT 'tablet_app_log', COUNT(*)::int
                    FROM {self.schema}.tablet_app_log

                    ORDER BY table_name
                """

                rows = self.db.fetch_all(sql)

                return {
                    "rows": rows,
                    "count": len(rows),
                }

            def list_devices(self, args: dict[str, Any]) -> dict[str, Any]:
                limit = self._limit(args, default=50)

                sql = f"""
                    SELECT
                        id,
                        tablet_uid,
                        tablet_name,
                        device_make,
                        device_model,
                        android_version,
                        app_version,
                        assigned_area,
                        assigned_station,
                        assigned_role,
                        is_active,
                        first_seen_at,
                        last_seen_at,
                        created_at,
                        updated_at
                    FROM {self.schema}.tablet_device
                    ORDER BY id DESC
                    LIMIT %s
                """

                rows = self.db.fetch_all(sql, (limit,))

                return {
                    "rows": rows,
                    "count": len(rows),
                }

            def latest_device_status(self, args: dict[str, Any]) -> dict[str, Any]:
                limit = self._limit(args, default=50)

                sql = f"""
                    SELECT
                        d.id,
                        d.tablet_uid,
                        d.tablet_name,
                        d.device_make,
                        d.device_model,
                        d.app_version,
                        d.is_active,
                        d.last_seen_at,
                        EXTRACT(EPOCH FROM (now() - d.last_seen_at))::int AS seconds_since_last_seen,
                        latest_sync.sync_type AS latest_sync_type,
                        latest_sync.status AS latest_sync_status,
                        latest_sync.started_at AS latest_sync_at,
                        latest_network.event_type AS latest_network_event_type,
                        latest_network.quality_level AS latest_network_quality,
                        latest_network.latency_ms AS latest_network_latency_ms,
                        latest_network.created_at AS latest_network_at
                    FROM {self.schema}.tablet_device d

                    LEFT JOIN LATERAL (
                        SELECT
                            s.sync_type,
                            s.status,
                            s.started_at
                        FROM {self.schema}.tablet_sync_event s
                        WHERE s.tablet_device_id = d.id
                        ORDER BY s.started_at DESC
                        LIMIT 1
                    ) latest_sync ON TRUE

                    LEFT JOIN LATERAL (
                        SELECT
                            e.event_type,
                            e.quality_level,
                            e.latency_ms,
                            e.created_at
                        FROM {self.schema}.tablet_network_event e
                        WHERE e.tablet_device_id = d.id
                        ORDER BY e.created_at DESC
                        LIMIT 1
                    ) latest_network ON TRUE

                    ORDER BY d.last_seen_at DESC NULLS LAST, d.id DESC
                    LIMIT %s
                """

                rows = self.db.fetch_all(sql, (limit,))

                return {
                    "rows": rows,
                    "count": len(rows),
                }

            def recent_sync_events(self, args: dict[str, Any]) -> dict[str, Any]:
                limit = self._limit(args, default=50)
                tablet_uid = self._tablet_uid(args)

                params: list[Any] = []
                where = ""

                if tablet_uid:
                    where = "WHERE d.tablet_uid = %s"
                    params.append(tablet_uid)

                params.append(limit)

                sql = f"""
                    SELECT
                        s.id,
                        d.tablet_uid,
                        d.tablet_name,
                        s.sync_type,
                        s.sync_direction,
                        s.status,
                        s.records_sent,
                        s.records_received,
                        s.records_failed,
                        s.duration_ms,
                        s.error_message,
                        s.started_at,
                        s.completed_at,
                        s.created_at
                    FROM {self.schema}.tablet_sync_event s
                    JOIN {self.schema}.tablet_device d
                        ON d.id = s.tablet_device_id
                    {where}
                    ORDER BY s.started_at DESC
                    LIMIT %s
                """

                rows = self.db.fetch_all(sql, tuple(params))

                return {
                    "rows": rows,
                    "count": len(rows),
                }

            def recent_network_events(self, args: dict[str, Any]) -> dict[str, Any]:
                limit = self._limit(args, default=50)
                tablet_uid = self._tablet_uid(args)

                params: list[Any] = []
                where = ""

                if tablet_uid:
                    where = "WHERE d.tablet_uid = %s"
                    params.append(tablet_uid)

                params.append(limit)

                sql = f"""
                    SELECT
                        e.id,
                        d.tablet_uid,
                        d.tablet_name,
                        e.event_type,
                        e.quality_level,
                        e.server_url,
                        e.page_url,
                        e.latency_ms,
                        e.avg_latency_ms,
                        e.consecutive_failures,
                        e.is_online,
                        e.ssid,
                        e.wifi_rssi,
                        e.signal_level,
                        e.ip_address,
                        e.gateway_address,
                        e.message,
                        e.event_started_at,
                        e.event_ended_at,
                        e.created_at
                    FROM {self.schema}.tablet_network_event e
                    JOIN {self.schema}.tablet_device d
                        ON d.id = e.tablet_device_id
                    {where}
                    ORDER BY e.created_at DESC
                    LIMIT %s
                """

                rows = self.db.fetch_all(sql, tuple(params))

                return {
                    "rows": rows,
                    "count": len(rows),
                }

            def recent_health_samples(self, args: dict[str, Any]) -> dict[str, Any]:
                limit = self._limit(args, default=50)
                tablet_uid = self._tablet_uid(args)

                params: list[Any] = []
                where = ""

                if tablet_uid:
                    where = "WHERE d.tablet_uid = %s"
                    params.append(tablet_uid)

                params.append(limit)

                sql = f"""
                    SELECT
                        h.id,
                        d.tablet_uid,
                        d.tablet_name,
                        h.sampled_at,
                        h.server_reachable,
                        h.server_latency_ms,
                        h.quality_level,
                        h.battery_percent,
                        h.is_charging,
                        h.ssid,
                        h.wifi_rssi,
                        h.signal_level,
                        h.app_foreground,
                        h.current_page_url,
                        h.created_at
                    FROM {self.schema}.tablet_health_sample h
                    JOIN {self.schema}.tablet_device d
                        ON d.id = h.tablet_device_id
                    {where}
                    ORDER BY h.sampled_at DESC
                    LIMIT %s
                """

                rows = self.db.fetch_all(sql, tuple(params))

                return {
                    "rows": rows,
                    "count": len(rows),
                }

            def recent_offline_events(self, args: dict[str, Any]) -> dict[str, Any]:
                limit = self._limit(args, default=50)
                tablet_uid = self._tablet_uid(args)

                params: list[Any] = []
                where = ""

                if tablet_uid:
                    where = "WHERE d.tablet_uid = %s"
                    params.append(tablet_uid)

                params.append(limit)

                sql = f"""
                    SELECT
                        o.id,
                        d.tablet_uid,
                        d.tablet_name,
                        o.local_event_id,
                        o.event_type,
                        o.event_payload,
                        o.client_created_at,
                        o.server_received_at,
                        o.processing_status,
                        o.processed_at,
                        o.error_message,
                        o.created_at
                    FROM {self.schema}.tablet_offline_event o
                    JOIN {self.schema}.tablet_device d
                        ON d.id = o.tablet_device_id
                    {where}
                    ORDER BY o.created_at DESC
                    LIMIT %s
                """

                rows = self.db.fetch_all(sql, tuple(params))

                return {
                    "rows": rows,
                    "count": len(rows),
                }

            def recent_app_logs(self, args: dict[str, Any]) -> dict[str, Any]:
                limit = self._limit(args, default=50)
                tablet_uid = self._tablet_uid(args)

                params: list[Any] = []
                where = ""

                if tablet_uid:
                    where = "WHERE d.tablet_uid = %s"
                    params.append(tablet_uid)

                params.append(limit)

                sql = f"""
                    SELECT
                        l.id,
                        d.tablet_uid,
                        d.tablet_name,
                        l.log_level,
                        l.log_source,
                        l.message,
                        l.context,
                        l.client_created_at,
                        l.server_received_at,
                        l.created_at
                    FROM {self.schema}.tablet_app_log l
                    LEFT JOIN {self.schema}.tablet_device d
                        ON d.id = l.tablet_device_id
                    {where}
                    ORDER BY l.created_at DESC
                    LIMIT %s
                """

                rows = self.db.fetch_all(sql, tuple(params))

                return {
                    "rows": rows,
                    "count": len(rows),
                }

            def sync_failures(self, args: dict[str, Any]) -> dict[str, Any]:
                limit = self._limit(args, default=50)

                sql = f"""
                    SELECT
                        s.id,
                        d.tablet_uid,
                        d.tablet_name,
                        s.sync_type,
                        s.sync_direction,
                        s.status,
                        s.records_sent,
                        s.records_received,
                        s.records_failed,
                        s.error_message,
                        s.started_at,
                        s.completed_at,
                        s.created_at
                    FROM {self.schema}.tablet_sync_event s
                    JOIN {self.schema}.tablet_device d
                        ON d.id = s.tablet_device_id
                    WHERE s.status IN ('failed', 'partial')
                       OR s.records_failed > 0
                    ORDER BY s.started_at DESC
                    LIMIT %s
                """

                rows = self.db.fetch_all(sql, (limit,))

                return {
                    "rows": rows,
                    "count": len(rows),
                }

            def inactive_devices(self, args: dict[str, Any]) -> dict[str, Any]:
                limit = self._limit(args, default=50)

                raw_minutes = args.get("minutes", 5)

                try:
                    minutes = int(raw_minutes)
                except Exception:
                    minutes = 5

                minutes = max(1, min(minutes, 1440))

                sql = f"""
                    SELECT
                        id,
                        tablet_uid,
                        tablet_name,
                        device_make,
                        device_model,
                        app_version,
                        is_active,
                        last_seen_at,
                        EXTRACT(EPOCH FROM (now() - last_seen_at))::int AS seconds_since_last_seen
                    FROM {self.schema}.tablet_device
                    WHERE is_active = TRUE
                      AND (
                            last_seen_at IS NULL
                            OR last_seen_at < now() - (%s * interval '1 minute')
                          )
                    ORDER BY last_seen_at ASC NULLS FIRST
                    LIMIT %s
                """

                rows = self.db.fetch_all(sql, (minutes, limit))

                return {
                    "minutes": minutes,
                    "rows": rows,
                    "count": len(rows),
                }

            def postgres_server_info(self, args: dict[str, Any]) -> dict[str, Any]:
                return {
                    "server_info": self.db.server_info(),
                }
        ''',
        args.overwrite,
    )

    write_file(
        package_dir / "server.py",
        r'''
        from __future__ import annotations

        import asyncio
        import json
        import time
        from typing import Any

        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse

        from .config import get_config
        from .logger import configure_logging, get_logger
        from .postgres_client import PostgresReadOnlyClient
        from .tablet_edge_tools import TabletEdgeTools


        config = get_config()
        configure_logging(log_file=config.log_file, level=config.log_level)
        logger = get_logger("emtac_query_mcp.server")

        db_client = PostgresReadOnlyClient(config)
        tablet_tools = TabletEdgeTools(db=db_client, config=config)

        app = FastAPI(
            title="EMTAC Query MCP Server",
            description="Read-only MCP-style query tools for EMTAC data.",
            version="0.1.0",
        )


        @app.on_event("startup")
        async def startup_event() -> None:
            logger.info("EMTAC Query MCP server starting")
            logger.info("Config: %s", config.safe_dict())

            if db_client.health():
                logger.info("PostgreSQL connection is healthy")
            else:
                logger.warning("PostgreSQL connection is not healthy at startup")


        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "ok": True,
                "service": "emtac_query_mcp",
                "version": "0.1.0",
                "postgres_available": db_client.health(),
                "config": config.safe_dict(),
            }


        @app.get("/tools")
        async def list_tools() -> dict[str, Any]:
            tools = tablet_tools.list_tools()

            return {
                "ok": True,
                "count": len(tools),
                "tools": tools,
            }


        @app.post("/tools/call")
        async def call_tool(request: Request):
            payload = await request.json()

            tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
            args = payload.get("args") or {}

            if not tool_name:
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "error": "Missing required field: tool",
                    },
                )

            if not isinstance(args, dict):
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "tool": tool_name,
                        "error": "args must be a JSON object",
                    },
                )

            result = tablet_tools.call_tool(tool_name, args)
            status_code = 200 if result.get("ok") else 400

            return JSONResponse(status_code=status_code, content=result)


        @app.get("/sse")
        async def sse():
            async def stream():
                logger.info("MCP SSE client connected")

                yield "event: ready\ndata: {\"service\":\"emtac_query_mcp\"}\n\n"

                while True:
                    await asyncio.sleep(20)
                    yield "event: ping\ndata: {}\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")


        def content_to_text(content: Any) -> str:
            if content is None:
                return ""

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, list):
                parts: list[str] = []

                for item in content:
                    if isinstance(item, str):
                        text = item.strip()
                        if text:
                            parts.append(text)
                        continue

                    if isinstance(item, dict):
                        for key in ("text", "content", "value"):
                            if key in item:
                                text = str(item.get(key) or "").strip()
                                if text:
                                    parts.append(text)
                                    break

                return "\n".join(parts).strip()

            if isinstance(content, dict):
                for key in ("text", "content", "value"):
                    if key in content:
                        return str(content.get(key) or "").strip()

            return str(content).strip()


        def route_text_to_tool(text: str) -> tuple[str, dict[str, Any]]:
            value = text.lower()

            if "table count" in value or "counts" in value:
                return "tablet.table_counts", {}

            if "inactive" in value or "not checked in" in value:
                return "tablet.inactive_devices", {"minutes": 5, "limit": 50}

            if "network" in value:
                return "tablet.recent_network_events", {"limit": 20}

            if "health" in value or "battery" in value:
                return "tablet.recent_health_samples", {"limit": 20}

            if "failure" in value or "failed" in value:
                return "tablet.sync_failures", {"limit": 20}

            if "sync" in value or "heartbeat" in value:
                return "tablet.recent_sync_events", {"limit": 20}

            if "app log" in value or "logs" in value:
                return "tablet.recent_app_logs", {"limit": 20}

            if "offline" in value:
                return "tablet.recent_offline_events", {"limit": 20}

            if "device" in value or "tablet" in value:
                return "tablet.list_devices", {"limit": 20}

            return "tablet.latest_device_status", {"limit": 20}


        @app.post("/mcp")
        async def mcp(request: Request):
            payload = await request.json()

            if "tool" in payload or "name" in payload:
                tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
                args = payload.get("args") or {}

                if not isinstance(args, dict):
                    args = {}

                result = tablet_tools.call_tool(tool_name, args)
                status_code = 200 if result.get("ok") else 400

                return JSONResponse(status_code=status_code, content=result)

            content = content_to_text(payload.get("content", ""))
            tool_name, args = route_text_to_tool(content)

            result = tablet_tools.call_tool(tool_name, args)

            return JSONResponse(
                status_code=200 if result.get("ok") else 400,
                content={
                    "ok": result.get("ok", False),
                    "routed_tool": tool_name,
                    "args": args,
                    "result": result,
                },
            )


        @app.get("/v1/models")
        async def list_models() -> dict[str, Any]:
            return {
                "object": "list",
                "data": [
                    {
                        "id": "emtac-query-mcp",
                        "object": "model",
                        "owned_by": "emtac",
                    }
                ],
            }


        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            payload = await request.json()

            messages = payload.get("messages", [])
            stream = bool(payload.get("stream", False))
            model = str(payload.get("model") or "emtac-query-mcp")

            last_user_text = ""

            if isinstance(messages, list):
                for message in reversed(messages):
                    if not isinstance(message, dict):
                        continue

                    if str(message.get("role", "")).lower() == "user":
                        last_user_text = content_to_text(message.get("content"))
                        break

            tool_name, args = route_text_to_tool(last_user_text)
            result = tablet_tools.call_tool(tool_name, args)

            response_text = json.dumps(
                {
                    "routed_tool": tool_name,
                    "args": args,
                    "result": result,
                },
                indent=2,
                default=str,
            )

            if stream:
                async def stream_response():
                    chunk = {
                        "id": "chatcmpl-emtac-query-mcp",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": response_text,
                                },
                                "finish_reason": None,
                            }
                        ],
                    }

                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.05)

                    done = {
                        "id": "chatcmpl-emtac-query-mcp",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }

                    yield f"data: {json.dumps(done)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_response(), media_type="text/event-stream")

            return JSONResponse(
                content={
                    "id": "chatcmpl-emtac-query-mcp",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        ''',
        args.overwrite,
    )

    write_file(
        package_dir / "launcher.py",
        r'''
        from __future__ import annotations

        from pathlib import Path
        from typing import Optional

        from dotenv import load_dotenv

        from .config import get_config
        from .logger import configure_logging, get_logger


        def load_project_env() -> Optional[Path]:
            """
            Load the project/service env file from common EMTAC locations.
            """
            package_file = Path(__file__).resolve()
            services_root = package_file.parents[1]
            emtac_root = services_root.parent

            candidates = [
                services_root / ".env",
                emtac_root / ".env",
                emtac_root / "dev_env" / ".env",
                Path(r"E:\emtac\dev_env\.env"),
                Path(r"E:\emtac\.env"),
            ]

            logger = get_logger("emtac_query_mcp.launcher.env")

            for env_path in candidates:
                if env_path.exists():
                    load_dotenv(env_path, override=False)
                    logger.info("Loaded environment from %s", env_path)
                    return env_path

            logger.warning("No environment file found in expected locations")
            return None


        def main() -> None:
            load_project_env()

            config = get_config()
            configure_logging(log_file=config.log_file, level=config.log_level)

            logger = get_logger("emtac_query_mcp.launcher")

            logger.info("Starting EMTAC Query MCP")
            logger.info("Config: %s", config.safe_dict())

            import uvicorn

            uvicorn.run(
                "emtac_query_mcp.server:app",
                host=config.host,
                port=config.port,
                reload=False,
            )


        if __name__ == "__main__":
            main()
        ''',
        args.overwrite,
    )

    write_file(
        package_dir / "README.md",
        r'''
        # EMTAC Query MCP Server

        Read-only MCP-style query server for EMTAC data.

        ## Run

        From:

        ```powershell
        E:\emtac\services
        ```

        Run:

        ```powershell
        python -m emtac_query_mcp.launcher
        ```

        Default URL:

        ```text
        http://127.0.0.1:8071
        ```

        ## Health

        ```text
        http://127.0.0.1:8071/health
        ```

        ## List tools

        ```text
        http://127.0.0.1:8071/tools
        ```

        ## Tool call example

        ```powershell
        $body = @{
            tool = "tablet.recent_network_events"
            args = @{
                limit = 10
            }
        } | ConvertTo-Json -Depth 10

        Invoke-RestMethod `
            -Uri "http://127.0.0.1:8071/tools/call" `
            -Method POST `
            -ContentType "application/json" `
            -Body $body
        ```

        ## Safety

        This server is read-only and exposes named query tools.
        ''',
        args.overwrite,
    )

    write_file(
        package_dir / ".env.example",
        r'''
        EMTAC_QUERY_MCP_HOST=127.0.0.1
        EMTAC_QUERY_MCP_PORT=8071
        EMTAC_QUERY_MCP_LOG_LEVEL=INFO
        EMTAC_QUERY_MCP_MAX_ROWS=250
        EMTAC_QUERY_MCP_TIMEOUT=10

        DATABASE_URL=postgresql+psycopg2://postgres:emtac123@127.0.0.1:5432/emtac

        POSTGRES_HOST=127.0.0.1
        POSTGRES_PORT=5432
        POSTGRES_DB=emtac
        POSTGRES_USER=postgres
        POSTGRES_PASSWORD=emtac123

        EMTAC_TABLET_SCHEMA=tablet_edge
        ''',
        args.overwrite,
    )

    write_file(
        package_dir / "requirements.txt",
        r'''
        fastapi
        uvicorn
        psycopg2-binary
        python-dotenv
        ''',
        args.overwrite,
    )

    write_file(
        services_root / "run_emtac_query_mcp.py",
        r'''
        from emtac_query_mcp.launcher import main


        if __name__ == "__main__":
            main()
        ''',
        args.overwrite,
    )

    print()
    print("[DONE] EMTAC Query MCP setup complete.")
    print()
    print(f"Package created at: {package_dir}")
    print()
    print("Run:")
    print("  cd E:\\emtac\\services")
    print("  python -m emtac_query_mcp.launcher")
    print()
    print("Or:")
    print("  python .\\run_emtac_query_mcp.py")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())