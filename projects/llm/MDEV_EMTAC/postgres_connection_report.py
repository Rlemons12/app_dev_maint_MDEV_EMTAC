#!/usr/bin/env python3
"""
postgres_connection_report.py

Connects to PostgreSQL and prints a clean connection report.

Supports these environment variables:
- POSTGRES_USER
- POSTGRES_PASSWORD
- POSTGRES_HOST
- POSTGRES_PORT
- POSTGRES_DB
- DATABASE_URL

Also supports command-line overrides.

Examples:
    python postgres_connection_report.py
    python postgres_connection_report.py --filter-db emtac
    python postgres_connection_report.py --show-locks
    python postgres_connection_report.py --limit 50
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import urlparse, unquote

import psycopg2
import psycopg2.extras


@dataclass
class Config:
    dsn: Optional[str]
    host: Optional[str]
    port: int
    dbname: Optional[str]
    user: Optional[str]
    password: Optional[str]
    filter_db: Optional[str]
    limit: int
    show_locks: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Print a PostgreSQL connection report from pg_stat_activity."
    )

    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection string. Defaults to DATABASE_URL env var.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("POSTGRES_HOST"),
        help="PostgreSQL host. Defaults to POSTGRES_HOST env var.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("POSTGRES_PORT", "5432")),
        help="PostgreSQL port. Defaults to POSTGRES_PORT env var or 5432.",
    )
    parser.add_argument(
        "--dbname",
        default=os.getenv("POSTGRES_DB"),
        help="Database name. Defaults to POSTGRES_DB env var.",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("POSTGRES_USER"),
        help="Database user. Defaults to POSTGRES_USER env var.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("POSTGRES_PASSWORD"),
        help="Database password. Defaults to POSTGRES_PASSWORD env var.",
    )
    parser.add_argument(
        "--filter-db",
        default=None,
        help="Only report sessions for one database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum rows shown in detail sections.",
    )
    parser.add_argument(
        "--show-locks",
        action="store_true",
        help="Include lock details.",
    )

    args = parser.parse_args()

    return Config(
        dsn=args.dsn,
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        filter_db=args.filter_db,
        limit=args.limit,
        show_locks=args.show_locks,
    )


def normalize_sqlalchemy_url(url: str) -> str:
    """
    Convert a SQLAlchemy-style URL like:
        postgresql+psycopg2://user:pass@host:5432/dbname
    into a psycopg2-compatible DSN string.

    psycopg2 accepts:
        postgresql://user:pass@host:5432/dbname
    """
    url = url.strip()

    if url.startswith("postgresql+psycopg2://"):
        return "postgresql://" + url[len("postgresql+psycopg2://") :]

    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]

    return url


def parse_database_url(url: str) -> dict[str, Any]:
    """
    Parse DATABASE_URL into psycopg2 connection parameters.

    Example input:
        postgresql+psycopg2://postgres:emtac123@127.0.0.1:5432/emtac
    """
    normalized = normalize_sqlalchemy_url(url)
    parsed = urlparse(normalized)

    if parsed.scheme not in {"postgresql", "postgres"}:
        raise ValueError(
            f"Unsupported DATABASE_URL scheme: {parsed.scheme!r}. "
            "Expected postgresql:// or postgresql+psycopg2://"
        )

    dbname = parsed.path.lstrip("/") if parsed.path else None

    return {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "dbname": dbname,
        "user": unquote(parsed.username) if parsed.username else None,
        "password": unquote(parsed.password) if parsed.password else None,
    }


def build_connection_params(config: Config) -> dict[str, Any]:
    """
    Preference order:
    1. DATABASE_URL / --dsn
    2. Individual POSTGRES_* values / CLI overrides
    """
    if config.dsn:
        parsed = parse_database_url(config.dsn)
        return parsed

    missing = []
    if not config.host:
        missing.append("host")
    if not config.dbname:
        missing.append("dbname")
    if not config.user:
        missing.append("user")
    if config.password is None:
        missing.append("password")

    if missing:
        raise ValueError(
            "Missing required connection parameters: "
            + ", ".join(missing)
            + ". Set DATABASE_URL or POSTGRES_* environment variables."
        )

    return {
        "host": config.host,
        "port": config.port,
        "dbname": config.dbname,
        "user": config.user,
        "password": config.password,
    }


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def truncate(value: Any, max_len: int = 100) -> str:
    text = safe_str(value).replace("\n", " ").replace("\r", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_timedelta(value: Any) -> str:
    return "" if value is None else str(value)


def print_header(title: str) -> None:
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def print_kv(label: str, value: Any) -> None:
    print(f"{label:<30}: {value}")


def print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], max_width: int = 42) -> None:
    if not rows:
        print("(none)")
        return

    widths: list[int] = []
    for key, header in columns:
        width = len(header)
        for row in rows:
            width = min(max(width, len(truncate(row.get(key), max_width))), max_width)
        widths.append(width)

    header_line = " | ".join(
        header.ljust(width)
        for (_, header), width in zip(columns, widths)
    )
    separator_line = "-+-".join("-" * width for width in widths)

    print(header_line)
    print(separator_line)

    for row in rows:
        line = " | ".join(
            truncate(row.get(key), max_width).ljust(width)
            for (key, _), width in zip(columns, widths)
        )
        print(line)


def fetch_all_dicts(cursor: psycopg2.extensions.cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def run_query(
    conn: psycopg2.extensions.connection,
    sql: str,
    params: Optional[tuple[Any, ...]] = None,
) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params or ())
        return fetch_all_dicts(cur)


def get_server_info(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    sql = """
    SELECT
        current_database() AS current_database,
        current_user AS current_user,
        inet_server_addr() AS server_addr,
        inet_server_port() AS server_port,
        version() AS server_version,
        now() AS report_time
    """
    return run_query(conn, sql)[0]


def get_connection_summary(conn: psycopg2.extensions.connection, filter_db: Optional[str]) -> list[dict[str, Any]]:
    sql = """
    SELECT
        datname,
        COALESCE(state, '(null)') AS state,
        COUNT(*) AS connection_count
    FROM pg_stat_activity
    WHERE (%s IS NULL OR datname = %s)
    GROUP BY datname, state
    ORDER BY datname, state
    """
    return run_query(conn, sql, (filter_db, filter_db))


def get_application_summary(conn: psycopg2.extensions.connection, filter_db: Optional[str]) -> list[dict[str, Any]]:
    sql = """
    SELECT
        COALESCE(NULLIF(application_name, ''), '(blank)') AS application_name,
        usename,
        COALESCE(client_addr::text, 'local') AS client_addr,
        COALESCE(state, '(null)') AS state,
        COUNT(*) AS connection_count
    FROM pg_stat_activity
    WHERE (%s IS NULL OR datname = %s)
    GROUP BY application_name, usename, client_addr, state
    ORDER BY connection_count DESC, application_name
    """
    return run_query(conn, sql, (filter_db, filter_db))


def get_detailed_sessions(conn: psycopg2.extensions.connection, filter_db: Optional[str], limit: int) -> list[dict[str, Any]]:
    sql = """
    SELECT
        pid,
        datname,
        usename,
        COALESCE(NULLIF(application_name, ''), '(blank)') AS application_name,
        COALESCE(client_addr::text, 'local') AS client_addr,
        client_port,
        backend_start,
        now() - backend_start AS connection_age,
        xact_start,
        CASE
            WHEN xact_start IS NOT NULL THEN now() - xact_start
            ELSE NULL
        END AS transaction_age,
        query_start,
        CASE
            WHEN query_start IS NOT NULL THEN now() - query_start
            ELSE NULL
        END AS query_age,
        state,
        wait_event_type,
        wait_event,
        backend_type,
        query
    FROM pg_stat_activity
    WHERE (%s IS NULL OR datname = %s)
    ORDER BY backend_start ASC
    LIMIT %s
    """
    return run_query(conn, sql, (filter_db, filter_db, limit))


def get_idle_sessions(conn: psycopg2.extensions.connection, filter_db: Optional[str], limit: int) -> list[dict[str, Any]]:
    sql = """
    SELECT
        pid,
        datname,
        usename,
        COALESCE(NULLIF(application_name, ''), '(blank)') AS application_name,
        COALESCE(client_addr::text, 'local') AS client_addr,
        state,
        state_change,
        now() - state_change AS idle_for,
        query
    FROM pg_stat_activity
    WHERE state = 'idle'
      AND (%s IS NULL OR datname = %s)
    ORDER BY state_change ASC
    LIMIT %s
    """
    return run_query(conn, sql, (filter_db, filter_db, limit))


def get_idle_in_transaction_sessions(
    conn: psycopg2.extensions.connection,
    filter_db: Optional[str],
    limit: int,
) -> list[dict[str, Any]]:
    sql = """
    SELECT
        pid,
        datname,
        usename,
        COALESCE(NULLIF(application_name, ''), '(blank)') AS application_name,
        COALESCE(client_addr::text, 'local') AS client_addr,
        xact_start,
        now() - xact_start AS transaction_age,
        state,
        wait_event_type,
        wait_event,
        query
    FROM pg_stat_activity
    WHERE state = 'idle in transaction'
      AND (%s IS NULL OR datname = %s)
    ORDER BY xact_start ASC
    LIMIT %s
    """
    return run_query(conn, sql, (filter_db, filter_db, limit))


def get_long_running_queries(
    conn: psycopg2.extensions.connection,
    filter_db: Optional[str],
    limit: int,
) -> list[dict[str, Any]]:
    sql = """
    SELECT
        pid,
        datname,
        usename,
        COALESCE(NULLIF(application_name, ''), '(blank)') AS application_name,
        COALESCE(client_addr::text, 'local') AS client_addr,
        query_start,
        now() - query_start AS query_age,
        state,
        wait_event_type,
        wait_event,
        query
    FROM pg_stat_activity
    WHERE query_start IS NOT NULL
      AND (%s IS NULL OR datname = %s)
    ORDER BY query_start ASC
    LIMIT %s
    """
    return run_query(conn, sql, (filter_db, filter_db, limit))


def get_lock_report(conn: psycopg2.extensions.connection, filter_db: Optional[str], limit: int) -> list[dict[str, Any]]:
    sql = """
    SELECT
        a.pid,
        a.datname,
        a.usename,
        COALESCE(NULLIF(a.application_name, ''), '(blank)') AS application_name,
        COALESCE(a.client_addr::text, 'local') AS client_addr,
        a.state,
        l.locktype,
        l.mode,
        l.granted,
        l.relation::regclass::text AS relation_name,
        a.query
    FROM pg_locks l
    JOIN pg_stat_activity a
        ON l.pid = a.pid
    WHERE (%s IS NULL OR a.datname = %s)
    ORDER BY a.pid, l.locktype, l.mode
    LIMIT %s
    """
    return run_query(conn, sql, (filter_db, filter_db, limit))


def normalize_interval_fields(rows: list[dict[str, Any]], keys: Iterable[str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        for key in keys:
            if key in row_copy:
                row_copy[key] = format_timedelta(row_copy[key])
        normalized.append(row_copy)
    return normalized


def print_server_info(info: dict[str, Any], filter_db: Optional[str], conn_params: dict[str, Any]) -> None:
    print_header("POSTGRESQL CONNECTION REPORT")
    print_kv("Report time", info.get("report_time"))
    print_kv("Connected database", info.get("current_database"))
    print_kv("Connected user", info.get("current_user"))
    print_kv("Server address", info.get("server_addr"))
    print_kv("Server port", info.get("server_port"))
    print_kv("Requested host", conn_params.get("host"))
    print_kv("Requested port", conn_params.get("port"))
    print_kv("Filter database", filter_db if filter_db else "(all databases)")
    print_kv("Server version", truncate(info.get("server_version"), 120))


def main() -> int:
    try:
        config = parse_args()
        conn_params = build_connection_params(config)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    conn: Optional[psycopg2.extensions.connection] = None

    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True

        server_info = get_server_info(conn)
        print_server_info(server_info, config.filter_db, conn_params)

        summary_rows = get_connection_summary(conn, config.filter_db)
        print_header("SUMMARY BY DATABASE / STATE")
        print_table(
            summary_rows,
            [
                ("datname", "Database"),
                ("state", "State"),
                ("connection_count", "Count"),
            ],
        )

        application_rows = get_application_summary(conn, config.filter_db)
        print_header("SUMMARY BY APPLICATION / USER / CLIENT / STATE")
        print_table(
            application_rows,
            [
                ("application_name", "Application"),
                ("usename", "User"),
                ("client_addr", "Client"),
                ("state", "State"),
                ("connection_count", "Count"),
            ],
        )

        detailed_rows = get_detailed_sessions(conn, config.filter_db, config.limit)
        detailed_rows = normalize_interval_fields(
            detailed_rows,
            ["connection_age", "transaction_age", "query_age"],
        )
        print_header(f"DETAILED SESSIONS (LIMIT {config.limit})")
        print_table(
            detailed_rows,
            [
                ("pid", "PID"),
                ("datname", "Database"),
                ("usename", "User"),
                ("application_name", "Application"),
                ("client_addr", "Client"),
                ("state", "State"),
                ("connection_age", "Conn Age"),
                ("transaction_age", "Txn Age"),
                ("query_age", "Query Age"),
                ("wait_event_type", "Wait Type"),
                ("wait_event", "Wait Event"),
                ("query", "Query"),
            ],
        )

        idle_rows = get_idle_sessions(conn, config.filter_db, config.limit)
        idle_rows = normalize_interval_fields(idle_rows, ["idle_for"])
        print_header(f"IDLE SESSIONS (LIMIT {config.limit})")
        print_table(
            idle_rows,
            [
                ("pid", "PID"),
                ("datname", "Database"),
                ("usename", "User"),
                ("application_name", "Application"),
                ("client_addr", "Client"),
                ("idle_for", "Idle For"),
                ("query", "Last Query"),
            ],
        )

        idle_tx_rows = get_idle_in_transaction_sessions(conn, config.filter_db, config.limit)
        idle_tx_rows = normalize_interval_fields(idle_tx_rows, ["transaction_age"])
        print_header(f"IDLE IN TRANSACTION (LIMIT {config.limit})")
        print_table(
            idle_tx_rows,
            [
                ("pid", "PID"),
                ("datname", "Database"),
                ("usename", "User"),
                ("application_name", "Application"),
                ("client_addr", "Client"),
                ("transaction_age", "Txn Age"),
                ("wait_event_type", "Wait Type"),
                ("wait_event", "Wait Event"),
                ("query", "Query"),
            ],
        )

        long_query_rows = get_long_running_queries(conn, config.filter_db, config.limit)
        long_query_rows = normalize_interval_fields(long_query_rows, ["query_age"])
        print_header(f"LONGEST RUNNING / OLDEST QUERIES (LIMIT {config.limit})")
        print_table(
            long_query_rows,
            [
                ("pid", "PID"),
                ("datname", "Database"),
                ("usename", "User"),
                ("application_name", "Application"),
                ("client_addr", "Client"),
                ("state", "State"),
                ("query_age", "Query Age"),
                ("wait_event_type", "Wait Type"),
                ("wait_event", "Wait Event"),
                ("query", "Query"),
            ],
        )

        if config.show_locks:
            lock_rows = get_lock_report(conn, config.filter_db, config.limit)
            print_header(f"LOCK REPORT (LIMIT {config.limit})")
            print_table(
                lock_rows,
                [
                    ("pid", "PID"),
                    ("datname", "Database"),
                    ("usename", "User"),
                    ("application_name", "Application"),
                    ("client_addr", "Client"),
                    ("state", "State"),
                    ("locktype", "Lock Type"),
                    ("mode", "Mode"),
                    ("granted", "Granted"),
                    ("relation_name", "Relation"),
                    ("query", "Query"),
                ],
            )

        print()
        print("Report complete.")
        return 0

    except psycopg2.Error as exc:
        print("PostgreSQL error:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print("Unexpected error:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 3
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())