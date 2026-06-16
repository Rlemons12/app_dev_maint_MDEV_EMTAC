#!/usr/bin/env python3
"""
postgres_connection_report.py

Connects to a PostgreSQL database and prints a clean connection report.

Features:
- Summary counts by database/state/application
- Detailed current session list
- Idle sessions
- Idle-in-transaction sessions
- Long-running connections
- Lock report

Usage examples:

    python postgres_connection_report.py \
        --host localhost \
        --port 5432 \
        --dbname postgres \
        --user postgres \
        --password yourpassword

    python postgres_connection_report.py \
        --dsn "host=localhost port=5432 dbname=postgres user=postgres password=yourpassword"

Optional:
    --filter-db your_database_name
    --limit 50
    --show-locks
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

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

    parser.add_argument("--dsn", help="Full PostgreSQL DSN string", default=None)
    parser.add_argument("--host", help="PostgreSQL host", default=os.getenv("PGHOST"))
    parser.add_argument(
        "--port",
        help="PostgreSQL port",
        type=int,
        default=int(os.getenv("PGPORT", "5432")),
    )
    parser.add_argument("--dbname", help="Database name", default=os.getenv("PGDATABASE"))
    parser.add_argument("--user", help="Database user", default=os.getenv("PGUSER"))
    parser.add_argument(
        "--password",
        help="Database password",
        default=os.getenv("PGPASSWORD"),
    )
    parser.add_argument(
        "--filter-db",
        help="Only report sessions for a specific database",
        default=None,
    )
    parser.add_argument(
        "--limit",
        help="Maximum rows shown in detail sections",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--show-locks",
        help="Include lock details",
        action="store_true",
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


def build_connection_params(config: Config) -> dict[str, Any]:
    """
    Build connection parameters for psycopg2.
    If DSN is provided, psycopg2 will use that directly.
    """
    if config.dsn:
        return {"dsn": config.dsn}

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
            + ". Provide them as arguments or environment variables."
        )

    return {
        "host": config.host,
        "port": config.port,
        "dbname": config.dbname,
        "user": config.user,
        "password": config.password,
    }


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def truncate(text: Any, max_len: int = 100) -> str:
    s = safe_str(text).replace("\n", " ").replace("\r", " ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def format_timedelta(value: Any) -> str:
    """
    PostgreSQL interval values come back as datetime.timedelta.
    """
    if value is None:
        return ""
    return str(value)


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_kv(label: str, value: Any) -> None:
    print(f"{label:<30}: {value}")


def print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], max_width: int = 40) -> None:
    """
    Print a simple fixed-width table.

    columns = [(key, display_name), ...]
    """
    if not rows:
        print("(none)")
        return

    widths: list[int] = []
    for key, display_name in columns:
        width = len(display_name)
        for row in rows:
            cell = row.get(key)
            cell_text = truncate(cell, max_width)
            width = min(max(width, len(cell_text)), max_width)
        widths.append(width)

    header = " | ".join(
        display_name.ljust(width)
        for (_, display_name), width in zip(columns, widths)
    )
    separator = "-+-".join("-" * width for width in widths)

    print(header)
    print(separator)

    for row in rows:
        line = " | ".join(
            truncate(row.get(key), max_width).ljust(width)
            for (key, _), width in zip(columns, widths)
        )
        print(line)


def fetch_all_dicts(cursor: psycopg2.extensions.cursor) -> list[dict[str, Any]]:
    rows = cursor.fetchall()
    return [dict(row) for row in rows]


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
    rows = run_query(conn, sql)
    return rows[0]


def get_connection_summary(conn: psycopg2.extensions.connection, filter_db: Optional[str]) -> list[dict[str, Any]]:
    sql = """
    SELECT
        datname,
        state,
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
        state,
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


def print_server_info(info: dict[str, Any], filter_db: Optional[str]) -> None:
    print_header("POSTGRESQL CONNECTION REPORT")
    print_kv("Report time", info.get("report_time"))
    print_kv("Connected database", info.get("current_database"))
    print_kv("Connected user", info.get("current_user"))
    print_kv("Server address", info.get("server_addr"))
    print_kv("Server port", info.get("server_port"))
    print_kv("Filter database", filter_db if filter_db else "(all databases)")
    print_kv("Server version", truncate(info.get("server_version"), 120))


def print_summary_section(title: str, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    print_header(title)
    print_table(rows, columns)


def normalize_interval_fields(rows: list[dict[str, Any]], keys: Iterable[str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        copy_row = dict(row)
        for key in keys:
            if key in copy_row:
                copy_row[key] = format_timedelta(copy_row[key])
        normalized.append(copy_row)
    return normalized


def main() -> int:
    try:
        config = parse_args()
        conn_params = build_connection_params(config)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    conn: Optional[psycopg2.extensions.connection] = None

    try:
        if "dsn" in conn_params:
            conn = psycopg2.connect(conn_params["dsn"])
        else:
            conn = psycopg2.connect(**conn_params)

        conn.autocommit = True

        server_info = get_server_info(conn)
        print_server_info(server_info, config.filter_db)

        summary_rows = get_connection_summary(conn, config.filter_db)
        print_summary_section(
            "SUMMARY BY DATABASE / STATE",
            summary_rows,
            [
                ("datname", "Database"),
                ("state", "State"),
                ("connection_count", "Count"),
            ],
        )

        application_rows = get_application_summary(conn, config.filter_db)
        print_summary_section(
            "SUMMARY BY APPLICATION / USER / CLIENT / STATE",
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
        print_summary_section(
            f"DETAILED SESSIONS (LIMIT {config.limit})",
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
        print_summary_section(
            f"IDLE SESSIONS (LIMIT {config.limit})",
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
        print_summary_section(
            f"IDLE IN TRANSACTION (LIMIT {config.limit})",
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
        print_summary_section(
            f"LONGEST RUNNING / OLDEST QUERIES (LIMIT {config.limit})",
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
            print_summary_section(
                f"LOCK REPORT (LIMIT {config.limit})",
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())