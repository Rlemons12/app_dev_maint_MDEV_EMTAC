#!/usr/bin/env python3
"""
postgres_live_report.py

Live PostgreSQL connection report/dashboard.

Features:
- Auto-refresh terminal dashboard
- Reads connection settings from .env
- Supports:
    POSTGRES_USER
    POSTGRES_PASSWORD
    POSTGRES_HOST
    POSTGRES_PORT
    POSTGRES_DB
    DATABASE_URL
- Shows:
    * overview summary
    * grouped connections
    * active sessions
    * idle sessions
    * idle in transaction
    * longest-running queries
    * lock summary
- Flags suspicious sessions

Run examples:
    python postgres_live_report.py
    python postgres_live_report.py --filter-db emtac
    python postgres_live_report.py --refresh 2
    python postgres_live_report.py --limit 20
    python postgres_live_report.py --show-locks
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
from psycopg2 import OperationalError
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


load_dotenv()

console = Console()


@dataclass
class Config:
    dsn: Optional[str]
    host: Optional[str]
    port: int
    dbname: Optional[str]
    user: Optional[str]
    password: Optional[str]
    filter_db: Optional[str]
    refresh: float
    limit: int
    show_locks: bool
    application_name: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Live PostgreSQL connection report."
    )
    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL DSN / SQLAlchemy URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("POSTGRES_HOST"),
        help="Database host. Defaults to POSTGRES_HOST.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("POSTGRES_PORT", "5432")),
        help="Database port. Defaults to POSTGRES_PORT or 5432.",
    )
    parser.add_argument(
        "--dbname",
        default=os.getenv("POSTGRES_DB"),
        help="Database name. Defaults to POSTGRES_DB.",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("POSTGRES_USER"),
        help="Database user. Defaults to POSTGRES_USER.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("POSTGRES_PASSWORD"),
        help="Database password. Defaults to POSTGRES_PASSWORD.",
    )
    parser.add_argument(
        "--filter-db",
        default=None,
        help="Only show activity for one database.",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=3.0,
        help="Refresh interval in seconds. Default: 3.0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Max rows per detail table. Default: 15",
    )
    parser.add_argument(
        "--show-locks",
        action="store_true",
        help="Show lock detail table.",
    )
    parser.add_argument(
        "--application-name",
        default="postgres_live_report",
        help="Application name used for this monitor connection.",
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
        refresh=args.refresh,
        limit=args.limit,
        show_locks=args.show_locks,
        application_name=args.application_name,
    )


def normalize_sqlalchemy_url(url: str) -> str:
    url = url.strip()
    if url.startswith("postgresql+psycopg2://"):
        return "postgresql://" + url[len("postgresql+psycopg2://"):]
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


def parse_database_url(url: str) -> dict[str, Any]:
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
    if config.dsn:
        params = parse_database_url(config.dsn)
    else:
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
                + ". Set DATABASE_URL or POSTGRES_* values in your environment or .env."
            )

        params = {
            "host": config.host,
            "port": config.port,
            "dbname": config.dbname,
            "user": config.user,
            "password": config.password,
        }

    params["application_name"] = config.application_name
    return params


def get_connection(config: Config) -> psycopg2.extensions.connection:
    params = build_connection_params(config)
    conn = psycopg2.connect(**params)
    conn.autocommit = True
    return conn


def run_query(
    conn: psycopg2.extensions.connection,
    sql: str,
    params: tuple[Any, ...] | None = None,
) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params or ())
        return [dict(row) for row in cur.fetchall()]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def truncate(value: Any, max_len: int = 80) -> str:
    text = safe_str(value).replace("\n", " ").replace("\r", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def age_to_str(value: Any) -> str:
    return "" if value is None else str(value).split(".")[0]


def get_overview(conn: psycopg2.extensions.connection, filter_db: Optional[str]) -> dict[str, Any]:
    sql = """
    SELECT
        COUNT(*) FILTER (WHERE (%s IS NULL OR datname = %s)) AS total_connections,
        COUNT(*) FILTER (WHERE (%s IS NULL OR datname = %s) AND state = 'active') AS active_connections,
        COUNT(*) FILTER (WHERE (%s IS NULL OR datname = %s) AND state = 'idle') AS idle_connections,
        COUNT(*) FILTER (WHERE (%s IS NULL OR datname = %s) AND state = 'idle in transaction') AS idle_in_transaction_connections,
        COUNT(*) FILTER (WHERE (%s IS NULL OR datname = %s) AND wait_event IS NOT NULL) AS waiting_connections,
        COUNT(*) FILTER (WHERE (%s IS NULL OR datname = %s) AND application_name = 'postgres_live_report') AS monitor_connections
    FROM pg_stat_activity
    """
    row = run_query(
        conn,
        sql,
        (
            filter_db, filter_db,
            filter_db, filter_db,
            filter_db, filter_db,
            filter_db, filter_db,
            filter_db, filter_db,
            filter_db, filter_db,
        ),
    )[0]
    return row


def get_connection_groups(conn: psycopg2.extensions.connection, filter_db: Optional[str]) -> list[dict[str, Any]]:
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
    LIMIT 20
    """
    return run_query(conn, sql, (filter_db, filter_db))


def get_active_sessions(conn: psycopg2.extensions.connection, filter_db: Optional[str], limit: int) -> list[dict[str, Any]]:
    sql = """
    SELECT
        pid,
        datname,
        usename,
        COALESCE(NULLIF(application_name, ''), '(blank)') AS application_name,
        COALESCE(client_addr::text, 'local') AS client_addr,
        now() - query_start AS query_age,
        wait_event_type,
        wait_event,
        state,
        query
    FROM pg_stat_activity
    WHERE (%s IS NULL OR datname = %s)
      AND state = 'active'
    ORDER BY query_start ASC NULLS LAST
    LIMIT %s
    """
    rows = run_query(conn, sql, (filter_db, filter_db, limit))
    for row in rows:
        row["query_age"] = age_to_str(row["query_age"])
    return rows


def get_idle_sessions(conn: psycopg2.extensions.connection, filter_db: Optional[str], limit: int) -> list[dict[str, Any]]:
    sql = """
    SELECT
        pid,
        datname,
        usename,
        COALESCE(NULLIF(application_name, ''), '(blank)') AS application_name,
        COALESCE(client_addr::text, 'local') AS client_addr,
        now() - state_change AS idle_for,
        state,
        query
    FROM pg_stat_activity
    WHERE (%s IS NULL OR datname = %s)
      AND state = 'idle'
    ORDER BY state_change ASC NULLS LAST
    LIMIT %s
    """
    rows = run_query(conn, sql, (filter_db, filter_db, limit))
    for row in rows:
        row["idle_for"] = age_to_str(row["idle_for"])
    return rows


def get_idle_in_transaction(
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
        now() - xact_start AS transaction_age,
        wait_event_type,
        wait_event,
        state,
        query
    FROM pg_stat_activity
    WHERE (%s IS NULL OR datname = %s)
      AND state = 'idle in transaction'
    ORDER BY xact_start ASC NULLS LAST
    LIMIT %s
    """
    rows = run_query(conn, sql, (filter_db, filter_db, limit))
    for row in rows:
        row["transaction_age"] = age_to_str(row["transaction_age"])
    return rows


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
        now() - query_start AS query_age,
        state,
        query
    FROM pg_stat_activity
    WHERE (%s IS NULL OR datname = %s)
      AND query_start IS NOT NULL
    ORDER BY query_start ASC
    LIMIT %s
    """
    rows = run_query(conn, sql, (filter_db, filter_db, limit))
    for row in rows:
        row["query_age"] = age_to_str(row["query_age"])
    return rows


def get_lock_summary(conn: psycopg2.extensions.connection, filter_db: Optional[str]) -> list[dict[str, Any]]:
    sql = """
    SELECT
        l.locktype,
        l.mode,
        l.granted,
        COUNT(*) AS lock_count
    FROM pg_locks l
    JOIN pg_stat_activity a
        ON a.pid = l.pid
    WHERE (%s IS NULL OR a.datname = %s)
    GROUP BY l.locktype, l.mode, l.granted
    ORDER BY lock_count DESC, l.locktype, l.mode
    """
    return run_query(conn, sql, (filter_db, filter_db))


def get_lock_details(conn: psycopg2.extensions.connection, filter_db: Optional[str], limit: int) -> list[dict[str, Any]]:
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
        COALESCE(l.relation::regclass::text, '') AS relation_name,
        a.query
    FROM pg_locks l
    JOIN pg_stat_activity a
        ON a.pid = l.pid
    WHERE (%s IS NULL OR a.datname = %s)
    ORDER BY a.pid, l.locktype, l.mode
    LIMIT %s
    """
    return run_query(conn, sql, (filter_db, filter_db, limit))


def make_simple_table(title: str, columns: list[str], rows: list[list[Any]]) -> Table:
    table = Table(title=title, expand=True, show_lines=False)
    for col in columns:
        table.add_column(col, overflow="fold")

    if not rows:
        table.add_row(*(["(none)"] + [""] * (len(columns) - 1)))
        return table

    for row in rows:
        table.add_row(*[safe_str(x) for x in row])

    return table


def state_style(state: str) -> str:
    state_lower = state.lower()
    if state_lower == "active":
        return "bold green"
    if state_lower == "idle in transaction":
        return "bold red"
    if state_lower == "idle":
        return "yellow"
    return "white"


def suspicious_style(age_text: str, threshold_seconds: int) -> str:
    """
    Very simple threshold based on parsed HH:MM:SS-like strings.
    """
    try:
        parts = age_text.split(":")
        if len(parts) < 3:
            return ""
        hours = int(parts[-3].split(" ")[-1])
        minutes = int(parts[-2])
        seconds = int(parts[-1])
        total = hours * 3600 + minutes * 60 + seconds
        if total >= threshold_seconds:
            return "bold red"
    except Exception:
        return ""
    return ""


def build_overview_panel(
    overview: dict[str, Any],
    filter_db: Optional[str],
    refresh: float,
) -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="left")

    grid.add_row("Filter DB", filter_db or "(all databases)")
    grid.add_row("Refresh (sec)", str(refresh))
    grid.add_row("Total connections", str(overview["total_connections"]))
    grid.add_row("Active", f"[green]{overview['active_connections']}[/green]")
    grid.add_row("Idle", f"[yellow]{overview['idle_connections']}[/yellow]")
    grid.add_row(
        "Idle in transaction",
        f"[bold red]{overview['idle_in_transaction_connections']}[/bold red]",
    )
    grid.add_row("Waiting", str(overview["waiting_connections"]))
    grid.add_row("Monitor sessions", str(overview["monitor_connections"]))

    return Panel(grid, title="Overview", border_style="cyan")


def build_group_table(groups: list[dict[str, Any]]) -> Table:
    table = Table(title="Connection Groups", expand=True)
    table.add_column("Application", overflow="fold")
    table.add_column("User")
    table.add_column("Client")
    table.add_column("State")
    table.add_column("Count", justify="right")

    if not groups:
        table.add_row("(none)", "", "", "", "")
        return table

    for row in groups:
        table.add_row(
            truncate(row["application_name"], 30),
            safe_str(row["usename"]),
            safe_str(row["client_addr"]),
            f"[{state_style(safe_str(row['state']))}]{safe_str(row['state'])}[/{state_style(safe_str(row['state']))}]",
            str(row["connection_count"]),
        )
    return table


def build_session_table(title: str, rows: list[dict[str, Any]], mode: str) -> Table:
    table = Table(title=title, expand=True, show_lines=False)
    table.add_column("PID", justify="right")
    table.add_column("DB")
    table.add_column("User")
    table.add_column("App", overflow="fold")
    table.add_column("Client")
    table.add_column("Age")
    table.add_column("State")
    table.add_column("Query", overflow="fold")

    if not rows:
        table.add_row("(none)", "", "", "", "", "", "", "")
        return table

    for row in rows:
        if mode == "active":
            age = row.get("query_age", "")
        elif mode == "idle":
            age = row.get("idle_for", "")
        else:
            age = row.get("transaction_age", "")

        age_style = suspicious_style(safe_str(age), 300)
        state = safe_str(row.get("state", ""))
        state_fmt = f"[{state_style(state)}]{state}[/{state_style(state)}]"
        age_fmt = f"[{age_style}]{age}[/{age_style}]" if age_style else safe_str(age)

        table.add_row(
            safe_str(row.get("pid")),
            safe_str(row.get("datname")),
            safe_str(row.get("usename")),
            truncate(row.get("application_name"), 24),
            safe_str(row.get("client_addr")),
            age_fmt,
            state_fmt,
            truncate(row.get("query"), 95),
        )
    return table


def build_long_query_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(title="Oldest Queries", expand=True)
    table.add_column("PID", justify="right")
    table.add_column("DB")
    table.add_column("User")
    table.add_column("App", overflow="fold")
    table.add_column("Age")
    table.add_column("State")
    table.add_column("Query", overflow="fold")

    if not rows:
        table.add_row("(none)", "", "", "", "", "", "")
        return table

    for row in rows:
        age = safe_str(row.get("query_age", ""))
        age_style = suspicious_style(age, 300)
        age_fmt = f"[{age_style}]{age}[/{age_style}]" if age_style else age
        state = safe_str(row.get("state", ""))

        table.add_row(
            safe_str(row.get("pid")),
            safe_str(row.get("datname")),
            safe_str(row.get("usename")),
            truncate(row.get("application_name"), 26),
            age_fmt,
            f"[{state_style(state)}]{state}[/{state_style(state)}]",
            truncate(row.get("query"), 100),
        )
    return table


def build_lock_summary_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(title="Lock Summary", expand=True)
    table.add_column("Lock Type")
    table.add_column("Mode")
    table.add_column("Granted")
    table.add_column("Count", justify="right")

    if not rows:
        table.add_row("(none)", "", "", "")
        return table

    for row in rows:
        granted = "yes" if row["granted"] else "no"
        granted_style = "green" if row["granted"] else "red"
        table.add_row(
            safe_str(row["locktype"]),
            safe_str(row["mode"]),
            f"[{granted_style}]{granted}[/{granted_style}]",
            str(row["lock_count"]),
        )
    return table


def build_lock_detail_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(title="Lock Details", expand=True)
    table.add_column("PID", justify="right")
    table.add_column("DB")
    table.add_column("User")
    table.add_column("App", overflow="fold")
    table.add_column("Lock Type")
    table.add_column("Mode")
    table.add_column("Granted")
    table.add_column("Relation", overflow="fold")
    table.add_column("Query", overflow="fold")

    if not rows:
        table.add_row("(none)", "", "", "", "", "", "", "", "")
        return table

    for row in rows:
        granted = "yes" if row["granted"] else "no"
        granted_style = "green" if row["granted"] else "red"

        table.add_row(
            safe_str(row["pid"]),
            safe_str(row["datname"]),
            safe_str(row["usename"]),
            truncate(row["application_name"], 20),
            safe_str(row["locktype"]),
            safe_str(row["mode"]),
            f"[{granted_style}]{granted}[/{granted_style}]",
            truncate(row["relation_name"], 25),
            truncate(row["query"], 80),
        )
    return table


def build_dashboard(
    conn: psycopg2.extensions.connection,
    config: Config,
) -> Group:
    overview = get_overview(conn, config.filter_db)
    groups = get_connection_groups(conn, config.filter_db)
    active = get_active_sessions(conn, config.filter_db, config.limit)
    idle = get_idle_sessions(conn, config.filter_db, config.limit)
    idle_tx = get_idle_in_transaction(conn, config.filter_db, config.limit)
    long_queries = get_long_running_queries(conn, config.filter_db, config.limit)
    lock_summary = get_lock_summary(conn, config.filter_db)

    header_text = Text("PostgreSQL Live Connection Report", style="bold cyan")
    subtitle = Text(
        "Press Ctrl+C to stop",
        style="dim",
    )

    components: list[Any] = [
        Align.center(header_text),
        Align.center(subtitle),
        build_overview_panel(overview, config.filter_db, config.refresh),
        build_group_table(groups),
        build_session_table("Active Sessions", active, mode="active"),
        build_session_table("Idle Sessions", idle, mode="idle"),
        build_session_table("Idle In Transaction", idle_tx, mode="idle_tx"),
        build_long_query_table(long_queries),
        build_lock_summary_table(lock_summary),
    ]

    if config.show_locks:
        lock_details = get_lock_details(conn, config.filter_db, config.limit)
        components.append(build_lock_detail_table(lock_details))

    return Group(*components)


def error_panel(message: str) -> Panel:
    return Panel(
        message,
        title="Connection / Query Error",
        border_style="red",
    )


def main() -> int:
    try:
        config = parse_args()
        build_connection_params(config)
    except Exception as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        return 1

    conn: Optional[psycopg2.extensions.connection] = None

    try:
        conn = get_connection(config)

        with Live(
            build_dashboard(conn, config),
            console=console,
            refresh_per_second=max(1, int(1 / config.refresh)) if config.refresh < 1 else 4,
            screen=True,
        ) as live:
            while True:
                try:
                    if conn.closed != 0:
                        conn = get_connection(config)

                    dashboard = build_dashboard(conn, config)
                    live.update(dashboard)
                    time.sleep(config.refresh)

                except OperationalError:
                    try:
                        if conn is not None:
                            conn.close()
                    except Exception:
                        pass

                    conn = None
                    live.update(error_panel("Lost PostgreSQL connection. Retrying..."))
                    time.sleep(config.refresh)

                    try:
                        conn = get_connection(config)
                    except Exception as reconnect_exc:
                        live.update(error_panel(f"Reconnect failed: {reconnect_exc}"))
                        time.sleep(config.refresh)

                except psycopg2.Error as db_exc:
                    live.update(error_panel(f"PostgreSQL error: {db_exc}"))
                    time.sleep(config.refresh)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Live report stopped by user.[/bold yellow]")
        return 0
    except Exception as exc:
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        return 2
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())