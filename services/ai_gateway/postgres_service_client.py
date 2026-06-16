from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from pathlib import Path

# ---------------------------------------------------------
# Load .env
# ---------------------------------------------------------
ENV_PATH = Path(r"E:\emtac\dev_env\.env")
load_dotenv(dotenv_path=ENV_PATH)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER_READ_ONLY") or os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD_READ_ONLY") or os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "public")
POSTGRES_TIMEOUT = int(os.getenv("POSTGRES_TIMEOUT", "10"))
POSTGRES_MAX_ROWS = int(os.getenv("POSTGRES_MAX_ROWS", "100"))
POSTGRES_MAX_TABLES = int(os.getenv("POSTGRES_MAX_TABLES", "20"))

POSTGRES_SYSTEM_PROMPT = os.getenv(
    "MCP_POSTGRES_SYSTEM_PROMPT",
    (
        "You are a database assistant with access to a PostgreSQL schema context.\n"
        "Answer questions about tables, columns, relationships, and data.\n"
        "When writing SQL, target PostgreSQL syntax.\n"
        "Be concise and specific. Reference table/column names exactly as they appear.\n"
        "If asked to query data, produce only the SQL — no explanation unless asked.\n"
        "Do not greet the user.\n"
    ),
)

POSTGRES_KEYWORDS = {
    "postgres", "postgresql", "database", "table", "column", "schema",
    "query", "sql", "select", "insert", "update", "delete", "join",
    "index", "primary key", "foreign key", "constraint", "view",
    "migration", "row", "record", "relation", "db",
}


# ---------------------------------------------------------
# Client
# ---------------------------------------------------------
class PostgresServiceClient:
    """
    Provides schema introspection and safe read-only query execution
    against a PostgreSQL instance to enrich GPU prompt requests.

    Public methods:
        health()              → bool
        get_tables()          → list[{name, schema, row_estimate, comment}]
        get_table_detail()    → {columns, indexes, foreign_keys, comment}
        execute_query()       → list[dict]  (SELECT only, row-capped)
        build_context()       → formatted string ready for prompt injection
    """

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        schema: str = "public",
        timeout: int = 10,
        max_rows: int = 100,
        max_tables: int = 20,
    ) -> None:
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.schema = schema
        self.timeout = timeout
        self.max_rows = max_rows
        self.max_tables = max_tables

        self._dsn = (
            f"host={host} port={port} dbname={dbname} "
            f"user={user} password={password} "
            f"connect_timeout={timeout}"
        )

        logger.info(
            "PostgresServiceClient initialized | host=%s:%s | db=%s | schema=%s | max_rows=%s | max_tables=%s",
            self.host,
            self.port,
            self.dbname,
            self.schema,
            self.max_rows,
            self.max_tables,
        )

    # ---- internal --------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[psycopg2.extensions.connection, None, None]:
        conn = psycopg2.connect(self._dsn)
        try:
            yield conn
        finally:
            conn.close()

    def _fetch(
        self,
        sql: str,
        params: Optional[tuple] = None,
        *,
        read_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL statement and return rows as a list of dicts.
        When read_only=True (default) the connection is opened in
        READ ONLY transaction mode, preventing any writes.
        """
        with self._connect() as conn:
            if read_only:
                conn.set_session(readonly=True, autocommit=True)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params or ())
                rows = cur.fetchmany(self.max_rows)
                return [dict(row) for row in rows]

    # ---- public ----------------------------------------------------------

    def health(self) -> bool:
        try:
            self._fetch("SELECT 1 AS alive")
            return True
        except Exception as exc:
            logger.warning("Postgres health check failed: %s", exc)
            return False

    def get_tables(self, query: str = "") -> List[Dict[str, Any]]:
        """
        Return table summaries for the configured schema.
        Optionally filter by a plain-text substring match on table name.
        """
        try:
            sql = """
                SELECT
                    t.table_name            AS name,
                    t.table_schema          AS schema,
                    COALESCE(s.n_live_tup, 0) AS row_estimate,
                    obj_description(
                        (quote_ident(t.table_schema) || '.' || quote_ident(t.table_name))::regclass,
                        'pg_class'
                    ) AS comment
                FROM information_schema.tables t
                LEFT JOIN pg_stat_user_tables s
                       ON s.schemaname = t.table_schema
                      AND s.relname    = t.table_name
                WHERE t.table_schema = %s
                  AND t.table_type   = 'BASE TABLE'
                ORDER BY t.table_name
                LIMIT %s
            """
            rows = self._fetch(sql, (self.schema, self.max_tables))

            if query:
                q = query.lower()
                rows = [r for r in rows if q in r["name"].lower()]

            return rows

        except Exception as exc:
            logger.warning("get_tables failed: %s", exc)
            return []

    def get_table_detail(self, table_name: str) -> Dict[str, Any]:
        """
        Return column definitions, indexes, and foreign keys for one table.
        """
        try:
            # Columns
            col_sql = """
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name   = %s
                ORDER BY ordinal_position
            """
            columns = self._fetch(col_sql, (self.schema, table_name))

            # Indexes
            idx_sql = """
                SELECT
                    indexname  AS name,
                    indexdef   AS definition
                FROM pg_indexes
                WHERE schemaname = %s
                  AND tablename  = %s
                ORDER BY indexname
            """
            indexes = self._fetch(idx_sql, (self.schema, table_name))

            # Foreign keys
            fk_sql = """
                SELECT
                    kcu.column_name,
                    ccu.table_name   AS foreign_table,
                    ccu.column_name  AS foreign_column,
                    rc.constraint_name
                FROM information_schema.table_constraints       tc
                JOIN information_schema.key_column_usage        kcu
                  ON kcu.constraint_name = tc.constraint_name
                 AND kcu.table_schema    = tc.table_schema
                JOIN information_schema.referential_constraints rc
                  ON rc.constraint_name = tc.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                  ON ccu.constraint_name = rc.unique_constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema    = %s
                  AND tc.table_name      = %s
            """
            foreign_keys = self._fetch(fk_sql, (self.schema, table_name))

            # Table-level comment
            comment_sql = """
                SELECT obj_description(
                    (quote_ident(%s) || '.' || quote_ident(%s))::regclass,
                    'pg_class'
                ) AS comment
            """
            comment_rows = self._fetch(comment_sql, (self.schema, table_name))
            comment = comment_rows[0]["comment"] if comment_rows else None

            return {
                "table": table_name,
                "schema": self.schema,
                "comment": comment,
                "columns": [dict(c) for c in columns],
                "indexes": [dict(i) for i in indexes],
                "foreign_keys": [dict(f) for f in foreign_keys],
            }

        except Exception as exc:
            logger.warning("get_table_detail(%s) failed: %s", table_name, exc)
            return {}

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """
        Run an arbitrary SQL statement in a read-only transaction.
        Raises ValueError for non-SELECT statements as a safety guard.
        """
        normalised = sql.strip().lstrip(";").strip().upper()
        if not normalised.startswith("SELECT"):
            raise ValueError(
                "execute_query only permits SELECT statements. "
                f"Got: {sql[:80]!r}"
            )

        try:
            logger.info(
                "Executing read-only query | chars=%s | preview=%r",
                len(sql),
                sql[:200],
            )
            rows = self._fetch(sql, read_only=True)
            logger.info("Query returned %s rows", len(rows))
            return rows

        except Exception as exc:
            logger.warning("execute_query failed: %s", exc)
            raise

    def build_context(self, user_query: str = "") -> str:
        """
        Assemble a [POSTGRES CONTEXT] block for prompt injection.
        Uses user_query to filter tables for relevance.
        """
        sections: List[str] = [
            f"[POSTGRES CONTEXT]\nDatabase: {self.dbname} | Schema: {self.schema} | Host: {self.host}:{self.port}"
        ]

        tables = self.get_tables(query=user_query)
        if tables:
            table_lines = []
            for t in tables:
                comment_part = f" — {t['comment']}" if t.get("comment") else ""
                table_lines.append(
                    f"  - {t['schema']}.{t['name']}"
                    f" (~{t['row_estimate']:,} rows){comment_part}"
                )
            sections.append("Tables:\n" + "\n".join(table_lines))

            # Include full detail when a single table matched
            if len(tables) == 1:
                detail = self.get_table_detail(tables[0]["name"])
                if detail.get("columns"):
                    col_lines = []
                    for c in detail["columns"]:
                        nullable = "" if c["is_nullable"] == "YES" else " NOT NULL"
                        default = f" DEFAULT {c['column_default']}" if c.get("column_default") else ""
                        max_len = f"({c['character_maximum_length']})" if c.get("character_maximum_length") else ""
                        col_lines.append(
                            f"    {c['column_name']}  {c['data_type']}{max_len}{nullable}{default}"
                        )
                    sections.append(
                        f"Columns in '{detail['table']}':\n" + "\n".join(col_lines)
                    )

                if detail.get("foreign_keys"):
                    fk_lines = [
                        f"  - {fk['column_name']} → {fk['foreign_table']}.{fk['foreign_column']}"
                        for fk in detail["foreign_keys"]
                    ]
                    sections.append("Foreign Keys:\n" + "\n".join(fk_lines))

                if detail.get("indexes"):
                    idx_lines = [
                        f"  - {idx['name']}: {idx['definition']}"
                        for idx in detail["indexes"]
                    ]
                    sections.append("Indexes:\n" + "\n".join(idx_lines))

        else:
            sections.append("Tables: none found")

        return "\n\n".join(sections) + "\n[END POSTGRES CONTEXT]"


# ---------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------
postgres_service_client = PostgresServiceClient(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    schema=POSTGRES_SCHEMA,
    timeout=POSTGRES_TIMEOUT,
    max_rows=POSTGRES_MAX_ROWS,
    max_tables=POSTGRES_MAX_TABLES,
)