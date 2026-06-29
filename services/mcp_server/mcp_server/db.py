from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import psycopg
from psycopg.rows import dict_row
from psycopg import sql as pg_sql


class PostgresClient:
    def __init__(self, dsn: str, application_name: str) -> None:
        self.dsn = dsn
        self.application_name = application_name

    @contextmanager
    def connection(self) -> Generator[psycopg.Connection, None, None]:
        conn = psycopg.connect(
            self.dsn,
            row_factory=dict_row,
            application_name=self.application_name,
        )

        try:
            yield conn
        finally:
            conn.close()

    def fetch_all(
        self,
        query: str | pg_sql.Composable,
        params: tuple[Any, ...] | None = None,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                rows = cur.fetchall()
                return [dict(row) for row in rows]

    def execute(
        self,
        query: str | pg_sql.Composable,
        params: tuple[Any, ...] | None = None,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params or ())

                    result: dict[str, Any] = {
                        "status": "ok",
                        "rowcount": cur.rowcount,
                    }

                    if cur.description:
                        rows = cur.fetchall()
                        result["rows"] = [dict(row) for row in rows]
                        result["returned_rows"] = len(rows)

                    conn.commit()
                    return result

            except Exception:
                conn.rollback()
                raise

    def test_connection(self) -> dict[str, Any]:
        rows = self.fetch_all(
            """
            SELECT
                current_database() AS database_name,
                current_user AS current_user,
                inet_server_addr()::text AS server_address,
                inet_server_port() AS server_port,
                version() AS postgres_version
            """
        )

        return rows[0] if rows else {}
