from __future__ import annotations

import re
from typing import Iterable

import sqlparse


READ_ALLOWED_STARTERS = {"select", "with", "show", "explain"}
READ_BLOCKED_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "call",
    "do",
    "merge",
    "vacuum",
    "analyze",
    "refresh",
    "reindex",
    "cluster",
    "comment",
}


def split_sql_statements(sql: str) -> list[str]:
    return [statement.strip() for statement in sqlparse.split(sql) if statement.strip()]


def first_token(sql: str) -> str:
    parsed = sqlparse.parse(sql)
    if not parsed:
        return ""
    for token in parsed[0].flatten():
        value = token.value.strip()
        if value:
            return value.lower()
    return ""


def contains_blocked_keyword(sql: str, blocked_keywords: Iterable[str]) -> bool:
    lowered = sql.lower()
    for keyword in blocked_keywords:
        pattern = rf"\b{re.escape(keyword)}\b"
        if re.search(pattern, lowered):
            return True
    return False


def validate_read_only_sql(sql: str) -> None:
    statements = split_sql_statements(sql)
    if not statements:
        raise ValueError("SQL is empty.")
    if len(statements) > 1:
        raise ValueError("Read-only tool accepts one SQL statement at a time.")

    statement = statements[0]
    starter = first_token(statement)
    if starter not in READ_ALLOWED_STARTERS:
        raise ValueError(
            "Read-only SQL must start with one of: "
            f"{', '.join(sorted(READ_ALLOWED_STARTERS))}."
        )
    if contains_blocked_keyword(statement, READ_BLOCKED_KEYWORDS):
        raise ValueError("Read-only SQL contains a blocked write/admin keyword.")


def ensure_limit(sql: str, max_rows: int) -> str:
    statement = sql.strip().rstrip(";")
    starter = first_token(statement)
    if starter not in {"select", "with"}:
        return statement
    if re.search(r"\blimit\b", statement, flags=re.IGNORECASE):
        return statement
    return f"{statement}\nLIMIT {int(max_rows)}"

