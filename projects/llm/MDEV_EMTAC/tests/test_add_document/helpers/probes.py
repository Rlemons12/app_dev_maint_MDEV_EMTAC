from sqlalchemy import text


def count_rows(session, table_name: str, schema: str = "public") -> int:
    q = text(f'SELECT COUNT(*) FROM {schema}."{table_name}"')
    return int(session.execute(q).scalar() or 0)


def snapshot_counts(session, table_names, schema: str = "public"):
    return {t: count_rows(session, t, schema=schema) for t in table_names}
