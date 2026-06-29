import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

log = logging.getLogger(__name__)


def get_test_db_url() -> str:
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "TEST_DATABASE_URL not set.\n"
            "Example:\n"
            "  postgresql+psycopg2://postgres:emtac123@127.0.0.1:5432/emtac_test"
        )
    return url


def make_engine():
    engine = create_engine(
        get_test_db_url(),
        future=True,
        pool_pre_ping=True,
    )
    return engine


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def truncate_all_tables(engine, schema: str = "public"):
    """
    Reset DB between tests.
    TRUNCATE .. CASCADE is the most reliable way to ensure clean state.
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = :schema
                """
            ),
            {"schema": schema},
        ).fetchall()

        tables = [r[0] for r in rows]
        if not tables:
            log.warning("No tables found in schema=%s to truncate.", schema)
            return

        joined = ", ".join(f'{schema}."{t}"' for t in tables)
        conn.execute(text(f"TRUNCATE {joined} RESTART IDENTITY CASCADE;"))
        log.debug("Truncated %d tables in schema=%s", len(tables), schema)
