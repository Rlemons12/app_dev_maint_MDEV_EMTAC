"""
test_config.db_runtime

Reusable database bootstrap + reset utilities
for EMTAC test / stress / pipeline environments.
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)


# ==========================================================
# ENGINE CREATION
# ==========================================================

def create_engine_from_env(echo: bool = False) -> Engine:
    """
    Create SQLAlchemy engine using DATABASE_URL.

    - Forces failure if DATABASE_URL not set
    - Enables optional SQL echo
    - Uses sane pooling defaults for stress testing
    """

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")

    log.info(f"[DB_RUNTIME] Creating engine for {db_url}")

    return create_engine(
        db_url,
        echo=echo,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        future=True,
    )


# ==========================================================
# SCHEMA BOOTSTRAP
# ==========================================================

def bootstrap_schema(Base, engine: Engine | None = None) -> Engine:
    """
    Ensures all ORM tables exist.

    Safe to call multiple times.
    """

    engine = engine or create_engine_from_env()

    log.info("[DB_RUNTIME] Bootstrapping schema...")
    Base.metadata.create_all(engine)
    log.info("[DB_RUNTIME] Schema ready.")

    return engine


# ==========================================================
# SAFE POSTGRES TRUNCATION
# ==========================================================

def truncate_all_tables(engine: Engine, schema: str = "public") -> None:
    """
    Safe Postgres table reset.

    - Keeps schema
    - Resets identities
    - Cascades FK
    - Skips alembic_version
    - Handles empty DB safely
    """

    log.info("[DB_RUNTIME] Truncating all tables...")

    with engine.begin() as conn:

        # Verify Postgres
        dialect = engine.dialect.name
        if dialect != "postgresql":
            raise RuntimeError(
                f"truncate_all_tables only supports PostgreSQL (got {dialect})"
            )

        conn.execute(text("SET session_replication_role = replica;"))

        rows = conn.execute(
            text("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = :schema;
            """),
            {"schema": schema},
        ).fetchall()

        table_names = [r[0] for r in rows]
        table_names = [t for t in table_names if t != "alembic_version"]

        if not table_names:
            log.warning("[DB_RUNTIME] No tables found to truncate.")
        else:
            quoted = ", ".join([f'"{t}"' for t in table_names])
            conn.execute(
                text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE;")
            )
            log.info(f"[DB_RUNTIME] Truncated {len(table_names)} tables.")

        conn.execute(text("SET session_replication_role = DEFAULT;"))

    log.info("[DB_RUNTIME] Truncate complete.")


# ==========================================================
# OPTIONAL: FULL RESET (SCHEMA DROP)
# ==========================================================

def drop_all_tables(Base, engine: Engine | None = None):
    """
    Drop entire schema (use carefully).
    Useful for deep pipeline resets.
    """

    engine = engine or create_engine_from_env()
    log.warning("[DB_RUNTIME] Dropping all tables...")
    Base.metadata.drop_all(engine)
    log.warning("[DB_RUNTIME] All tables dropped.")
