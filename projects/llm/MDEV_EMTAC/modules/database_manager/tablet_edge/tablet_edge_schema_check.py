"""
Check the EMTAC Tablet Edge Agent PostgreSQL schema.

Run from project root:
    python -m modules.database_manager.tablet_edge.tablet_edge_schema_check
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


logger = logging.getLogger("tablet_edge_schema_check")


REQUIRED_TABLES = [
    "tablet_device",
    "tablet_network_event",
    "tablet_health_sample",
    "tablet_dropdown_cache_manifest",
    "tablet_sync_event",
    "tablet_offline_event",
    "tablet_app_log",
]

REQUIRED_INDEXES = [
    "idx_tablet_device_uid",
    "idx_tablet_device_last_seen",
    "idx_tablet_network_event_device_created",
    "idx_tablet_network_event_quality_created",
    "idx_tablet_network_event_type_created",
    "idx_tablet_health_sample_device_sampled",
    "idx_tablet_health_sample_quality_sampled",
    "idx_tablet_dropdown_cache_device",
    "idx_tablet_sync_event_device_started",
    "idx_tablet_sync_event_status",
    "idx_tablet_offline_event_status",
    "idx_tablet_offline_event_payload_gin",
    "idx_tablet_app_log_device_created",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _engine_from_database_config() -> Optional[Engine]:
    try:
        from modules.configuration.config_env import DatabaseConfig
    except Exception as exc:
        logger.warning("Could not import DatabaseConfig: %s", exc)
        return None

    try:
        db_config = DatabaseConfig()
    except Exception as exc:
        logger.warning("Could not initialize DatabaseConfig: %s", exc)
        return None

    for attr_name in ("engine", "db_engine", "postgres_engine"):
        engine = getattr(db_config, attr_name, None)
        if engine is not None:
            logger.info("Using DatabaseConfig.%s engine.", attr_name)
            return engine

    for method_name in ("get_engine", "create_engine", "get_postgres_engine"):
        method = getattr(db_config, method_name, None)
        if callable(method):
            try:
                engine = method()
                if engine is not None:
                    logger.info("Using DatabaseConfig.%s().", method_name)
                    return engine
            except Exception as exc:
                logger.warning("DatabaseConfig.%s() failed: %s", method_name, exc)

    for attr_name in (
        "DATABASE_URL",
        "database_url",
        "SQLALCHEMY_DATABASE_URI",
        "sqlalchemy_database_uri",
        "postgres_url",
    ):
        url = getattr(db_config, attr_name, None)
        if url:
            logger.info("Using DatabaseConfig.%s connection URL.", attr_name)
            return create_engine(url)

    return None


def get_engine() -> Engine:
    engine = _engine_from_database_config()

    if engine is not None:
        return engine

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "Could not create database engine. "
            "DatabaseConfig did not expose an engine/url and DATABASE_URL is not set."
        )

    logger.info("Using DATABASE_URL from environment.")
    return create_engine(database_url)


def check_schema() -> int:
    engine = get_engine()

    missing_items: list[str] = []

    with engine.connect() as conn:
        schema_exists = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.schemata
                    WHERE schema_name = 'tablet_edge'
                )
                """
            )
        ).scalar()

        if not schema_exists:
            missing_items.append("schema: tablet_edge")

        existing_tables = {
            row[0]
            for row in conn.execute(
                text(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'tablet_edge'
                    """
                )
            )
        }

        for table_name in REQUIRED_TABLES:
            if table_name not in existing_tables:
                missing_items.append(f"table: tablet_edge.{table_name}")

        existing_indexes = {
            row[0]
            for row in conn.execute(
                text(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = 'tablet_edge'
                    """
                )
            )
        }

        for index_name in REQUIRED_INDEXES:
            if index_name not in existing_indexes:
                missing_items.append(f"index: {index_name}")

    if missing_items:
        logger.error("tablet_edge schema check FAILED.")
        for item in missing_items:
            logger.error("Missing %s", item)
        return 1

    logger.info("tablet_edge schema check PASSED.")
    logger.info("Required tables found: %s", len(REQUIRED_TABLES))
    logger.info("Required indexes found: %s", len(REQUIRED_INDEXES))
    return 0


def main() -> None:
    configure_logging()
    raise SystemExit(check_schema())


if __name__ == "__main__":
    main()
