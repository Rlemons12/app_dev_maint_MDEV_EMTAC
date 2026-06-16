"""
Create or update the EMTAC Tablet Edge Agent PostgreSQL schema.

File:
    modules/database_manager/tablet_edge/create_tablet_edge_schema.py

Run from project root:
    python -m modules.database_manager.tablet_edge.create_tablet_edge_schema

Notes:
    This script tries to use the existing EMTAC DatabaseConfig first.
    If that fails, it falls back to DATABASE_URL from the environment.

    The SQL file is expected at:

        modules/database_manager/tablet_edge/create_tablet_edge_schema.sql

    This Python runner does not define the table structure itself. It executes
    the SQL file and then validates that the expected tablet_edge tables/columns
    exist.

    Tablet app update validation added:
        - tablet_edge.tablet_device.app_version_code
        - tablet_edge.tablet_app_release
        - tablet_edge.v_latest_tablet_app_update_status

    To skip validation temporarily:

        set TABLET_EDGE_SKIP_VALIDATION=1

    To override the SQL file path:

        set TABLET_EDGE_SQL_FILE=E:\\path\\to\\create_tablet_edge_schema.sql
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


LOGGER_NAME = "tablet_edge_schema_setup"
logger = logging.getLogger(LOGGER_NAME)

TABLET_EDGE_SCHEMA = "tablet_edge"


EXPECTED_TABLES: tuple[str, ...] = (
    "tablet_device",
    "tablet_app_release",
    "tablet_wifi_access_point",
    "tablet_wifi_observation",
    "tablet_network_event",
    "tablet_health_sample",
    "tablet_dropdown_cache_manifest",
    "tablet_sync_event",
    "tablet_offline_event",
    "tablet_app_log",
)


EXPECTED_VIEWS: tuple[str, ...] = (
    "v_latest_tablet_wifi_status",
    "v_latest_tablet_health_status",
    "v_latest_tablet_app_update_status",
)


EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "tablet_device": (
        "id",
        "tablet_uid",
        "tablet_name",
        "device_make",
        "device_model",
        "android_version",
        "app_version",
        "app_version_code",
        "assigned_area",
        "assigned_station",
        "assigned_role",
        "is_active",
        "first_seen_at",
        "last_seen_at",
        "created_at",
        "updated_at",
    ),
    "tablet_app_release": (
        "id",
        "app_package",
        "release_channel",
        "version_name",
        "version_code",
        "apk_filename",
        "apk_file_path",
        "apk_sha256",
        "apk_size_bytes",
        "release_notes",
        "min_supported_version_code",
        "max_supported_version_code",
        "is_active",
        "is_required",
        "rollout_percent",
        "created_by",
        "published_at",
        "retired_at",
        "created_at",
        "updated_at",
    ),
    "tablet_wifi_access_point": (
        "id",
        "ssid",
        "bssid",
        "router_ip",
        "router_name",
        "friendly_name",
        "assigned_area",
        "assigned_station",
        "physical_location",
        "is_approved",
        "notes",
        "first_seen_at",
        "last_seen_at",
        "created_at",
        "updated_at",
    ),
    "tablet_wifi_observation": (
        "id",
        "tablet_device_id",
        "access_point_id",
        "sampled_at",
        "is_online",
        "ssid",
        "bssid",
        "router_ip",
        "router_name",
        "ip_address",
        "gateway_address",
        "dhcp_server_address",
        "dns_servers",
        "wifi_rssi",
        "signal_level",
        "frequency_mhz",
        "wifi_band",
        "link_speed_mbps",
        "server_url",
        "server_reachable",
        "server_latency_ms",
        "quality_level",
        "created_at",
    ),
    "tablet_network_event": (
        "id",
        "tablet_device_id",
        "access_point_id",
        "event_type",
        "quality_level",
        "server_url",
        "page_url",
        "latency_ms",
        "avg_latency_ms",
        "consecutive_failures",
        "is_online",
        "ssid",
        "bssid",
        "router_ip",
        "router_name",
        "wifi_rssi",
        "signal_level",
        "ip_address",
        "gateway_address",
        "dhcp_server_address",
        "dns_servers",
        "frequency_mhz",
        "wifi_band",
        "link_speed_mbps",
        "message",
        "event_started_at",
        "event_ended_at",
        "created_at",
    ),
    "tablet_health_sample": (
        "id",
        "tablet_device_id",
        "access_point_id",
        "sampled_at",
        "server_reachable",
        "server_latency_ms",
        "quality_level",
        "battery_percent",
        "is_charging",
        "ssid",
        "bssid",
        "router_ip",
        "router_name",
        "wifi_rssi",
        "signal_level",
        "ip_address",
        "gateway_address",
        "frequency_mhz",
        "wifi_band",
        "link_speed_mbps",
        "app_foreground",
        "current_page_url",
        "created_at",
    ),
    "tablet_dropdown_cache_manifest": (
        "id",
        "tablet_device_id",
        "cache_name",
        "cache_version",
        "record_count",
        "last_full_sync_at",
        "last_delta_sync_at",
        "sync_status",
        "created_at",
        "updated_at",
    ),
    "tablet_sync_event": (
        "id",
        "tablet_device_id",
        "sync_type",
        "sync_direction",
        "status",
        "records_sent",
        "records_received",
        "records_failed",
        "started_at",
        "completed_at",
        "duration_ms",
        "error_message",
        "created_at",
    ),
    "tablet_offline_event": (
        "id",
        "tablet_device_id",
        "local_event_id",
        "event_type",
        "event_payload",
        "client_created_at",
        "server_received_at",
        "processing_status",
        "processed_at",
        "error_message",
        "created_at",
    ),
    "tablet_app_log": (
        "id",
        "tablet_device_id",
        "log_level",
        "log_source",
        "message",
        "context",
        "client_created_at",
        "server_received_at",
        "created_at",
    ),
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_project_root_on_path() -> None:
    project_root = get_project_root()
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def load_env_file_if_available() -> None:
    """
    Load the project .env file when python-dotenv is available.

    This keeps the script friendly when run directly from PowerShell and avoids
    failing when python-dotenv is not installed.
    """
    project_root = get_project_root()
    env_path = project_root / ".env"

    try:
        from dotenv import load_dotenv
    except Exception:
        logger.info("python-dotenv is not available. Skipping .env loading.")
        return

    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded environment from %s", env_path)
    else:
        load_dotenv()
        logger.info("No project .env found at %s. Called load_dotenv() anyway.", env_path)


def get_sql_path() -> Path:
    override_path = os.getenv("TABLET_EDGE_SQL_FILE")

    if override_path:
        return Path(override_path).expanduser().resolve()

    return Path(__file__).resolve().parent / "create_tablet_edge_schema.sql"


def _engine_from_database_config() -> Optional[Engine]:
    """
    Best-effort adapter for EMTAC DatabaseConfig.

    This avoids hard-coding one exact method name because DatabaseConfig has
    changed across project versions.
    """
    ensure_project_root_on_path()

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

    candidate_attrs = (
        "engine",
        "db_engine",
        "postgres_engine",
        "sqlalchemy_engine",
    )

    for attr_name in candidate_attrs:
        engine = getattr(db_config, attr_name, None)

        if engine is not None:
            logger.info("Using DatabaseConfig.%s engine.", attr_name)
            return engine

    candidate_methods = (
        "get_engine",
        "create_engine",
        "get_postgres_engine",
        "get_sqlalchemy_engine",
    )

    for method_name in candidate_methods:
        method = getattr(db_config, method_name, None)

        if callable(method):
            try:
                engine = method()

                if engine is not None:
                    logger.info("Using DatabaseConfig.%s().", method_name)
                    return engine

            except Exception as exc:
                logger.warning("DatabaseConfig.%s() failed: %s", method_name, exc)

    url_attrs = (
        "DATABASE_URL",
        "database_url",
        "SQLALCHEMY_DATABASE_URI",
        "sqlalchemy_database_uri",
        "postgres_url",
        "POSTGRES_URL",
    )

    for attr_name in url_attrs:
        url = getattr(db_config, attr_name, None)

        if url:
            logger.info("Using DatabaseConfig.%s connection URL.", attr_name)
            return create_engine(url)

    logger.warning("Could not resolve engine from DatabaseConfig.")
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


def read_sql_file(sql_path: Path) -> str:
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8-sig").lstrip("\ufeff").strip()

    if not sql_text:
        raise RuntimeError(f"SQL file is empty: {sql_path}")

    return sql_text


def validate_connection(engine: Engine) -> None:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();")).scalar_one_or_none()

    logger.info("Database connection validated.")
    logger.info("PostgreSQL version: %s", result)


def run_sql(engine: Engine, sql_text: str) -> None:
    """
    Execute the schema SQL.

    exec_driver_sql is used instead of wrapping the whole file in text(...)
    because this is a raw SQL script that may contain multiple PostgreSQL DDL
    statements.
    """
    with engine.begin() as conn:
        conn.exec_driver_sql(sql_text)


def _schema_exists(engine: Engine) -> bool:
    query = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.schemata
            WHERE schema_name = :schema_name
        );
        """
    )

    with engine.connect() as conn:
        return bool(conn.execute(query, {"schema_name": TABLET_EDGE_SCHEMA}).scalar())


def _get_existing_tables(engine: Engine) -> set[str]:
    query = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema_name
          AND table_type = 'BASE TABLE';
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, {"schema_name": TABLET_EDGE_SCHEMA}).fetchall()

    return {str(row[0]) for row in rows}


def _get_existing_views(engine: Engine) -> set[str]:
    query = text(
        """
        SELECT table_name
        FROM information_schema.views
        WHERE table_schema = :schema_name;
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, {"schema_name": TABLET_EDGE_SCHEMA}).fetchall()

    return {str(row[0]) for row in rows}


def _get_existing_columns(engine: Engine, table_name: str) -> set[str]:
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema_name
          AND table_name = :table_name;
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {
                "schema_name": TABLET_EDGE_SCHEMA,
                "table_name": table_name,
            },
        ).fetchall()

    return {str(row[0]) for row in rows}


def validate_schema(engine: Engine) -> None:
    """
    Validate that the SQL file created/updated the expected schema structure.

    This catches the common issue where the ORM models were updated but the SQL
    schema file was not updated yet.
    """
    if os.getenv("TABLET_EDGE_SKIP_VALIDATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        logger.warning("TABLET_EDGE_SKIP_VALIDATION is enabled. Skipping schema validation.")
        return

    logger.info("Validating tablet_edge schema.")

    if not _schema_exists(engine):
        raise RuntimeError(f"Schema does not exist: {TABLET_EDGE_SCHEMA}")

    existing_tables = _get_existing_tables(engine)
    missing_tables = sorted(set(EXPECTED_TABLES) - existing_tables)

    if missing_tables:
        raise RuntimeError(
            "tablet_edge schema validation failed. Missing tables: "
            + ", ".join(missing_tables)
        )

    existing_views = _get_existing_views(engine)
    missing_views = sorted(set(EXPECTED_VIEWS) - existing_views)

    if missing_views:
        raise RuntimeError(
            "tablet_edge schema validation failed. Missing views: "
            + ", ".join(missing_views)
        )

    missing_column_messages: list[str] = []

    for table_name, expected_columns in EXPECTED_COLUMNS.items():
        existing_columns = _get_existing_columns(engine, table_name)
        missing_columns = sorted(set(expected_columns) - existing_columns)

        if missing_columns:
            missing_column_messages.append(
                f"{TABLET_EDGE_SCHEMA}.{table_name}: {', '.join(missing_columns)}"
            )

    if missing_column_messages:
        raise RuntimeError(
            "tablet_edge schema validation failed. Missing columns:\n"
            + "\n".join(f"  - {message}" for message in missing_column_messages)
        )

    logger.info("tablet_edge schema validation passed.")


def run_schema_setup() -> None:
    ensure_project_root_on_path()
    load_env_file_if_available()

    sql_path = get_sql_path()
    sql_text = read_sql_file(sql_path)

    logger.info("Starting tablet_edge schema setup.")
    logger.info("Project root: %s", get_project_root())
    logger.info("SQL file: %s", sql_path)

    engine = get_engine()

    validate_connection(engine)

    logger.info("Executing tablet_edge schema SQL.")
    run_sql(engine, sql_text)

    validate_schema(engine)

    logger.info("tablet_edge schema setup completed successfully.")


def main() -> None:
    configure_logging()
    run_schema_setup()


if __name__ == "__main__":
    main()