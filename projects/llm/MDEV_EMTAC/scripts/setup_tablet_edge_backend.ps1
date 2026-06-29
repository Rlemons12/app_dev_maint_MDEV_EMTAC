# ============================================================
# setup_tablet_edge_backend.ps1
#
# EMTAC Tablet Edge Agent backend setup scaffold
#
# Run from:
#   E:\emtac\projects\llm\MDEV_EMTAC
#
# What this creates:
#   - tablet_edge backend folders
#   - create_tablet_edge_schema.sql
#   - create_tablet_edge_schema.py
#   - tablet_edge_schema_check.py
#   - seed_tablet_edge_dev_data.py
#   - test_tablet_edge_routes.py
#   - reports/tablet_edge/.gitkeep
#
# Safe default:
#   Existing files are NOT overwritten unless -Overwrite is used.
#
# Usage:
#   .\setup_tablet_edge_backend.ps1
#
# Overwrite generated files:
#   .\setup_tablet_edge_backend.ps1 -Overwrite
# ============================================================

param(
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

function Write-SetupInfo {
    param([string]$Message)
    Write-Host "[TABLET_EDGE_SETUP] $Message" -ForegroundColor Cyan
}

function Write-SetupWarn {
    param([string]$Message)
    Write-Host "[TABLET_EDGE_SETUP][WARN] $Message" -ForegroundColor Yellow
}

function Write-SetupError {
    param([string]$Message)
    Write-Host "[TABLET_EDGE_SETUP][ERROR] $Message" -ForegroundColor Red
}

function New-DirectoryIfMissing {
    param([string]$Path)

    if (!(Test-Path $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
        Write-SetupInfo "Created directory: $Path"
    }
    else {
        Write-SetupInfo "Directory already exists: $Path"
    }
}

function Write-FileSafe {
    param(
        [string]$Path,
        [string]$Content
    )

    if ((Test-Path $Path) -and (-not $Overwrite)) {
        Write-SetupWarn "Skipped existing file: $Path"
        return
    }

    $Parent = Split-Path -Parent $Path

    if (!(Test-Path $Parent)) {
        New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    }

    Set-Content -Path $Path -Value $Content -Encoding UTF8
    Write-SetupInfo "Wrote file: $Path"
}

# ------------------------------------------------------------
# Validate project root
# ------------------------------------------------------------

$ProjectRoot = (Get-Location).Path

Write-SetupInfo "Project root: $ProjectRoot"

$ExpectedMarkers = @(
    "modules",
    "blueprints"
)

foreach ($Marker in $ExpectedMarkers) {
    $MarkerPath = Join-Path $ProjectRoot $Marker

    if (!(Test-Path $MarkerPath)) {
        Write-SetupError "Expected folder not found: $MarkerPath"
        Write-SetupError "Run this script from the MDEV_EMTAC project root."
        exit 1
    }
}

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

$BlueprintDir = Join-Path $ProjectRoot "blueprints"

$CoordinatorDir = Join-Path $ProjectRoot "modules\coordinators"
$OrchestratorDir = Join-Path $ProjectRoot "modules\orchestrators"
$ServiceDir = Join-Path $ProjectRoot "modules\services\tablet_edge"
$DbModelDir = Join-Path $ProjectRoot "modules\emtacdb"
$DbManagerDir = Join-Path $ProjectRoot "modules\database_manager\tablet_edge"

$StaticJsDir = Join-Path $ProjectRoot "static\js\module_template"
$StaticCssDir = Join-Path $ProjectRoot "static\css\module_template"

$ReportsDir = Join-Path $ProjectRoot "reports\tablet_edge"

# ------------------------------------------------------------
# Create directories
# ------------------------------------------------------------

$Directories = @(
    $BlueprintDir,
    $CoordinatorDir,
    $OrchestratorDir,
    $ServiceDir,
    $DbModelDir,
    $DbManagerDir,
    $StaticJsDir,
    $StaticCssDir,
    $ReportsDir
)

foreach ($Dir in $Directories) {
    New-DirectoryIfMissing $Dir
}

# ------------------------------------------------------------
# SQL schema
# ------------------------------------------------------------

$SchemaSqlPath = Join-Path $DbManagerDir "create_tablet_edge_schema.sql"

$SchemaSql = @'
-- ============================================================
-- EMTAC Tablet Edge Agent
-- PostgreSQL schema setup
--
-- File:
--   modules/database_manager/tablet_edge/create_tablet_edge_schema.sql
--
-- Schema:
--   tablet_edge
-- ============================================================

CREATE SCHEMA IF NOT EXISTS tablet_edge;

-- ------------------------------------------------------------
-- tablet_device
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tablet_edge.tablet_device (
    id BIGSERIAL PRIMARY KEY,

    tablet_uid UUID NOT NULL UNIQUE,
    tablet_name VARCHAR(150) NOT NULL,

    device_make VARCHAR(100),
    device_model VARCHAR(100),
    android_version VARCHAR(50),
    app_version VARCHAR(50),

    assigned_area VARCHAR(150),
    assigned_station VARCHAR(150),
    assigned_role VARCHAR(100),

    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- tablet_network_event
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tablet_edge.tablet_network_event (
    id BIGSERIAL PRIMARY KEY,

    tablet_device_id BIGINT NOT NULL
        REFERENCES tablet_edge.tablet_device(id)
        ON DELETE CASCADE,

    event_type VARCHAR(100) NOT NULL,
    quality_level VARCHAR(50) NOT NULL,

    server_url TEXT,
    page_url TEXT,

    latency_ms INTEGER,
    avg_latency_ms INTEGER,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,

    is_online BOOLEAN,
    ssid VARCHAR(150),
    wifi_rssi INTEGER,
    signal_level INTEGER,
    ip_address VARCHAR(100),
    gateway_address VARCHAR(100),

    message TEXT,

    event_started_at TIMESTAMPTZ,
    event_ended_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- tablet_health_sample
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tablet_edge.tablet_health_sample (
    id BIGSERIAL PRIMARY KEY,

    tablet_device_id BIGINT NOT NULL
        REFERENCES tablet_edge.tablet_device(id)
        ON DELETE CASCADE,

    sampled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    server_reachable BOOLEAN NOT NULL DEFAULT FALSE,
    server_latency_ms INTEGER,

    quality_level VARCHAR(50) NOT NULL,

    battery_percent INTEGER,
    is_charging BOOLEAN,

    ssid VARCHAR(150),
    wifi_rssi INTEGER,
    signal_level INTEGER,

    app_foreground BOOLEAN,
    current_page_url TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- tablet_dropdown_cache_manifest
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tablet_edge.tablet_dropdown_cache_manifest (
    id BIGSERIAL PRIMARY KEY,

    tablet_device_id BIGINT NOT NULL
        REFERENCES tablet_edge.tablet_device(id)
        ON DELETE CASCADE,

    cache_name VARCHAR(150) NOT NULL,
    cache_version VARCHAR(150) NOT NULL,

    record_count INTEGER NOT NULL DEFAULT 0,

    last_full_sync_at TIMESTAMPTZ,
    last_delta_sync_at TIMESTAMPTZ,

    sync_status VARCHAR(50) NOT NULL DEFAULT 'unknown',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_tablet_cache_name
        UNIQUE (tablet_device_id, cache_name)
);

-- ------------------------------------------------------------
-- tablet_sync_event
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tablet_edge.tablet_sync_event (
    id BIGSERIAL PRIMARY KEY,

    tablet_device_id BIGINT NOT NULL
        REFERENCES tablet_edge.tablet_device(id)
        ON DELETE CASCADE,

    sync_type VARCHAR(100) NOT NULL,
    sync_direction VARCHAR(50) NOT NULL,

    status VARCHAR(50) NOT NULL,

    records_sent INTEGER NOT NULL DEFAULT 0,
    records_received INTEGER NOT NULL DEFAULT 0,
    records_failed INTEGER NOT NULL DEFAULT 0,

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    duration_ms INTEGER,

    error_message TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- tablet_offline_event
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tablet_edge.tablet_offline_event (
    id BIGSERIAL PRIMARY KEY,

    tablet_device_id BIGINT NOT NULL
        REFERENCES tablet_edge.tablet_device(id)
        ON DELETE CASCADE,

    local_event_id UUID NOT NULL,

    event_type VARCHAR(100) NOT NULL,
    event_payload JSONB NOT NULL,

    client_created_at TIMESTAMPTZ NOT NULL,
    server_received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    processing_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    processed_at TIMESTAMPTZ,

    error_message TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_tablet_local_event
        UNIQUE (tablet_device_id, local_event_id)
);

-- ------------------------------------------------------------
-- tablet_app_log
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tablet_edge.tablet_app_log (
    id BIGSERIAL PRIMARY KEY,

    tablet_device_id BIGINT
        REFERENCES tablet_edge.tablet_device(id)
        ON DELETE SET NULL,

    log_level VARCHAR(50) NOT NULL,
    log_source VARCHAR(150),
    message TEXT NOT NULL,

    context JSONB,

    client_created_at TIMESTAMPTZ,
    server_received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- indexes
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_tablet_device_uid
ON tablet_edge.tablet_device (tablet_uid);

CREATE INDEX IF NOT EXISTS idx_tablet_device_last_seen
ON tablet_edge.tablet_device (last_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_tablet_network_event_device_created
ON tablet_edge.tablet_network_event (tablet_device_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tablet_network_event_quality_created
ON tablet_edge.tablet_network_event (quality_level, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tablet_network_event_type_created
ON tablet_edge.tablet_network_event (event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tablet_health_sample_device_sampled
ON tablet_edge.tablet_health_sample (tablet_device_id, sampled_at DESC);

CREATE INDEX IF NOT EXISTS idx_tablet_health_sample_quality_sampled
ON tablet_edge.tablet_health_sample (quality_level, sampled_at DESC);

CREATE INDEX IF NOT EXISTS idx_tablet_dropdown_cache_device
ON tablet_edge.tablet_dropdown_cache_manifest (tablet_device_id, cache_name);

CREATE INDEX IF NOT EXISTS idx_tablet_sync_event_device_started
ON tablet_edge.tablet_sync_event (tablet_device_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_tablet_sync_event_status
ON tablet_edge.tablet_sync_event (status, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_tablet_offline_event_status
ON tablet_edge.tablet_offline_event (processing_status, created_at);

CREATE INDEX IF NOT EXISTS idx_tablet_offline_event_payload_gin
ON tablet_edge.tablet_offline_event
USING GIN (event_payload);

CREATE INDEX IF NOT EXISTS idx_tablet_app_log_device_created
ON tablet_edge.tablet_app_log (tablet_device_id, created_at DESC);

-- ------------------------------------------------------------
-- updated_at trigger function
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION tablet_edge.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_tablet_device_updated_at
ON tablet_edge.tablet_device;

CREATE TRIGGER trg_tablet_device_updated_at
BEFORE UPDATE ON tablet_edge.tablet_device
FOR EACH ROW
EXECUTE FUNCTION tablet_edge.set_updated_at();

DROP TRIGGER IF EXISTS trg_tablet_dropdown_cache_manifest_updated_at
ON tablet_edge.tablet_dropdown_cache_manifest;

CREATE TRIGGER trg_tablet_dropdown_cache_manifest_updated_at
BEFORE UPDATE ON tablet_edge.tablet_dropdown_cache_manifest
FOR EACH ROW
EXECUTE FUNCTION tablet_edge.set_updated_at();
'@

Write-FileSafe -Path $SchemaSqlPath -Content $SchemaSql

# ------------------------------------------------------------
# create_tablet_edge_schema.py
# ------------------------------------------------------------

$CreateSchemaPyPath = Join-Path $DbManagerDir "create_tablet_edge_schema.py"

$CreateSchemaPy = @'
"""
Create the EMTAC Tablet Edge Agent PostgreSQL schema.

File:
    modules/database_manager/tablet_edge/create_tablet_edge_schema.py

Run from project root:
    python -m modules.database_manager.tablet_edge.create_tablet_edge_schema

Notes:
    This script tries to use the existing EMTAC DatabaseConfig first.
    If that fails, it falls back to DATABASE_URL from the environment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


LOGGER_NAME = "tablet_edge_schema_setup"
logger = logging.getLogger(LOGGER_NAME)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_sql_path() -> Path:
    return Path(__file__).resolve().parent / "create_tablet_edge_schema.sql"


def _engine_from_database_config() -> Optional[Engine]:
    """
    Best-effort adapter for EMTAC DatabaseConfig.

    This avoids hard-coding one exact method name because DatabaseConfig has
    changed across project versions.
    """
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


def run_schema_setup() -> None:
    sql_path = get_sql_path()

    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8")

    logger.info("Starting tablet_edge schema setup.")
    logger.info("SQL file: %s", sql_path)

    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text(sql_text))

    logger.info("tablet_edge schema setup completed successfully.")


def main() -> None:
    configure_logging()
    run_schema_setup()


if __name__ == "__main__":
    main()
'@

Write-FileSafe -Path $CreateSchemaPyPath -Content $CreateSchemaPy

# ------------------------------------------------------------
# tablet_edge_schema_check.py
# ------------------------------------------------------------

$SchemaCheckPyPath = Join-Path $DbManagerDir "tablet_edge_schema_check.py"

$SchemaCheckPy = @'
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
'@

Write-FileSafe -Path $SchemaCheckPyPath -Content $SchemaCheckPy

# ------------------------------------------------------------
# seed_tablet_edge_dev_data.py
# ------------------------------------------------------------

$SeedPyPath = Join-Path $DbManagerDir "seed_tablet_edge_dev_data.py"

$SeedPy = @'
"""
Seed development tablet records for the EMTAC Tablet Edge Agent.

Run from project root:
    python -m modules.database_manager.tablet_edge.seed_tablet_edge_dev_data
"""

from __future__ import annotations

import logging
import os
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


logger = logging.getLogger("tablet_edge_seed_dev")


DEV_TABLETS = [
    {
        "tablet_uid": "00000000-0000-0000-0000-000000000101",
        "tablet_name": "EMTAC-GALAXY-DEV-01",
        "device_make": "Samsung",
        "device_model": "Galaxy Tablet",
        "android_version": "unknown",
        "app_version": "0.1.0-dev",
        "assigned_area": "Development",
        "assigned_station": "Dev Bench",
        "assigned_role": "maintenance_tablet",
    },
    {
        "tablet_uid": "00000000-0000-0000-0000-000000000102",
        "tablet_name": "EMTAC-LENOVO-DEV-01",
        "device_make": "Lenovo",
        "device_model": "Lenovo Tablet",
        "android_version": "unknown",
        "app_version": "0.1.0-dev",
        "assigned_area": "Development",
        "assigned_station": "Dev Bench",
        "assigned_role": "maintenance_tablet",
    },
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_engine() -> Engine:
    try:
        from modules.database_manager.tablet_edge.create_tablet_edge_schema import get_engine as get_project_engine
        return get_project_engine()
    except Exception as exc:
        logger.warning("Project engine helper failed: %s", exc)

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not set and project engine helper failed.")

    return create_engine(database_url)


def seed_dev_tablets() -> None:
    engine = get_engine()

    sql = text(
        """
        INSERT INTO tablet_edge.tablet_device (
            tablet_uid,
            tablet_name,
            device_make,
            device_model,
            android_version,
            app_version,
            assigned_area,
            assigned_station,
            assigned_role,
            last_seen_at,
            is_active
        )
        VALUES (
            :tablet_uid,
            :tablet_name,
            :device_make,
            :device_model,
            :android_version,
            :app_version,
            :assigned_area,
            :assigned_station,
            :assigned_role,
            NOW(),
            TRUE
        )
        ON CONFLICT (tablet_uid)
        DO UPDATE SET
            tablet_name = EXCLUDED.tablet_name,
            device_make = EXCLUDED.device_make,
            device_model = EXCLUDED.device_model,
            android_version = EXCLUDED.android_version,
            app_version = EXCLUDED.app_version,
            assigned_area = EXCLUDED.assigned_area,
            assigned_station = EXCLUDED.assigned_station,
            assigned_role = EXCLUDED.assigned_role,
            last_seen_at = NOW(),
            is_active = TRUE,
            updated_at = NOW()
        """
    )

    with engine.begin() as conn:
        for tablet in DEV_TABLETS:
            UUID(tablet["tablet_uid"])
            conn.execute(sql, tablet)
            logger.info("Seeded dev tablet: %s", tablet["tablet_name"])

    logger.info("Development tablet seed completed.")


def main() -> None:
    configure_logging()
    seed_dev_tablets()


if __name__ == "__main__":
    main()
'@

Write-FileSafe -Path $SeedPyPath -Content $SeedPy

# ------------------------------------------------------------
# test_tablet_edge_routes.py
# ------------------------------------------------------------

$TestRoutesPyPath = Join-Path $DbManagerDir "test_tablet_edge_routes.py"

$TestRoutesPy = @'
"""
Simple route smoke tests for EMTAC Tablet Edge Agent endpoints.

Run from project root while Flask server is running:
    python -m modules.database_manager.tablet_edge.test_tablet_edge_routes

Optional:
    set EMTAC_TABLET_EDGE_BASE_URL=http://127.0.0.1:8060
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

import requests


logger = logging.getLogger("tablet_edge_route_tests")


BASE_URL = os.getenv("EMTAC_TABLET_EDGE_BASE_URL", "http://127.0.0.1:8060").rstrip("/")
TABLET_UID = os.getenv("EMTAC_TABLET_EDGE_TEST_UID", str(uuid4()))


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def request_json(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    logger.info("%s %s", method.upper(), url)

    started = time.perf_counter()

    response = requests.request(
        method=method,
        url=url,
        json=payload,
        timeout=10,
    )

    duration_ms = int((time.perf_counter() - started) * 1000)

    logger.info("Status=%s Duration=%sms", response.status_code, duration_ms)

    try:
        body = response.json()
    except Exception:
        body = {"raw_text": response.text}

    logger.info("Response body:\n%s", json.dumps(body, indent=2, default=str))

    if response.status_code >= 400:
        raise RuntimeError(f"Request failed: {method} {path} status={response.status_code}")

    return body


def run_tests() -> None:
    logger.info("Using BASE_URL=%s", BASE_URL)
    logger.info("Using TABLET_UID=%s", TABLET_UID)

    request_json("GET", "/tablet-edge/health")

    register_payload = {
        "tablet_uid": TABLET_UID,
        "tablet_name": "EMTAC-ROUTE-TEST-01",
        "device_make": "RouteTest",
        "device_model": "PowerShell/Python",
        "android_version": "test",
        "app_version": "0.1.0-test",
        "assigned_area": "Development",
        "assigned_station": "Route Test",
        "assigned_role": "maintenance_tablet",
    }

    request_json("POST", "/tablet-edge/register", register_payload)

    heartbeat_payload = {
        "tablet_uid": TABLET_UID,
        "app_version": "0.1.0-test",
        "current_page_url": "/assistant",
        "quality_level": "good",
    }

    request_json("POST", "/tablet-edge/heartbeat", heartbeat_payload)

    network_event_payload = {
        "tablet_uid": TABLET_UID,
        "events": [
            {
                "event_type": "server_health_good",
                "quality_level": "good",
                "server_url": BASE_URL,
                "page_url": "/assistant",
                "latency_ms": 120,
                "avg_latency_ms": 140,
                "consecutive_failures": 0,
                "is_online": True,
                "ssid": "TestWiFi",
                "wifi_rssi": -55,
                "signal_level": 4,
                "ip_address": "127.0.0.1",
                "gateway_address": "127.0.0.1",
                "message": "Route smoke test network event.",
            }
        ],
    }

    request_json("POST", "/tablet-edge/network-events", network_event_payload)

    now_utc = datetime.now(timezone.utc).isoformat()

    health_sample_payload = {
        "tablet_uid": TABLET_UID,
        "samples": [
            {
                "sampled_at": now_utc,
                "server_reachable": True,
                "server_latency_ms": 120,
                "quality_level": "good",
                "battery_percent": 88,
                "is_charging": False,
                "ssid": "TestWiFi",
                "wifi_rssi": -55,
                "signal_level": 4,
                "app_foreground": True,
                "current_page_url": "/assistant",
            }
        ],
    }

    request_json("POST", "/tablet-edge/health-samples", health_sample_payload)

    request_json("GET", f"/tablet-edge/dropdown-cache/status?tablet_uid={TABLET_UID}")
    request_json("GET", "/tablet-edge/dropdown-cache/full")

    offline_event_payload = {
        "tablet_uid": TABLET_UID,
        "events": [
            {
                "local_event_id": str(uuid4()),
                "event_type": "route_test_event",
                "client_created_at": now_utc,
                "event_payload": {
                    "message": "Offline event route smoke test.",
                    "source": "test_tablet_edge_routes.py",
                },
            }
        ],
    }

    request_json("POST", "/tablet-edge/offline-events/sync", offline_event_payload)

    logger.info("All route smoke tests completed.")


def main() -> None:
    configure_logging()
    run_tests()


if __name__ == "__main__":
    main()
'@

Write-FileSafe -Path $TestRoutesPyPath -Content $TestRoutesPy

# ------------------------------------------------------------
# package init files
# ------------------------------------------------------------

$ServiceInitPath = Join-Path $ServiceDir "__init__.py"
Write-FileSafe -Path $ServiceInitPath -Content '"""Services for EMTAC Tablet Edge Agent."""'

$DbManagerInitPath = Join-Path $DbManagerDir "__init__.py"
Write-FileSafe -Path $DbManagerInitPath -Content '"""Database manager scripts for EMTAC Tablet Edge Agent."""'

# ------------------------------------------------------------
# report folder marker
# ------------------------------------------------------------

$ReportGitkeep = Join-Path $ReportsDir ".gitkeep"
Write-FileSafe -Path $ReportGitkeep -Content ""

# ------------------------------------------------------------
# create placeholder reminder files
# ------------------------------------------------------------

$ReadmePath = Join-Path $DbManagerDir "README_tablet_edge_setup.md"

$ReadmeLines = @(
    "# EMTAC Tablet Edge Backend Setup",
    "",
    "Generated by:",
    "",
    "setup_tablet_edge_backend.ps1",
    "",
    "## Generated files",
    "",
    "modules/database_manager/tablet_edge/create_tablet_edge_schema.sql",
    "modules/database_manager/tablet_edge/create_tablet_edge_schema.py",
    "modules/database_manager/tablet_edge/tablet_edge_schema_check.py",
    "modules/database_manager/tablet_edge/seed_tablet_edge_dev_data.py",
    "modules/database_manager/tablet_edge/test_tablet_edge_routes.py",
    "reports/tablet_edge/.gitkeep",
    "",
    "## First run order",
    "",
    "From the MDEV_EMTAC project root:",
    "",
    ".\.venv\Scripts\python.exe -m modules.database_manager.tablet_edge.create_tablet_edge_schema",
    ".\.venv\Scripts\python.exe -m modules.database_manager.tablet_edge.tablet_edge_schema_check",
    ".\.venv\Scripts\python.exe -m modules.database_manager.tablet_edge.seed_tablet_edge_dev_data",
    "",
    "After the Flask blueprint/routes are implemented and the server is running:",
    "",
    ".\.venv\Scripts\python.exe -m modules.database_manager.tablet_edge.test_tablet_edge_routes",
    "",
    "## Next files to implement",
    "",
    "modules/emtacdb/tablet_edge_models.py",
    "blueprints/tablet_edge_bp.py",
    "modules/coordinators/tablet_edge_coordinator.py",
    "modules/orchestrators/tablet_edge_orchestrator.py",
    "modules/services/tablet_edge/tablet_device_service.py",
    "modules/services/tablet_edge/tablet_network_event_service.py",
    "modules/services/tablet_edge/tablet_health_sample_service.py",
    "modules/services/tablet_edge/tablet_dropdown_cache_service.py",
    "modules/services/tablet_edge/tablet_sync_service.py",
    "modules/services/tablet_edge/tablet_offline_event_service.py"
)

$Readme = $ReadmeLines -join [Environment]::NewLine

Write-FileSafe -Path $ReadmePath -Content $Readme

Write-Host ""
Write-SetupInfo "Tablet Edge backend setup scaffold complete."
Write-Host ""
Write-Host "Next commands from project root:" -ForegroundColor Green
Write-Host ""
Write-Host ".\.venv\Scripts\python.exe -m modules.database_manager.tablet_edge.create_tablet_edge_schema"
Write-Host ".\.venv\Scripts\python.exe -m modules.database_manager.tablet_edge.tablet_edge_schema_check"
Write-Host ".\.venv\Scripts\python.exe -m modules.database_manager.tablet_edge.seed_tablet_edge_dev_data"
Write-Host ""
Write-SetupWarn "This setup script creates the database/scripts scaffold only."
Write-SetupWarn "The next step is implementing models, blueprint, coordinator, orchestrator, and services."
