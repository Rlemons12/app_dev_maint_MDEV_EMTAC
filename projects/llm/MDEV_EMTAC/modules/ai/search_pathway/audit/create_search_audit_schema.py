# ============================================================
# File:
#   modules/ai/search_pathway/audit/create_search_audit_schema.py
#
# Purpose:
#   Create PostgreSQL audit schema and tables for AI/search pathway auditing.
#
# Run from project root:
#   cd E:\emtac\projects\llm\MDEV_EMTAC
#   python -m modules.ai.search_pathway.audit.create_search_audit_schema
# ============================================================

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


# ============================================================
# Project path setup
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

# File path:
#   modules/ai/search_pathway/audit/create_search_audit_schema.py
#
# parents[0] = audit
# parents[1] = search_pathway
# parents[2] = ai
# parents[3] = modules
# parents[4] = project root
PROJECT_ROOT = CURRENT_FILE.parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

try:
    from modules.configuration.config_env import get_db_config
except Exception as import_error:
    raise RuntimeError(
        "Could not import get_db_config from modules.configuration.config_env. "
        "Make sure you are running this from the project root."
    ) from import_error


try:
    from modules.ai.search_pathway.audit.search_audit_logger import (
        get_search_audit_logger,
    )

    logger = get_search_audit_logger()
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("search_audit_schema_creator")


# ============================================================
# Constants
# ============================================================

AUDIT_SCHEMA_NAME = "audit"

EXPECTED_AUDIT_TABLES = {
    "search_audit_run",
    "search_audit_stage",
    "search_audit_candidate",
    "search_audit_payload_item",
    "search_audit_validation",
    "search_expected_case",
}


# ============================================================
# SQL statements
# ============================================================

CREATE_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS audit;
"""


CREATE_PGCRYPTO_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
"""


CHECK_GEN_RANDOM_UUID_SQL = """
SELECT gen_random_uuid();
"""


CHECK_QANDA_TABLE_SQL = """
SELECT EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name = 'qanda'
);
"""


CREATE_SEARCH_AUDIT_RUN_SQL = """
CREATE TABLE IF NOT EXISTS audit.search_audit_run (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    qanda_id UUID NULL REFERENCES public.qanda(id) ON DELETE SET NULL,

    request_id TEXT NOT NULL,
    user_id TEXT NULL,
    session_id UUID NULL,

    question TEXT NULL,
    normalized_question TEXT NULL,
    final_answer TEXT NULL,

    pathway_name TEXT NOT NULL DEFAULT 'unknown',
    pathway_version TEXT NULL,
    search_mode TEXT NULL,

    payload_status TEXT NOT NULL DEFAULT 'unknown',
    validation_status TEXT NOT NULL DEFAULT 'not_validated',

    answer_hash TEXT NULL,
    response_hash TEXT NULL,
    payload_hash TEXT NULL,

    model_name TEXT NULL,

    raw_request JSONB NULL,
    raw_response JSONB NULL,
    raw_payload JSONB NULL,
    raw_chunks JSONB NULL,
    raw_relationship_map JSONB NULL,
    validation_summary JSONB NULL,

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NULL,
    duration_ms INTEGER NULL,

    error_text TEXT NULL
);
"""


CREATE_SEARCH_AUDIT_STAGE_SQL = """
CREATE TABLE IF NOT EXISTS audit.search_audit_stage (
    id BIGSERIAL PRIMARY KEY,

    audit_run_id UUID NOT NULL REFERENCES audit.search_audit_run(id) ON DELETE CASCADE,

    stage_name TEXT NOT NULL,
    stage_order INTEGER NULL,

    status TEXT NOT NULL DEFAULT 'started',

    input_snapshot JSONB NULL,
    output_snapshot JSONB NULL,

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NULL,
    duration_ms INTEGER NULL,

    error_text TEXT NULL
);
"""


CREATE_SEARCH_AUDIT_CANDIDATE_SQL = """
CREATE TABLE IF NOT EXISTS audit.search_audit_candidate (
    id BIGSERIAL PRIMARY KEY,

    audit_run_id UUID NOT NULL REFERENCES audit.search_audit_run(id) ON DELETE CASCADE,
    stage_id BIGINT NULL REFERENCES audit.search_audit_stage(id) ON DELETE SET NULL,

    candidate_type TEXT NOT NULL,
    source_table TEXT NULL,
    source_id INTEGER NULL,

    title TEXT NULL,
    label TEXT NULL,

    rank INTEGER NULL,
    score DOUBLE PRECISION NULL,

    selected BOOLEAN NOT NULL DEFAULT FALSE,
    rejected_reason TEXT NULL,

    evidence JSONB NULL,
    candidate_hash TEXT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


CREATE_SEARCH_AUDIT_PAYLOAD_ITEM_SQL = """
CREATE TABLE IF NOT EXISTS audit.search_audit_payload_item (
    id BIGSERIAL PRIMARY KEY,

    audit_run_id UUID NOT NULL REFERENCES audit.search_audit_run(id) ON DELETE CASCADE,

    item_type TEXT NOT NULL,
    source_table TEXT NULL,
    source_id INTEGER NULL,

    title TEXT NULL,
    label TEXT NULL,
    file_path TEXT NULL,
    url TEXT NULL,

    rank INTEGER NULL,
    score DOUBLE PRECISION NULL,

    relationship_path TEXT NULL,
    evidence JSONB NULL,

    item_hash TEXT NULL,

    exists_in_db BOOLEAN NULL,
    exists_on_disk BOOLEAN NULL,

    validation_status TEXT NOT NULL DEFAULT 'not_validated',
    validation_message TEXT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


CREATE_SEARCH_AUDIT_VALIDATION_SQL = """
CREATE TABLE IF NOT EXISTS audit.search_audit_validation (
    id BIGSERIAL PRIMARY KEY,

    audit_run_id UUID NOT NULL REFERENCES audit.search_audit_run(id) ON DELETE CASCADE,

    check_name TEXT NOT NULL,
    check_status TEXT NOT NULL,

    expected_count INTEGER NULL,
    actual_count INTEGER NULL,

    details JSONB NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


CREATE_SEARCH_EXPECTED_CASE_SQL = """
CREATE TABLE IF NOT EXISTS audit.search_expected_case (
    id BIGSERIAL PRIMARY KEY,

    case_name TEXT NOT NULL,
    question TEXT NOT NULL,
    question_hash TEXT NOT NULL,

    pathway_name TEXT NOT NULL DEFAULT 'rag',

    expected_payload JSONB NOT NULL,

    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    notes TEXT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NULL
);
"""


CREATE_INDEXES_SQL = [
    # --------------------------------------------------------
    # audit.search_audit_run indexes
    # --------------------------------------------------------
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_run_request_id
    ON audit.search_audit_run (request_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_run_qanda_id
    ON audit.search_audit_run (qanda_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_run_user_id
    ON audit.search_audit_run (user_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_run_session_id
    ON audit.search_audit_run (session_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_run_pathway_name
    ON audit.search_audit_run (pathway_name);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_run_validation_status
    ON audit.search_audit_run (validation_status);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_run_started_at
    ON audit.search_audit_run (started_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_run_raw_response_gin
    ON audit.search_audit_run USING GIN (raw_response);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_run_raw_payload_gin
    ON audit.search_audit_run USING GIN (raw_payload);
    """,

    # --------------------------------------------------------
    # audit.search_audit_stage indexes
    # --------------------------------------------------------
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_stage_run_id
    ON audit.search_audit_stage (audit_run_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_stage_name
    ON audit.search_audit_stage (stage_name);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_stage_status
    ON audit.search_audit_stage (status);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_stage_order
    ON audit.search_audit_stage (audit_run_id, stage_order);
    """,

    # --------------------------------------------------------
    # audit.search_audit_candidate indexes
    # --------------------------------------------------------
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_candidate_run_id
    ON audit.search_audit_candidate (audit_run_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_candidate_stage_id
    ON audit.search_audit_candidate (stage_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_candidate_type_source
    ON audit.search_audit_candidate (candidate_type, source_table, source_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_candidate_selected
    ON audit.search_audit_candidate (selected);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_candidate_evidence_gin
    ON audit.search_audit_candidate USING GIN (evidence);
    """,

    # --------------------------------------------------------
    # audit.search_audit_payload_item indexes
    # --------------------------------------------------------
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_payload_item_run_id
    ON audit.search_audit_payload_item (audit_run_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_payload_item_type_source
    ON audit.search_audit_payload_item (item_type, source_table, source_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_payload_item_validation_status
    ON audit.search_audit_payload_item (validation_status);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_payload_item_created_at
    ON audit.search_audit_payload_item (created_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_payload_item_evidence_gin
    ON audit.search_audit_payload_item USING GIN (evidence);
    """,

    # --------------------------------------------------------
    # audit.search_audit_validation indexes
    # --------------------------------------------------------
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_validation_run_id
    ON audit.search_audit_validation (audit_run_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_validation_check_name
    ON audit.search_audit_validation (check_name);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_audit_validation_check_status
    ON audit.search_audit_validation (check_status);
    """,

    # --------------------------------------------------------
    # audit.search_expected_case indexes
    # --------------------------------------------------------
    """
    CREATE INDEX IF NOT EXISTS idx_search_expected_case_question_hash
    ON audit.search_expected_case (question_hash);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_expected_case_pathway_name
    ON audit.search_expected_case (pathway_name);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_expected_case_active
    ON audit.search_expected_case (is_active);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_search_expected_case_payload_gin
    ON audit.search_expected_case USING GIN (expected_payload);
    """,
]


ALL_TABLE_SQL = [
    CREATE_SEARCH_AUDIT_RUN_SQL,
    CREATE_SEARCH_AUDIT_STAGE_SQL,
    CREATE_SEARCH_AUDIT_CANDIDATE_SQL,
    CREATE_SEARCH_AUDIT_PAYLOAD_ITEM_SQL,
    CREATE_SEARCH_AUDIT_VALIDATION_SQL,
    CREATE_SEARCH_EXPECTED_CASE_SQL,
]


VERIFY_AUDIT_TABLES_SQL = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'audit'
ORDER BY table_name;
"""


# ============================================================
# Schema creator
# ============================================================

class SearchAuditSchemaCreator:
    """
    Creates the audit schema and search-audit tables.

    This script intentionally runs DDL only.
    It does not insert audit records.
    """

    def __init__(self) -> None:
        self.db_config = get_db_config()

    def create_schema(self) -> bool:
        session = None

        try:
            logger.info("Starting search audit schema creation.")
            logger.info("Project root detected as: %s", PROJECT_ROOT)

            session = self.db_config.get_main_session()

            self._verify_qanda_table_exists(session)
            self._create_audit_schema(session)
            self._create_uuid_extension_if_possible(session)
            self._verify_uuid_function_available(session)
            self._create_tables(session)
            self._create_indexes(session)
            self._verify_expected_tables(session)

            logger.info("Search audit schema creation completed successfully.")
            return True

        except SQLAlchemyError as exc:
            if session:
                session.rollback()

            logger.error("SQLAlchemy error while creating search audit schema: %s", exc)
            logger.error(traceback.format_exc())
            return False

        except Exception as exc:
            if session:
                session.rollback()

            logger.error("Unexpected error while creating search audit schema: %s", exc)
            logger.error(traceback.format_exc())
            return False

        finally:
            if session:
                session.close()
                logger.debug("Database session closed.")

    @staticmethod
    def _verify_qanda_table_exists(session) -> None:
        """
        Verify public.qanda exists before creating the audit FK.

        The audit.search_audit_run table references public.qanda(id).
        If qanda does not exist, table creation will fail later with a less clear error.
        """

        logger.info("Verifying required table public.qanda exists.")

        qanda_exists = session.execute(text(CHECK_QANDA_TABLE_SQL)).scalar()

        if not qanda_exists:
            raise RuntimeError(
                "Required table public.qanda does not exist. "
                "Create the main application tables before creating the audit schema."
            )

        logger.info("Verified public.qanda exists.")

    @staticmethod
    def _create_audit_schema(session) -> None:
        logger.info("Creating audit schema if needed.")
        session.execute(text(CREATE_SCHEMA_SQL))
        session.commit()
        logger.info("Audit schema ready.")

    @staticmethod
    def _create_uuid_extension_if_possible(session) -> None:
        """
        Try to create pgcrypto.

        Some PostgreSQL users may not have permission to create extensions.
        On newer PostgreSQL versions, gen_random_uuid may already be available.
        So this method warns but does not fail immediately.
        """

        try:
            logger.info("Creating PostgreSQL extension pgcrypto if available.")
            session.execute(text(CREATE_PGCRYPTO_SQL))
            session.commit()
            logger.info("pgcrypto extension ready or already present.")

        except Exception as extension_error:
            session.rollback()
            logger.warning(
                "Could not create pgcrypto extension. Will test gen_random_uuid() next. "
                "Error: %s",
                extension_error,
            )

    @staticmethod
    def _verify_uuid_function_available(session) -> None:
        """
        Verify gen_random_uuid() is available.

        The audit tables use:
            id UUID PRIMARY KEY DEFAULT gen_random_uuid()

        If this function is not available, table creation should stop with
        a clear error message.
        """

        logger.info("Verifying gen_random_uuid() is available.")

        try:
            session.execute(text(CHECK_GEN_RANDOM_UUID_SQL)).scalar()
            session.rollback()
            logger.info("Verified gen_random_uuid() is available.")

        except Exception as uuid_error:
            session.rollback()
            raise RuntimeError(
                "PostgreSQL function gen_random_uuid() is not available. "
                "Install/enable pgcrypto or update the audit table ID defaults."
            ) from uuid_error

    @staticmethod
    def _create_tables(session) -> None:
        logger.info("Creating audit tables.")

        for statement in ALL_TABLE_SQL:
            session.execute(text(statement))

        session.commit()
        logger.info("Audit tables ready.")

    @staticmethod
    def _create_indexes(session) -> None:
        logger.info("Creating audit indexes.")

        for statement in CREATE_INDEXES_SQL:
            session.execute(text(statement))

        session.commit()
        logger.info("Audit indexes ready.")

    @staticmethod
    def _verify_expected_tables(session) -> None:
        logger.info("Verifying expected audit tables exist.")

        rows = session.execute(text(VERIFY_AUDIT_TABLES_SQL)).all()
        found_tables = {row[0] for row in rows}

        missing_tables = EXPECTED_AUDIT_TABLES - found_tables

        if missing_tables:
            raise RuntimeError(
                f"Audit schema verification failed. Missing tables: "
                f"{sorted(missing_tables)}"
            )

        logger.info(
            "Verified audit tables: %s",
            sorted(found_tables),
        )


# ============================================================
# Main
# ============================================================

def main() -> int:
    creator = SearchAuditSchemaCreator()
    success = creator.create_schema()

    if success:
        logger.info("Done. Audit schema is ready.")
        return 0

    logger.error("Failed to create audit schema.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())