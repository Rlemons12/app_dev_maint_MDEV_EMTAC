from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(r"E:\emtac\projects\llm\MDEV_EMTAC")

AUDIT_DIR = PROJECT_ROOT / "modules" / "ai" / "search_pathway" / "audit"

# Leave False unless you intentionally want this script to overwrite
# files you have already edited by hand.
OVERWRITE_EXISTING = True


# ============================================================
# Logging setup for this setup script only
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# File contents
# ============================================================

FILES_TO_CREATE = {
    "__init__.py": dedent(
        '''\
        """
        Search pathway auditing package.

        This package contains logic for auditing AI/search pathways.

        Initial focus:
            - RAG search auditing

        Future expansion:
            - keyword search auditing
            - vector search auditing
            - hybrid search auditing
            - part search auditing
            - drawing search auditing
            - image search auditing
            - AI-based result review
        """
        '''
    ),

    "search_audit_types.py": dedent(
        '''\
        from __future__ import annotations

        from enum import Enum


        class SearchPathwayName(str, Enum):
            """
            Names for supported search pathways.

            These values help identify which pathway produced the result.
            """

            RAG = "rag"
            UNIFIED_SEARCH = "unified_search"
            FORCED_CHUNK_DEBUG = "forced_chunk_debug"
            KEYWORD_SEARCH = "keyword_search"
            VECTOR_SEARCH = "vector_search"
            HYBRID_SEARCH = "hybrid_search"
            PAYLOAD_PROJECTION = "payload_projection"


        class SearchAuditStageName(str, Enum):
            """
            Common stages inside a search pathway.
            """

            NORMALIZE_QUESTION = "normalize_question"
            CLASSIFY_INTENT = "classify_intent"
            GENERATE_EMBEDDING = "generate_embedding"
            RETRIEVE_CANDIDATES = "retrieve_candidates"
            VECTOR_RETRIEVE_CHUNKS = "vector_retrieve_chunks"
            KEYWORD_RETRIEVE_CHUNKS = "keyword_retrieve_chunks"
            RERANK_CHUNKS = "rerank_chunks"
            BUILD_CONTEXT = "build_context"
            GENERATE_ANSWER = "generate_answer"
            RESOLVE_RELATIONSHIPS = "resolve_relationships"
            BUILD_PAYLOAD = "build_payload"
            VALIDATE_PAYLOAD = "validate_payload"
            AI_REVIEW = "ai_review"


        class SearchAuditItemType(str, Enum):
            """
            Types of items that may be returned by search pathways.
            """

            CHUNK = "chunk"
            DOCUMENT = "document"
            COMPLETE_DOCUMENT = "complete_document"
            IMAGE = "image"
            DRAWING = "drawing"
            PART = "part"
            POSITION = "position"
            PROBLEM = "problem"
            SOLUTION = "solution"
            TASK = "task"


        class SearchAuditStatus(str, Enum):
            """
            General audit status values.
            """

            STARTED = "started"
            COMPLETED = "completed"
            FAILED = "failed"
            SKIPPED = "skipped"


        class SearchAuditValidationStatus(str, Enum):
            """
            Validation result values.
            """

            NOT_VALIDATED = "not_validated"
            PASSED = "passed"
            WARNING = "warning"
            FAILED = "failed"
        '''
    ),

    "search_audit_logger.py": dedent(
        '''\
        from __future__ import annotations

        from functools import lru_cache
        from logging import LoggerAdapter

        from modules.configuration.log_config import SearchAuditLogManager


        @lru_cache(maxsize=1)
        def get_search_audit_log_manager() -> SearchAuditLogManager:
            """
            Returns the global search audit log manager.

            This prevents duplicate logger handlers from being attached repeatedly.
            """

            return SearchAuditLogManager(
                run_name="global",
                to_console=False,
            )


        def get_search_audit_logger() -> LoggerAdapter:
            """
            Returns the dedicated search audit logger adapter.
            """

            return get_search_audit_log_manager().logger
        '''
    ),

    "search_audit_payload_extractor.py": dedent(
        '''\
        from __future__ import annotations

        from typing import Any


        class SearchAuditPayloadExtractor:
            """
            Extracts audit-friendly data from a search or payload response.

            The goal is to normalize different response shapes into predictable groups:

                - chunks
                - documents
                - images
                - drawings
                - parts

            This keeps the audit service from needing to know every possible
            frontend/backend payload structure.
            """

            @staticmethod
            def _as_list(value: Any) -> list[dict[str, Any]]:
                """
                Convert a possible payload section into a list of dictionaries.

                Supported shapes:
                    list[dict]
                    {"items": list[dict]}
                    {"results": list[dict]}
                    {"documents": list[dict]}
                    {"images": list[dict]}
                    {"drawings": list[dict]}
                    {"parts": list[dict]}
                    {"chunks": list[dict]}
                """

                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]

                if isinstance(value, dict):
                    for key in (
                        "items",
                        "results",
                        "documents",
                        "images",
                        "drawings",
                        "parts",
                        "chunks",
                    ):
                        nested = value.get(key)
                        if isinstance(nested, list):
                            return [item for item in nested if isinstance(item, dict)]

                return []

            @classmethod
            def extract_chunks(cls, response: dict[str, Any]) -> list[dict[str, Any]]:
                response = response or {}

                return cls._as_list(
                    response.get("chunks")
                    or response.get("used_chunks")
                    or response.get("retrieved_chunks")
                )

            @classmethod
            def extract_documents(cls, response: dict[str, Any]) -> list[dict[str, Any]]:
                response = response or {}

                return cls._as_list(
                    response.get("documents")
                    or response.get("document_panel")
                    or response.get("docs")
                )

            @classmethod
            def extract_images(cls, response: dict[str, Any]) -> list[dict[str, Any]]:
                response = response or {}

                return cls._as_list(
                    response.get("images")
                    or response.get("image_panel")
                    or response.get("thumbnails")
                )

            @classmethod
            def extract_drawings(cls, response: dict[str, Any]) -> list[dict[str, Any]]:
                response = response or {}

                return cls._as_list(
                    response.get("drawings")
                    or response.get("drawing_panel")
                )

            @classmethod
            def extract_parts(cls, response: dict[str, Any]) -> list[dict[str, Any]]:
                response = response or {}

                return cls._as_list(
                    response.get("parts")
                    or response.get("part_panel")
                )

            @classmethod
            def extract_all(cls, response: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
                response = response or {}

                return {
                    "chunks": cls.extract_chunks(response),
                    "documents": cls.extract_documents(response),
                    "images": cls.extract_images(response),
                    "drawings": cls.extract_drawings(response),
                    "parts": cls.extract_parts(response),
                }
        '''
    ),

    "search_audit_service.py": dedent(
        '''\
        from __future__ import annotations

        import hashlib
        import json
        from datetime import datetime
        from typing import Any
        from uuid import UUID

        from modules.ai.search_pathway.audit.search_audit_logger import (
            get_search_audit_log_manager,
        )
        from modules.ai.search_pathway.audit.search_audit_payload_extractor import (
            SearchAuditPayloadExtractor,
        )
        from modules.ai.search_pathway.audit.search_audit_types import (
            SearchAuditValidationStatus,
            SearchPathwayName,
        )


        class SearchAuditService:
            """
            Records audit data for AI search pathways.

            Important architecture rule:
                This service should not create sessions.
                This service should not commit.
                This service should not rollback.

            The orchestrator should own transaction control.
            """

            @staticmethod
            def stable_hash(value: Any) -> str:
                """
                Create a stable hash for dictionaries, lists, strings, and other values.
                """

                serialized = json.dumps(value, sort_keys=True, default=str)
                return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

            @classmethod
            def record_search_result(
                cls,
                *,
                session,
                request_id: str,
                user_id: str | None,
                session_id: UUID | None,
                qanda_id: UUID | None,
                question: str,
                answer: str | None,
                response: dict[str, Any],
                pathway_name: str = SearchPathwayName.RAG.value,
                pathway_version: str = "1.0",
                duration_ms: int | None = None,
                model_name: str | None = None,
            ) -> dict[str, Any]:
                """
                Temporary audit entry point.

                For now, this returns an audit summary dictionary.

                Later, this method can be expanded to insert records into:

                    audit.search_audit_run
                    audit.search_audit_stage
                    audit.search_audit_candidate
                    audit.search_audit_payload_item
                    audit.search_audit_validation
                """

                response = response or {}
                extracted = SearchAuditPayloadExtractor.extract_all(response)

                counts = {
                    "chunks": len(extracted["chunks"]),
                    "documents": len(extracted["documents"]),
                    "images": len(extracted["images"]),
                    "drawings": len(extracted["drawings"]),
                    "parts": len(extracted["parts"]),
                }

                audit_summary = {
                    "request_id": request_id,
                    "user_id": user_id,
                    "session_id": str(session_id) if session_id else None,
                    "qanda_id": str(qanda_id) if qanda_id else None,
                    "question": question,
                    "answer_hash": cls.stable_hash(answer or ""),
                    "response_hash": cls.stable_hash(response),
                    "pathway_name": pathway_name,
                    "pathway_version": pathway_version,
                    "model_name": model_name,
                    "duration_ms": duration_ms,
                    "counts": counts,
                    "validation_status": SearchAuditValidationStatus.NOT_VALIDATED.value,
                    "created_at": datetime.utcnow().isoformat(),
                }

                audit_log_manager = get_search_audit_log_manager()

                audit_log_manager.log_payload_counts(
                    request_id=request_id,
                    pathway_name=pathway_name,
                    counts=counts,
                )

                audit_log_manager.log_run_success(
                    request_id=request_id,
                    pathway_name=pathway_name,
                    duration_ms=duration_ms,
                    counts=counts,
                )

                return audit_summary
        '''
    ),

    "search_audit_stage_tracker.py": dedent(
        '''\
        from __future__ import annotations

        import time
        from dataclasses import dataclass, field
        from typing import Any

        from modules.ai.search_pathway.audit.search_audit_logger import (
            get_search_audit_logger,
        )


        logger = get_search_audit_logger()


        @dataclass
        class SearchAuditStageRecord:
            stage_name: str
            started_at: float
            completed_at: float | None = None
            duration_ms: int | None = None
            status: str = "started"
            input_snapshot: dict[str, Any] | None = None
            output_snapshot: dict[str, Any] | None = None
            error_text: str | None = None


        @dataclass
        class SearchAuditStageTracker:
            """
            Lightweight in-memory tracker for search pathway stages.

            This can later be connected to audit.search_audit_stage.
            """

            request_id: str
            pathway_name: str = "rag"
            stages: list[SearchAuditStageRecord] = field(default_factory=list)

            def start_stage(
                self,
                stage_name: str,
                input_snapshot: dict[str, Any] | None = None,
            ) -> SearchAuditStageRecord:
                record = SearchAuditStageRecord(
                    stage_name=stage_name,
                    started_at=time.perf_counter(),
                    input_snapshot=input_snapshot,
                )

                self.stages.append(record)

                logger.debug(
                    "AUDIT_STAGE_START request_id=%s pathway=%s stage=%s",
                    self.request_id,
                    self.pathway_name,
                    stage_name,
                )

                return record

            def complete_stage(
                self,
                record: SearchAuditStageRecord,
                output_snapshot: dict[str, Any] | None = None,
            ) -> None:
                completed_at = time.perf_counter()
                record.completed_at = completed_at
                record.duration_ms = int((completed_at - record.started_at) * 1000)
                record.status = "completed"
                record.output_snapshot = output_snapshot

                logger.debug(
                    "AUDIT_STAGE_SUCCESS request_id=%s pathway=%s stage=%s duration_ms=%s",
                    self.request_id,
                    self.pathway_name,
                    record.stage_name,
                    record.duration_ms,
                )

            def fail_stage(
                self,
                record: SearchAuditStageRecord,
                error: Exception,
            ) -> None:
                completed_at = time.perf_counter()
                record.completed_at = completed_at
                record.duration_ms = int((completed_at - record.started_at) * 1000)
                record.status = "failed"
                record.error_text = str(error)

                logger.error(
                    "AUDIT_STAGE_FAILURE request_id=%s pathway=%s stage=%s duration_ms=%s error=%s",
                    self.request_id,
                    self.pathway_name,
                    record.stage_name,
                    record.duration_ms,
                    error,
                    exc_info=True,
                )

            def to_dict(self) -> dict[str, Any]:
                return {
                    "request_id": self.request_id,
                    "pathway_name": self.pathway_name,
                    "stages": [
                        {
                            "stage_name": stage.stage_name,
                            "duration_ms": stage.duration_ms,
                            "status": stage.status,
                            "input_snapshot": stage.input_snapshot,
                            "output_snapshot": stage.output_snapshot,
                            "error_text": stage.error_text,
                        }
                        for stage in self.stages
                    ],
                }
        '''
    ),

    "search_audit_validator.py": dedent(
        '''\
        from __future__ import annotations

        from typing import Any

        from modules.ai.search_pathway.audit.search_audit_logger import (
            get_search_audit_logger,
        )


        logger = get_search_audit_logger()


        class SearchAuditValidator:
            """
            Validates search audit results.

            This class should eventually check things like:

                - returned chunks exist
                - returned documents exist
                - returned images exist
                - returned drawings exist
                - returned parts exist
                - files exist on disk
                - relationship paths are valid
                - duplicate payload items were not returned
            """

            @staticmethod
            def validate_basic_counts(
                extracted_payload: dict[str, list[dict[str, Any]]],
            ) -> dict[str, Any]:
                counts = {
                    key: len(value)
                    for key, value in extracted_payload.items()
                }

                result = {
                    "check_name": "basic_payload_counts",
                    "check_status": "passed",
                    "counts": counts,
                }

                logger.debug(
                    "AUDIT_VALIDATION check=basic_payload_counts status=passed counts=%s",
                    counts,
                )

                return result
        '''
    ),

    "search_audit_evidence_builder.py": dedent(
        '''\
        from __future__ import annotations

        from typing import Any


        class SearchAuditEvidenceBuilder:
            """
            Builds relationship/evidence data explaining why a payload item was returned.

            This starts simple and can grow as the RAG relationship map becomes more formal.
            """

            @staticmethod
            def build_basic_evidence(
                *,
                item: dict[str, Any],
                item_type: str,
                relationship_map: dict[str, Any] | None = None,
                used_chunk_ids: list[int] | None = None,
            ) -> dict[str, Any]:
                return {
                    "item_type": item_type,
                    "payload_item": item,
                    "used_chunk_ids": used_chunk_ids or [],
                    "relationship_map": relationship_map or {},
                }
        '''
    ),

    "search_audit_ai_reviewer.py": dedent(
        '''\
        from __future__ import annotations

        from typing import Any


        class SearchAuditAIReviewer:
            """
            Future AI-based review layer.

            Purpose:
                Judge whether the returned search result actually matched the user's intent.

            This is different from technical validation.

            Technical validation asks:
                Did the returned IDs/files/relationships exist?

            AI review asks:
                Did the returned data make sense for the question?
            """

            @staticmethod
            def review_result_placeholder(
                *,
                question: str,
                answer: str | None,
                extracted_payload: dict[str, list[dict[str, Any]]],
            ) -> dict[str, Any]:
                return {
                    "review_status": "not_implemented",
                    "question": question,
                    "answer_present": bool(answer),
                    "payload_counts": {
                        key: len(value)
                        for key, value in extracted_payload.items()
                    },
                }
        '''
    ),

    "create_search_audit_schema.py": dedent(
        '''\
        # ============================================================
        # File:
        #   modules/ai/search_pathway/audit/create_search_audit_schema.py
        #
        # Purpose:
        #   Create PostgreSQL audit schema and tables for AI/search pathway auditing.
        #
        # Run from project root:
        #   cd E:\\emtac\\projects\\llm\\MDEV_EMTAC
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
        # SQL statements
        # ============================================================

        CREATE_SCHEMA_SQL = """
        CREATE SCHEMA IF NOT EXISTS audit;
        """


        CREATE_PGCRYPTO_SQL = """
        CREATE EXTENSION IF NOT EXISTS pgcrypto;
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

                    session = self.db_config.get_main_session()

                    logger.info("Creating audit schema.")
                    session.execute(text(CREATE_SCHEMA_SQL))
                    session.commit()

                    try:
                        logger.info("Creating PostgreSQL extension pgcrypto if available.")
                        session.execute(text(CREATE_PGCRYPTO_SQL))
                        session.commit()
                    except Exception as extension_error:
                        session.rollback()
                        logger.warning(
                            "Could not create pgcrypto extension. Continuing. Error: %s",
                            extension_error,
                        )

                    logger.info("Creating audit tables.")
                    for statement in ALL_TABLE_SQL:
                        session.execute(text(statement))

                    logger.info("Creating audit indexes.")
                    for statement in CREATE_INDEXES_SQL:
                        session.execute(text(statement))

                    session.commit()

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
        '''
    ),
}


# ============================================================
# Main creation logic
# ============================================================

def create_audit_folder() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Audit folder ready: %s", AUDIT_DIR)


def create_files() -> None:
    create_audit_folder()

    for file_name, content in FILES_TO_CREATE.items():
        file_path = AUDIT_DIR / file_name

        if file_path.exists() and not OVERWRITE_EXISTING:
            logger.info("Skipped existing file: %s", file_path)
            continue

        file_path.write_text(content, encoding="utf-8", newline="\n")
        logger.info("Created file: %s", file_path)


def main() -> int:
    logger.info("Starting search audit file setup.")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Audit directory: %s", AUDIT_DIR)
    logger.info("Overwrite existing files: %s", OVERWRITE_EXISTING)

    create_files()

    logger.info("Search audit starter files setup completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())