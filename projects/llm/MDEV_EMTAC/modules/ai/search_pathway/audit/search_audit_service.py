from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import text

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

    The orchestrator or background worker owns transaction control.

    Design:
        - record_search_result() writes the audit run and lightweight validations.
        - Small payloads can be detailed inline.
        - Large payloads are deferred so /ask/payload can return quickly.
        - record_payload_items_for_existing_run() can be called by a background worker.

    This service writes to:

        audit.search_audit_run
        audit.search_audit_payload_item
        audit.search_audit_validation
    """

    INLINE_DETAIL_ITEM_THRESHOLD = 50

    PAYLOAD_WARNING_THRESHOLDS = {
        "chunks": 100,
        "documents": 25,
        "images": 50,
        "drawings": 100,
        "parts": 100,
    }

    # ============================================================
    # SQL
    # ============================================================

    INSERT_AUDIT_RUN_SQL = text(
        """
        INSERT INTO audit.search_audit_run (
            qanda_id,
            request_id,
            user_id,
            session_id,
            question,
            normalized_question,
            final_answer,
            pathway_name,
            pathway_version,
            search_mode,
            payload_status,
            validation_status,
            answer_hash,
            response_hash,
            payload_hash,
            model_name,
            raw_request,
            raw_response,
            raw_payload,
            raw_chunks,
            raw_relationship_map,
            validation_summary,
            started_at,
            completed_at,
            duration_ms,
            error_text
        )
        VALUES (
            :qanda_id,
            :request_id,
            :user_id,
            :session_id,
            :question,
            :normalized_question,
            :final_answer,
            :pathway_name,
            :pathway_version,
            :search_mode,
            :payload_status,
            :validation_status,
            :answer_hash,
            :response_hash,
            :payload_hash,
            :model_name,
            CAST(:raw_request AS JSONB),
            CAST(:raw_response AS JSONB),
            CAST(:raw_payload AS JSONB),
            CAST(:raw_chunks AS JSONB),
            CAST(:raw_relationship_map AS JSONB),
            CAST(:validation_summary AS JSONB),
            :started_at,
            :completed_at,
            :duration_ms,
            :error_text
        )
        RETURNING id
        """
    )

    INSERT_PAYLOAD_ITEM_SQL = text(
        """
        INSERT INTO audit.search_audit_payload_item (
            audit_run_id,
            item_type,
            source_table,
            source_id,
            title,
            label,
            file_path,
            url,
            rank,
            score,
            relationship_path,
            evidence,
            item_hash,
            exists_in_db,
            exists_on_disk,
            validation_status,
            validation_message,
            created_at
        )
        VALUES (
            :audit_run_id,
            :item_type,
            :source_table,
            :source_id,
            :title,
            :label,
            :file_path,
            :url,
            :rank,
            :score,
            :relationship_path,
            CAST(:evidence AS JSONB),
            :item_hash,
            :exists_in_db,
            :exists_on_disk,
            :validation_status,
            :validation_message,
            :created_at
        )
        """
    )

    INSERT_VALIDATION_SQL = text(
        """
        INSERT INTO audit.search_audit_validation (
            audit_run_id,
            check_name,
            check_status,
            expected_count,
            actual_count,
            details,
            created_at
        )
        VALUES (
            :audit_run_id,
            :check_name,
            :check_status,
            :expected_count,
            :actual_count,
            CAST(:details AS JSONB),
            :created_at
        )
        """
    )

    COUNT_PAYLOAD_ITEMS_SQL = text(
        """
        SELECT COUNT(*)
        FROM audit.search_audit_payload_item
        WHERE audit_run_id = :audit_run_id
        """
    )

    DELETE_PAYLOAD_ITEMS_SQL = text(
        """
        DELETE FROM audit.search_audit_payload_item
        WHERE audit_run_id = :audit_run_id
        """
    )

    UPDATE_AUDIT_RUN_VALIDATION_SQL = text(
        """
        UPDATE audit.search_audit_run
        SET
            validation_status = :validation_status,
            validation_summary = CAST(:validation_summary AS JSONB),
            completed_at = :completed_at
        WHERE id = :audit_run_id
        """
    )

    # ============================================================
    # Public API
    # ============================================================

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
        capture_items_inline: bool | None = None,
        inline_item_threshold: int = INLINE_DETAIL_ITEM_THRESHOLD,
        store_full_snapshot: bool = False,
    ) -> dict[str, Any]:
        """
        Record one search pathway audit result.

        Fast behavior:
            - Always inserts audit.search_audit_run.
            - Always inserts lightweight validation rows.
            - Inserts detailed item rows inline only for small payloads by default.

        Large payload behavior:
            - Large payload details are deferred.
            - Use record_payload_items_for_existing_run() from a background worker
              to insert every returned chunk/document/image/drawing/part later.

        This method does NOT:
            - create a session
            - commit
            - rollback
        """

        response = response or {}
        request_id = request_id or "unknown"

        audit_log_manager = get_search_audit_log_manager()

        extracted = SearchAuditPayloadExtractor.extract_all(response)
        counts = cls._build_counts(extracted)
        total_items = sum(counts.values())

        duplicates = cls._find_duplicate_items(extracted)
        size_warnings = cls._build_payload_size_warnings(counts)

        payload_present = bool(response)

        should_capture_inline = cls._should_capture_items_inline(
            capture_items_inline=capture_items_inline,
            total_items=total_items,
            inline_item_threshold=inline_item_threshold,
        )

        detail_capture_status = (
            "inline"
            if should_capture_inline
            else "deferred"
        )

        validation_status = cls._resolve_run_validation_status(
            payload_present=payload_present,
            duplicates=duplicates,
            size_warnings=size_warnings,
        )

        now = cls._utc_now()

        answer_hash = cls.stable_hash(answer or "")
        response_hash = cls.stable_hash(response)
        payload_hash = cls.stable_hash(
            {
                "counts": counts,
                "chunks": cls._summarize_items(extracted.get("chunks", [])),
                "documents": cls._summarize_items(extracted.get("documents", [])),
                "images": cls._summarize_items(extracted.get("images", [])),
                "drawings": cls._summarize_items(extracted.get("drawings", [])),
                "parts": cls._summarize_items(extracted.get("parts", [])),
            }
        )

        validation_summary = {
            "status": validation_status,
            "counts": counts,
            "total_items": total_items,
            "duplicates": duplicates,
            "size_warnings": size_warnings,
            "detail_capture_status": detail_capture_status,
            "checks": [
                "payload_exists",
                "payload_counts_captured",
                "no_duplicate_payload_items",
                "payload_size_reasonable",
                "payload_item_detail_capture",
            ],
        }

        raw_chunks = {
            "chunks": cls._summarize_items(extracted.get("chunks", [])),
            "used_chunks": cls._summarize_items(response.get("used_chunks") or []),
            "retrieved_chunks": cls._summarize_items(response.get("retrieved_chunks") or []),
        }

        raw_relationship_map = response.get("relationship_map") or {}

        raw_response = (
            response
            if store_full_snapshot
            else cls._build_light_response_snapshot(
                response=response,
                counts=counts,
                detail_capture_status=detail_capture_status,
            )
        )

        raw_payload = (
            {
                "documents": extracted.get("documents", []),
                "images": extracted.get("images", []),
                "drawings": extracted.get("drawings", []),
                "parts": extracted.get("parts", []),
            }
            if store_full_snapshot
            else {
                "counts": counts,
                "detail_capture_status": detail_capture_status,
                "documents_sample": cls._summarize_items(extracted.get("documents", []), limit=5),
                "images_sample": cls._summarize_items(extracted.get("images", []), limit=5),
                "drawings_sample": cls._summarize_items(extracted.get("drawings", []), limit=5),
                "parts_sample": cls._summarize_items(extracted.get("parts", []), limit=5),
            }
        )

        audit_run_id = cls._insert_audit_run(
            session=session,
            qanda_id=qanda_id,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            question=question,
            normalized_question=(question or "").strip(),
            final_answer=answer,
            pathway_name=pathway_name,
            pathway_version=pathway_version,
            search_mode=response.get("method") or response.get("strategy"),
            payload_status=response.get("payload_status", "unknown"),
            validation_status=validation_status,
            answer_hash=answer_hash,
            response_hash=response_hash,
            payload_hash=payload_hash,
            model_name=model_name or response.get("model_name"),
            raw_request={
                "request_id": request_id,
                "user_id": user_id,
                "session_id": str(session_id) if session_id else None,
                "qanda_id": str(qanda_id) if qanda_id else None,
                "question": question,
                "pathway_name": pathway_name,
                "pathway_version": pathway_version,
            },
            raw_response=raw_response,
            raw_payload=raw_payload,
            raw_chunks=raw_chunks,
            raw_relationship_map=raw_relationship_map,
            validation_summary=validation_summary,
            started_at=now,
            completed_at=now,
            duration_ms=duration_ms,
            error_text=None,
        )

        item_counts = {
            "chunks": 0,
            "documents": 0,
            "images": 0,
            "drawings": 0,
            "parts": 0,
        }

        if should_capture_inline:
            item_counts = cls._insert_payload_items(
                session=session,
                audit_run_id=audit_run_id,
                extracted=extracted,
                relationship_map_stored_on_run=bool(raw_relationship_map),
            )

        validation_rows = cls._insert_basic_validations(
            session=session,
            audit_run_id=audit_run_id,
            counts=counts,
            duplicates=duplicates,
            size_warnings=size_warnings,
            payload_present=payload_present,
            detail_capture_status=detail_capture_status,
            detail_rows_inserted=sum(item_counts.values()),
        )

        audit_summary = {
            "audit_run_id": str(audit_run_id),
            "request_id": request_id,
            "user_id": user_id,
            "session_id": str(session_id) if session_id else None,
            "qanda_id": str(qanda_id) if qanda_id else None,
            "question": question,
            "answer_hash": answer_hash,
            "response_hash": response_hash,
            "payload_hash": payload_hash,
            "pathway_name": pathway_name,
            "pathway_version": pathway_version,
            "model_name": model_name or response.get("model_name"),
            "duration_ms": duration_ms,
            "counts": counts,
            "total_items": total_items,
            "item_rows_inserted": item_counts,
            "validation_rows_inserted": validation_rows,
            "validation_status": validation_status,
            "duplicates": duplicates,
            "size_warnings": size_warnings,
            "detail_capture_status": detail_capture_status,
            "detail_capture_deferred": not should_capture_inline,
            "created_at": now.isoformat(),
        }

        audit_log_manager.log_payload_counts(
            request_id=request_id,
            pathway_name=pathway_name,
            counts=counts,
        )

        audit_log_manager.log_validation_result(
            request_id=request_id,
            pathway_name=pathway_name,
            check_name="search_audit_run_persisted",
            check_status=validation_status,
            details={
                "audit_run_id": str(audit_run_id),
                "detail_capture_status": detail_capture_status,
                "item_rows_inserted": item_counts,
                "validation_rows_inserted": validation_rows,
                "total_items": total_items,
            },
        )

        return audit_summary

    @classmethod
    def record_payload_items_for_existing_run(
        cls,
        *,
        session,
        audit_run_id: UUID | str,
        response: dict[str, Any],
        request_id: str = "unknown",
        pathway_name: str = "unknown",
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        """
        Insert detailed payload item rows for an existing audit run.

        Intended caller:
            SearchAuditBackgroundWorker

        This method:
            - extracts chunks/documents/images/drawings/parts from response
            - batch inserts item rows
            - inserts a validation row showing detailed capture completed
            - updates the audit run validation summary

        This method does NOT:
            - create a session
            - commit
            - rollback
        """

        response = response or {}
        request_id = request_id or "unknown"

        existing_count = cls._count_existing_payload_items(
            session=session,
            audit_run_id=audit_run_id,
        )

        if existing_count and not replace_existing:
            details = {
                "message": "Payload item details already exist. Skipping insert.",
                "existing_count": existing_count,
                "replace_existing": replace_existing,
            }

            cls._insert_validation_row(
                session=session,
                audit_run_id=audit_run_id,
                check_name="payload_item_details_already_exist",
                check_status=SearchAuditValidationStatus.WARNING.value,
                expected_count=None,
                actual_count=existing_count,
                details=details,
            )

            return {
                "audit_run_id": str(audit_run_id),
                "request_id": request_id,
                "pathway_name": pathway_name,
                "status": "skipped_existing_items",
                "existing_count": existing_count,
                "item_rows_inserted": {
                    "chunks": 0,
                    "documents": 0,
                    "images": 0,
                    "drawings": 0,
                    "parts": 0,
                },
            }

        if existing_count and replace_existing:
            session.execute(
                cls.DELETE_PAYLOAD_ITEMS_SQL,
                {"audit_run_id": audit_run_id},
            )

        extracted = SearchAuditPayloadExtractor.extract_all(response)
        counts = cls._build_counts(extracted)
        duplicates = cls._find_duplicate_items(extracted)
        size_warnings = cls._build_payload_size_warnings(counts)

        item_counts = cls._insert_payload_items(
            session=session,
            audit_run_id=audit_run_id,
            extracted=extracted,
            relationship_map_stored_on_run=bool(response.get("relationship_map")),
        )

        total_inserted = sum(item_counts.values())

        cls._insert_validation_row(
            session=session,
            audit_run_id=audit_run_id,
            check_name="payload_item_details_inserted",
            check_status=SearchAuditValidationStatus.PASSED.value,
            expected_count=sum(counts.values()),
            actual_count=total_inserted,
            details={
                "message": "Detailed payload item rows inserted.",
                "counts": counts,
                "item_rows_inserted": item_counts,
                "replace_existing": replace_existing,
            },
        )

        final_validation_status = cls._resolve_run_validation_status(
            payload_present=bool(response),
            duplicates=duplicates,
            size_warnings=size_warnings,
        )

        validation_summary = {
            "status": final_validation_status,
            "counts": counts,
            "total_items": sum(counts.values()),
            "duplicates": duplicates,
            "size_warnings": size_warnings,
            "detail_capture_status": "complete",
            "detail_rows_inserted": item_counts,
            "checks": [
                "payload_exists",
                "payload_counts_captured",
                "no_duplicate_payload_items",
                "payload_size_reasonable",
                "payload_item_details_inserted",
            ],
        }

        cls._update_audit_run_validation_summary(
            session=session,
            audit_run_id=audit_run_id,
            validation_status=final_validation_status,
            validation_summary=validation_summary,
        )

        get_search_audit_log_manager().log_validation_result(
            request_id=request_id,
            pathway_name=pathway_name,
            check_name="payload_item_details_inserted",
            check_status=final_validation_status,
            details={
                "audit_run_id": str(audit_run_id),
                "item_rows_inserted": item_counts,
                "total_inserted": total_inserted,
            },
        )

        return {
            "audit_run_id": str(audit_run_id),
            "request_id": request_id,
            "pathway_name": pathway_name,
            "status": "inserted",
            "counts": counts,
            "item_rows_inserted": item_counts,
            "total_inserted": total_inserted,
            "validation_status": final_validation_status,
            "duplicates": duplicates,
            "size_warnings": size_warnings,
        }

    # ============================================================
    # Hash / JSON helpers
    # ============================================================

    @staticmethod
    def stable_hash(value: Any) -> str:
        serialized = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, sort_keys=True, default=str)

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    # ============================================================
    # Insert / update methods
    # ============================================================

    @classmethod
    def _insert_audit_run(
        cls,
        *,
        session,
        qanda_id: UUID | None,
        request_id: str,
        user_id: str | None,
        session_id: UUID | None,
        question: str | None,
        normalized_question: str | None,
        final_answer: str | None,
        pathway_name: str,
        pathway_version: str,
        search_mode: str | None,
        payload_status: str,
        validation_status: str,
        answer_hash: str,
        response_hash: str,
        payload_hash: str,
        model_name: str | None,
        raw_request: dict[str, Any],
        raw_response: dict[str, Any],
        raw_payload: dict[str, Any],
        raw_chunks: dict[str, Any],
        raw_relationship_map: dict[str, Any],
        validation_summary: dict[str, Any],
        started_at: datetime,
        completed_at: datetime,
        duration_ms: int | None,
        error_text: str | None,
    ) -> UUID:
        result = session.execute(
            cls.INSERT_AUDIT_RUN_SQL,
            {
                "qanda_id": qanda_id,
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
                "question": question,
                "normalized_question": normalized_question,
                "final_answer": final_answer,
                "pathway_name": pathway_name,
                "pathway_version": pathway_version,
                "search_mode": search_mode,
                "payload_status": payload_status,
                "validation_status": validation_status,
                "answer_hash": answer_hash,
                "response_hash": response_hash,
                "payload_hash": payload_hash,
                "model_name": model_name,
                "raw_request": cls._json_dumps(raw_request),
                "raw_response": cls._json_dumps(raw_response),
                "raw_payload": cls._json_dumps(raw_payload),
                "raw_chunks": cls._json_dumps(raw_chunks),
                "raw_relationship_map": cls._json_dumps(raw_relationship_map),
                "validation_summary": cls._json_dumps(validation_summary),
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_ms": duration_ms,
                "error_text": error_text,
            },
        )

        return result.scalar_one()

    @classmethod
    def _insert_payload_items(
        cls,
        *,
        session,
        audit_run_id: UUID | str,
        extracted: dict[str, list[dict[str, Any]]],
        relationship_map_stored_on_run: bool,
    ) -> dict[str, int]:
        inserted_counts = {
            "chunks": 0,
            "documents": 0,
            "images": 0,
            "drawings": 0,
            "parts": 0,
        }

        rows: list[dict[str, Any]] = []

        item_groups = [
            ("chunks", "chunk", "document"),
            ("documents", "document", "complete_document"),
            ("images", "image", "image"),
            ("drawings", "drawing", "drawing"),
            ("parts", "part", "part"),
        ]

        for group_key, item_type, source_table in item_groups:
            items = extracted.get(group_key, [])

            for rank, item in enumerate(items, start=1):
                rows.append(
                    cls._build_payload_item_row(
                        audit_run_id=audit_run_id,
                        item=item,
                        item_type=item_type,
                        source_table=source_table,
                        rank=rank,
                        relationship_map_stored_on_run=relationship_map_stored_on_run,
                    )
                )

                inserted_counts[group_key] += 1

        if rows:
            session.execute(cls.INSERT_PAYLOAD_ITEM_SQL, rows)

        return inserted_counts

    @classmethod
    def _build_payload_item_row(
        cls,
        *,
        audit_run_id: UUID | str,
        item: dict[str, Any],
        item_type: str,
        source_table: str,
        rank: int,
        relationship_map_stored_on_run: bool,
    ) -> dict[str, Any]:
        source_id = cls._extract_source_id(item=item, item_type=item_type)
        title = cls._extract_title(item=item, item_type=item_type)
        label = cls._extract_label(item=item, item_type=item_type)
        file_path = cls._extract_file_path(item)
        url = cls._extract_url(item)
        score = cls._extract_score(item)
        relationship_path = cls._extract_relationship_path(item)

        item_hash = cls.stable_hash(
            {
                "item_type": item_type,
                "source_table": source_table,
                "source_id": source_id,
                "item": item,
            }
        )

        evidence = {
            "item_type": item_type,
            "source_table": source_table,
            "source_id": source_id,
            "rank": rank,
            "payload_item_hash": item_hash,
            "payload_item_keys": sorted(str(key) for key in item.keys()),
            "relationship_map_stored_on_run": relationship_map_stored_on_run,
        }

        return {
            "audit_run_id": audit_run_id,
            "item_type": item_type,
            "source_table": source_table,
            "source_id": source_id,
            "title": title,
            "label": label,
            "file_path": file_path,
            "url": url,
            "rank": rank,
            "score": score,
            "relationship_path": relationship_path,
            "evidence": cls._json_dumps(evidence),
            "item_hash": item_hash,
            "exists_in_db": None,
            "exists_on_disk": None,
            "validation_status": SearchAuditValidationStatus.NOT_VALIDATED.value,
            "validation_message": "Technical relationship validation has not run yet.",
            "created_at": cls._utc_now(),
        }

    @classmethod
    def _insert_basic_validations(
        cls,
        *,
        session,
        audit_run_id: UUID | str,
        counts: dict[str, int],
        duplicates: dict[str, list[Any]],
        size_warnings: dict[str, Any],
        payload_present: bool,
        detail_capture_status: str,
        detail_rows_inserted: int,
    ) -> int:
        validations = []

        validations.append(
            {
                "check_name": "payload_exists",
                "check_status": (
                    SearchAuditValidationStatus.PASSED.value
                    if payload_present
                    else SearchAuditValidationStatus.FAILED.value
                ),
                "expected_count": 1,
                "actual_count": 1 if payload_present else 0,
                "details": {
                    "message": (
                        "Payload/response was present."
                        if payload_present
                        else "Payload/response was empty."
                    )
                },
            }
        )

        total_items = sum(counts.values())

        validations.append(
            {
                "check_name": "payload_counts_captured",
                "check_status": SearchAuditValidationStatus.PASSED.value,
                "expected_count": None,
                "actual_count": total_items,
                "details": {
                    "counts": counts,
                    "message": "Payload item counts captured.",
                },
            }
        )

        duplicate_total = sum(len(values) for values in duplicates.values())

        validations.append(
            {
                "check_name": "no_duplicate_payload_items",
                "check_status": (
                    SearchAuditValidationStatus.PASSED.value
                    if duplicate_total == 0
                    else SearchAuditValidationStatus.WARNING.value
                ),
                "expected_count": 0,
                "actual_count": duplicate_total,
                "details": {
                    "duplicates": duplicates,
                    "message": (
                        "No duplicate payload items detected."
                        if duplicate_total == 0
                        else "Duplicate payload items detected."
                    ),
                },
            }
        )

        size_warning_total = len(size_warnings)

        validations.append(
            {
                "check_name": "payload_size_reasonable",
                "check_status": (
                    SearchAuditValidationStatus.PASSED.value
                    if size_warning_total == 0
                    else SearchAuditValidationStatus.WARNING.value
                ),
                "expected_count": 0,
                "actual_count": size_warning_total,
                "details": {
                    "thresholds": cls.PAYLOAD_WARNING_THRESHOLDS,
                    "warnings": size_warnings,
                    "message": (
                        "Payload size is within configured warning thresholds."
                        if size_warning_total == 0
                        else "Payload size exceeded one or more warning thresholds."
                    ),
                },
            }
        )

        validations.append(
            {
                "check_name": "payload_item_detail_capture",
                "check_status": (
                    SearchAuditValidationStatus.PASSED.value
                    if detail_capture_status == "inline"
                    else SearchAuditValidationStatus.NOT_VALIDATED.value
                ),
                "expected_count": total_items,
                "actual_count": detail_rows_inserted,
                "details": {
                    "detail_capture_status": detail_capture_status,
                    "detail_rows_inserted": detail_rows_inserted,
                    "message": (
                        "Payload item details captured inline."
                        if detail_capture_status == "inline"
                        else "Payload item details deferred for background capture."
                    ),
                },
            }
        )

        for validation in validations:
            cls._insert_validation_row(
                session=session,
                audit_run_id=audit_run_id,
                check_name=validation["check_name"],
                check_status=validation["check_status"],
                expected_count=validation["expected_count"],
                actual_count=validation["actual_count"],
                details=validation["details"],
            )

        return len(validations)

    @classmethod
    def _insert_validation_row(
        cls,
        *,
        session,
        audit_run_id: UUID | str,
        check_name: str,
        check_status: str,
        expected_count: int | None,
        actual_count: int | None,
        details: dict[str, Any],
    ) -> None:
        session.execute(
            cls.INSERT_VALIDATION_SQL,
            {
                "audit_run_id": audit_run_id,
                "check_name": check_name,
                "check_status": check_status,
                "expected_count": expected_count,
                "actual_count": actual_count,
                "details": cls._json_dumps(details),
                "created_at": cls._utc_now(),
            },
        )

    @classmethod
    def _count_existing_payload_items(
        cls,
        *,
        session,
        audit_run_id: UUID | str,
    ) -> int:
        return int(
            session.execute(
                cls.COUNT_PAYLOAD_ITEMS_SQL,
                {"audit_run_id": audit_run_id},
            ).scalar()
            or 0
        )

    @classmethod
    def _update_audit_run_validation_summary(
        cls,
        *,
        session,
        audit_run_id: UUID | str,
        validation_status: str,
        validation_summary: dict[str, Any],
    ) -> None:
        session.execute(
            cls.UPDATE_AUDIT_RUN_VALIDATION_SQL,
            {
                "audit_run_id": audit_run_id,
                "validation_status": validation_status,
                "validation_summary": cls._json_dumps(validation_summary),
                "completed_at": cls._utc_now(),
            },
        )

    # ============================================================
    # Count / validation helpers
    # ============================================================

    @staticmethod
    def _build_counts(
        extracted: dict[str, list[dict[str, Any]]],
    ) -> dict[str, int]:
        return {
            "chunks": len(extracted.get("chunks", [])),
            "documents": len(extracted.get("documents", [])),
            "images": len(extracted.get("images", [])),
            "drawings": len(extracted.get("drawings", [])),
            "parts": len(extracted.get("parts", [])),
        }

    @classmethod
    def _build_payload_size_warnings(
        cls,
        counts: dict[str, int],
    ) -> dict[str, Any]:
        warnings: dict[str, Any] = {}

        for key, threshold in cls.PAYLOAD_WARNING_THRESHOLDS.items():
            actual = counts.get(key, 0)

            if actual > threshold:
                warnings[key] = {
                    "actual": actual,
                    "threshold": threshold,
                    "over_by": actual - threshold,
                }

        return warnings

    @classmethod
    def _resolve_run_validation_status(
        cls,
        *,
        payload_present: bool,
        duplicates: dict[str, list[Any]],
        size_warnings: dict[str, Any],
    ) -> str:
        if not payload_present:
            return SearchAuditValidationStatus.FAILED.value

        if duplicates or size_warnings:
            return SearchAuditValidationStatus.WARNING.value

        return SearchAuditValidationStatus.PASSED.value

    @staticmethod
    def _should_capture_items_inline(
        *,
        capture_items_inline: bool | None,
        total_items: int,
        inline_item_threshold: int,
    ) -> bool:
        if capture_items_inline is not None:
            return bool(capture_items_inline)

        return total_items <= inline_item_threshold

    @classmethod
    def _find_duplicate_items(
        cls,
        extracted: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[Any]]:
        duplicates: dict[str, list[Any]] = {
            "chunks": [],
            "documents": [],
            "images": [],
            "drawings": [],
            "parts": [],
        }

        item_type_by_group = {
            "chunks": "chunk",
            "documents": "document",
            "images": "image",
            "drawings": "drawing",
            "parts": "part",
        }

        for group_key, items in extracted.items():
            seen = set()
            repeated = set()
            item_type = item_type_by_group.get(group_key, group_key)

            for item in items:
                source_id = cls._extract_source_id(
                    item=item,
                    item_type=item_type,
                )

                dedupe_key = source_id if source_id is not None else cls.stable_hash(item)

                if dedupe_key in seen:
                    repeated.add(dedupe_key)
                else:
                    seen.add(dedupe_key)

            duplicates[group_key] = sorted(
                [str(value) for value in repeated]
            )

        return {
            key: value
            for key, value in duplicates.items()
            if value
        }

    # ============================================================
    # Snapshot helpers
    # ============================================================

    @classmethod
    def _build_light_response_snapshot(
        cls,
        *,
        response: dict[str, Any],
        counts: dict[str, int],
        detail_capture_status: str,
    ) -> dict[str, Any]:
        answer = response.get("answer")

        if isinstance(answer, str) and len(answer) > 1000:
            answer_preview = answer[:1000] + "...[truncated]"
        else:
            answer_preview = answer

        return {
            "status": response.get("status"),
            "method": response.get("method"),
            "strategy": response.get("strategy"),
            "payload_status": response.get("payload_status"),
            "retriever_top_k": response.get("retriever_top_k"),
            "debug_mode": response.get("debug_mode"),
            "debug_chunk_id": response.get("debug_chunk_id"),
            "model_name": response.get("model_name"),
            "answer_preview": answer_preview,
            "counts": counts,
            "detail_capture_status": detail_capture_status,
            "has_relationship_map": isinstance(response.get("relationship_map"), dict),
            "relationship_summary": response.get("relationship_summary"),
        }

    @classmethod
    def _summarize_items(
        cls,
        items: Any,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []

        selected_items = items if limit is None else items[:limit]

        summaries = []

        for item in selected_items:
            if not isinstance(item, dict):
                continue

            item_type = cls._guess_item_type(item)

            summaries.append(
                {
                    "source_id": cls._extract_source_id(
                        item=item,
                        item_type=item_type,
                    ),
                    "title": cls._extract_title(
                        item=item,
                        item_type=item_type,
                    ),
                    "file_path": cls._extract_file_path(item),
                    "url": cls._extract_url(item),
                    "score": cls._extract_score(item),
                    "keys": sorted(str(key) for key in item.keys()),
                }
            )

        return summaries

    @staticmethod
    def _guess_item_type(item: dict[str, Any]) -> str:
        keys = set(item.keys())

        if {"drawing_id", "drw_id", "drawing_name", "drw_name", "drawing_number", "drw_number"} & keys:
            return "drawing"

        if {"image_id", "image_title"} & keys:
            return "image"

        if {"part_id", "part_number", "item_number"} & keys:
            return "part"

        if {"chunk_id"} & keys:
            return "chunk"

        if {"complete_document_id", "document_id", "document_title"} & keys:
            return "document"

        return "document"

    # ============================================================
    # Payload item extraction helpers
    # ============================================================

    @staticmethod
    def _extract_source_id(
        *,
        item: dict[str, Any],
        item_type: str,
    ) -> int | None:
        possible_keys = [
            "id",
            "source_id",
            "db_id",
            f"{item_type}_id",
        ]

        if item_type == "chunk":
            possible_keys.extend(
                [
                    "chunk_id",
                    "document_id",
                    "doc_id",
                ]
            )

        if item_type == "document":
            possible_keys.extend(
                [
                    "complete_document_id",
                    "document_id",
                    "doc_id",
                ]
            )

        if item_type == "image":
            possible_keys.extend(["image_id"])

        if item_type == "drawing":
            possible_keys.extend(["drawing_id", "drw_id"])

        if item_type == "part":
            possible_keys.extend(["part_id"])

        for key in possible_keys:
            value = item.get(key)
            coerced = SearchAuditService._coerce_int_or_none(value)

            if coerced is not None:
                return coerced

        return None

    @staticmethod
    def _extract_title(
        *,
        item: dict[str, Any],
        item_type: str,
    ) -> str | None:
        possible_keys = [
            "title",
            "name",
            "label",
            "display_name",
        ]

        if item_type in {"chunk", "document"}:
            possible_keys.extend(
                [
                    "document_title",
                    "complete_document_title",
                    "filename",
                    "file_name",
                ]
            )

        if item_type == "drawing":
            possible_keys.extend(
                [
                    "drawing_name",
                    "drw_name",
                    "drawing_number",
                    "drw_number",
                ]
            )

        if item_type == "part":
            possible_keys.extend(
                [
                    "part_number",
                    "item_number",
                    "description",
                    "name",
                ]
            )

        if item_type == "image":
            possible_keys.extend(
                [
                    "image_title",
                    "filename",
                    "file_name",
                ]
            )

        for key in possible_keys:
            value = item.get(key)

            if value is not None:
                text_value = str(value).strip()

                if text_value:
                    return text_value

        return None

    @staticmethod
    def _extract_label(
        *,
        item: dict[str, Any],
        item_type: str,
    ) -> str | None:
        for key in ("label", "display_label", "caption"):
            value = item.get(key)

            if value is not None:
                text_value = str(value).strip()

                if text_value:
                    return text_value

        return SearchAuditService._extract_title(item=item, item_type=item_type)

    @staticmethod
    def _extract_file_path(item: dict[str, Any]) -> str | None:
        for key in (
            "file_path",
            "path",
            "local_path",
            "absolute_path",
            "relative_path",
            "src",
        ):
            value = item.get(key)

            if value is not None:
                text_value = str(value).strip()

                if text_value:
                    return text_value

        return None

    @staticmethod
    def _extract_url(item: dict[str, Any]) -> str | None:
        for key in (
            "url",
            "href",
            "link",
            "viewer_url",
            "download_url",
        ):
            value = item.get(key)

            if value is not None:
                text_value = str(value).strip()

                if text_value:
                    return text_value

        return None

    @staticmethod
    def _extract_score(item: dict[str, Any]) -> float | None:
        for key in (
            "score",
            "similarity",
            "similarity_score",
            "distance",
            "rank_score",
        ):
            value = item.get(key)

            if value is None:
                continue

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _extract_relationship_path(item: dict[str, Any]) -> str | None:
        value = item.get("relationship_path")

        if value is None:
            return None

        if isinstance(value, str):
            text_value = value.strip()
            return text_value or None

        if isinstance(value, list):
            return " -> ".join(str(part) for part in value)

        return str(value)

    @staticmethod
    def _coerce_int_or_none(value: Any) -> int | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None