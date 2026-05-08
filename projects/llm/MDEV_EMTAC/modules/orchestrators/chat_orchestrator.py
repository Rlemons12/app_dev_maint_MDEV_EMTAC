from __future__ import annotations

import time
from typing import Dict, Any, Optional
from uuid import UUID

from modules.observability.high_end_tracer import tracer
from modules.decorators.trace_decorator import trace_entrypoint
from modules.orchestrators.base_orchestrator import BaseOrchestrator

from modules.services.ai_steward_manager_service import AIStewardManagerService
from modules.services.qanda_service import QandAService

from modules.ai.search_pathway.audit.search_audit_logger import (
    get_search_audit_log_manager,
)
from modules.ai.search_pathway.audit.search_audit_service import SearchAuditService
from modules.ai.search_pathway.audit.search_audit_types import SearchPathwayName

from modules.configuration.config import (
    FORCE_DEBUG_CHUNK,
    FORCE_DEBUG_CHUNK_ID,
)

from modules.configuration.log_config import (
    with_request_id,
    info_id,
    warning_id,
    error_id,
    debug_id,
)


class ChatOrchestrator(BaseOrchestrator):
    """
    Answer-only chat orchestrator.

    Responsibilities:
        - Own the database transaction/session boundary for answer generation
        - Coordinate RAG / AI answer generation
        - Persist Q&A seed data for later payload loading
        - Record search-pathway audit summary for answer-first path
        - Return the text answer quickly

    Does NOT:
        - Build documents/images/parts/drawings UI payload
        - Render frontend HTML
        - Read Flask request objects
        - Own route validation
    """

    DEFAULT_METHOD = "rag"
    AUDIT_PATHWAY_NAME = SearchPathwayName.RAG.value
    AUDIT_PATHWAY_VERSION = "1.0"

    EMPTY_BLOCKS = {
        "documents-container": [],
        "parts-container": [],
        "images-container": [],
        "drawings-container": [],
    }

    def __init__(
        self,
        *,
        ai_service: Optional[AIStewardManagerService] = None,
        qanda_service: Optional[QandAService] = None,
    ):
        super().__init__()

        self.ai_service = ai_service or AIStewardManagerService()
        self.qanda_service = qanda_service or QandAService()
        self.audit_log_manager = get_search_audit_log_manager()

    @with_request_id
    @trace_entrypoint(
        name="chat_answer_pipeline",
        deep_profile=False,
        capture_args=False,
        capture_return=False,
    )
    def handle_question(
        self,
        *,
        user_id: str,
        question: str,
        client_type: str,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        request_start = time.perf_counter()

        ai_time = 0.0
        persist_time = 0.0
        audit_time = 0.0

        ai_result: Dict[str, Any] = {}
        audit_summary: Dict[str, Any] = {}
        qanda_id: Optional[Any] = None

        normalized_user_id = (user_id or "anonymous").strip() or "anonymous"
        normalized_question = (question or "").strip()
        normalized_client_type = (client_type or "web").strip().lower() or "web"

        self.audit_log_manager.log_run_start(
            request_id=request_id or "unknown",
            pathway_name=self.AUDIT_PATHWAY_NAME,
            question=normalized_question,
        )

        try:
            forced_chunk_id = self._resolve_forced_chunk_id(
                request_id=request_id,
            )

            # --------------------------------------------------
            # 1. Generate answer only
            # --------------------------------------------------
            ai_start = time.perf_counter()

            with self.transaction() as session:
                with tracer.span(
                    "ai_answer_execute",
                    meta={
                        "user_id": normalized_user_id,
                        "client_type": normalized_client_type,
                        "forced_chunk_id": forced_chunk_id,
                        "include_payload": False,
                        "audit_pathway": self.AUDIT_PATHWAY_NAME,
                    },
                ):
                    ai_result = self.ai_service.execute(
                        session=session,
                        user_id=normalized_user_id,
                        question=normalized_question,
                        client_type=normalized_client_type,
                        request_id=request_id,
                        forced_chunk_id=forced_chunk_id,
                        include_payload=False,
                    )

            ai_time = time.perf_counter() - ai_start

            if not isinstance(ai_result, dict):
                raise ValueError("AI service returned invalid response format")

            self._apply_debug_metadata(
                ai_result=ai_result,
                forced_chunk_id=forced_chunk_id,
            )

            ai_result.setdefault("payload_status", "pending")

            # --------------------------------------------------
            # 2. Persist Q&A seed and record answer audit summary
            # --------------------------------------------------
            # The raw_response should include the answer, used_chunks,
            # chunks, and relationship_map. The payload orchestrator can
            # use this later to build the UI payload.
            persist_start = time.perf_counter()

            try:
                with self.transaction() as session:
                    with tracer.span("persist_qanda_seed"):
                        qanda_record = self.qanda_service.create_interaction(
                            session=session,
                            user_id=normalized_user_id,
                            question=normalized_question,
                            answer=ai_result.get("answer", ""),
                            request_id=request_id,
                            processing_time_ms=int(ai_time * 1000),
                            raw_response=ai_result,
                        )

                        qanda_id = self._extract_record_id(qanda_record)

                    audit_start = time.perf_counter()

                    with tracer.span(
                        "audit_answer_search_pathway",
                        meta={
                            "request_id": request_id,
                            "pathway_name": self.AUDIT_PATHWAY_NAME,
                            "qanda_id": str(qanda_id) if qanda_id else None,
                        },
                    ):
                        audit_summary = SearchAuditService.record_search_result(
                            session=session,
                            request_id=request_id or "unknown",
                            user_id=normalized_user_id,
                            session_id=None,
                            qanda_id=self._coerce_uuid_or_none(qanda_id),
                            question=normalized_question,
                            answer=ai_result.get("answer", ""),
                            response=ai_result,
                            pathway_name=self.AUDIT_PATHWAY_NAME,
                            pathway_version=self.AUDIT_PATHWAY_VERSION,
                            duration_ms=int((time.perf_counter() - request_start) * 1000),
                            model_name=ai_result.get("model_name"),
                        )

                    audit_time = time.perf_counter() - audit_start

            except Exception as persist_error:
                warning_id(
                    f"Q&A seed persistence or answer audit failed but answer response will continue: {persist_error}",
                    request_id,
                    exc_info=True,
                )

            persist_time = time.perf_counter() - persist_start

            # --------------------------------------------------
            # 3. Return answer-only response
            # --------------------------------------------------
            total_time = time.perf_counter() - request_start

            response = self._answer_response(
                ai_result=ai_result,
                request_id=request_id,
                client_type=normalized_client_type,
                total_time=total_time,
                ai_time=ai_time,
                persist_time=persist_time,
                audit_time=audit_time,
                audit_summary=audit_summary,
            )

            self.audit_log_manager.log_run_success(
                request_id=request_id or "unknown",
                pathway_name=self.AUDIT_PATHWAY_NAME,
                duration_ms=int(total_time * 1000),
                counts=(audit_summary or {}).get("counts"),
            )

            info_id(
                f"Chat answer processed in {total_time:.3f}s "
                f"(AI: {ai_time:.3f}s | Persist seed: {persist_time:.3f}s | "
                f"Audit: {audit_time:.3f}s)",
                request_id,
            )

            return response

        except Exception as e:
            total_time = time.perf_counter() - request_start

            self.audit_log_manager.log_run_failure(
                request_id=request_id or "unknown",
                pathway_name=self.AUDIT_PATHWAY_NAME,
                error=e,
                duration_ms=int(total_time * 1000),
            )

            error_id(
                f"ChatOrchestrator answer failure after {total_time:.3f}s: {e}",
                request_id,
                exc_info=True,
            )

            return self._error_response(
                request_id=request_id,
                total_time=total_time,
                ai_time=ai_time,
                persist_time=persist_time,
                audit_time=audit_time,
            )

    def _resolve_forced_chunk_id(
        self,
        *,
        request_id: Optional[str],
    ) -> Optional[int]:

        if not FORCE_DEBUG_CHUNK:
            return None

        if FORCE_DEBUG_CHUNK_ID is None:
            warning_id(
                "FORCE_DEBUG_CHUNK is enabled but FORCE_DEBUG_CHUNK_ID is None",
                request_id,
            )
            return None

        try:
            forced_chunk_id = int(FORCE_DEBUG_CHUNK_ID)
        except (TypeError, ValueError):
            warning_id(
                f"FORCE_DEBUG_CHUNK_ID is invalid: {FORCE_DEBUG_CHUNK_ID!r}",
                request_id,
            )
            return None

        debug_id(
            f"[ChatOrchestrator] DEBUG MODE ENABLED -> chunk={forced_chunk_id}",
            request_id,
        )

        return forced_chunk_id

    @staticmethod
    def _apply_debug_metadata(
        *,
        ai_result: Dict[str, Any],
        forced_chunk_id: Optional[int],
    ) -> None:

        if forced_chunk_id is not None:
            ai_result.setdefault("debug_mode", True)
            ai_result.setdefault("debug_chunk_id", forced_chunk_id)
        else:
            ai_result.setdefault("debug_mode", False)
            ai_result.setdefault("debug_chunk_id", None)

    def _answer_response(
        self,
        *,
        ai_result: Dict[str, Any],
        request_id: Optional[str],
        client_type: str,
        total_time: float,
        ai_time: float,
        persist_time: float,
        audit_time: float,
        audit_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        method = ai_result.get("method") or ai_result.get("strategy") or self.DEFAULT_METHOD
        strategy = ai_result.get("strategy") or method

        response: Dict[str, Any] = {
            "status": ai_result.get("status", "success"),
            "answer": ai_result.get("answer", ""),
            "method": method,
            "strategy": strategy,
            "request_id": request_id,
            "response_time": total_time,
            "payload_status": "pending",
            "payload_endpoint": "/ask/payload",
            "debug_mode": bool(ai_result.get("debug_mode", False)),
            "debug_chunk_id": ai_result.get("debug_chunk_id"),
            "retriever_top_k": ai_result.get("retriever_top_k"),
            "used_chunks_count": len(ai_result.get("used_chunks") or []),
            "blocks": {
                "documents-container": [],
                "parts-container": [],
                "images-container": [],
                "drawings-container": [],
            },
            "documents": [],
            "parts": [],
            "images": [],
            "drawings": [],
        }

        if client_type == "debug":
            response["performance"] = {
                "total_time": total_time,
                "ai_time": ai_time,
                "persist_time": persist_time,
                "audit_time": audit_time,
                "method": method,
                "strategy": strategy,
                "debug_mode": response["debug_mode"],
                "debug_chunk_id": response["debug_chunk_id"],
            }

            response["audit"] = {
                "pathway_name": self.AUDIT_PATHWAY_NAME,
                "pathway_version": self.AUDIT_PATHWAY_VERSION,
                "summary": audit_summary or {},
            }

        else:
            response["performance"] = {
                "total_time": total_time,
                "method": method,
                "strategy": strategy,
            }

        return response

    def _error_response(
        self,
        *,
        request_id: Optional[str],
        total_time: float,
        ai_time: float,
        persist_time: float,
        audit_time: float,
    ) -> Dict[str, Any]:

        return {
            "status": "error",
            "answer": "An unexpected error occurred while processing your request.",
            "method": "error",
            "strategy": "error",
            "request_id": request_id,
            "response_time": total_time,
            "payload_status": "unavailable",
            "payload_endpoint": None,
            "debug_mode": False,
            "debug_chunk_id": None,
            "blocks": {
                "documents-container": [],
                "parts-container": [],
                "images-container": [],
                "drawings-container": [],
            },
            "documents": [],
            "parts": [],
            "images": [],
            "drawings": [],
            "used_chunks_count": 0,
            "retriever_top_k": None,
            "performance": {
                "total_time": total_time,
                "ai_time": ai_time,
                "persist_time": persist_time,
                "audit_time": audit_time,
                "method": "error",
                "strategy": "error",
            },
        }

    @staticmethod
    def _extract_record_id(record: Any) -> Optional[Any]:
        """
        Extract an id from a Q&A record returned by QandAService.

        Supports:
            - ORM object with .id
            - dictionary with "id"
            - None
        """

        if record is None:
            return None

        if isinstance(record, dict):
            return record.get("id")

        return getattr(record, "id", None)

    @staticmethod
    def _coerce_uuid_or_none(value: Any) -> Optional[UUID]:
        """
        Coerce a value to UUID if possible.

        SearchAuditService currently accepts qanda_id as UUID | None.
        If a future QandAService returns a non-UUID ID, this safely returns None.
        """

        if value is None:
            return None

        if isinstance(value, UUID):
            return value

        try:
            return UUID(str(value))
        except (TypeError, ValueError, AttributeError):
            return None