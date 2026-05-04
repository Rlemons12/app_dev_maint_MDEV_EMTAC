from __future__ import annotations

import time
from typing import Dict, Any, Optional

from modules.observability.high_end_tracer import tracer
from modules.decorators.trace_decorator import trace_entrypoint
from modules.orchestrators.base_orchestrator import BaseOrchestrator

from modules.services.ai_steward_manager_service import AIStewardManagerService
from modules.services.qanda_service import QandAService

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
        - Return the text answer quickly

    Does NOT:
        - Build documents/images/parts/drawings UI payload
        - Render frontend HTML
        - Read Flask request objects
        - Own route validation
    """

    DEFAULT_METHOD = "rag"

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

        ai_result: Dict[str, Any] = {}

        normalized_user_id = (user_id or "anonymous").strip() or "anonymous"
        normalized_question = (question or "").strip()
        normalized_client_type = (client_type or "web").strip().lower() or "web"

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
            # 2. Persist Q&A seed
            # --------------------------------------------------
            # The raw_response should include the answer, used_chunks,
            # chunks, and relationship_map. The payload orchestrator can
            # use this later to build the UI payload.
            persist_start = time.perf_counter()

            try:
                with self.transaction() as session:
                    with tracer.span("persist_qanda_seed"):
                        self.qanda_service.create_interaction(
                            session=session,
                            user_id=normalized_user_id,
                            question=normalized_question,
                            answer=ai_result.get("answer", ""),
                            request_id=request_id,
                            processing_time_ms=int(ai_time * 1000),
                            raw_response=ai_result,
                        )

            except Exception as persist_error:
                warning_id(
                    f"Q&A seed persistence failed but answer response will continue: {persist_error}",
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
            )

            info_id(
                f"Chat answer processed in {total_time:.3f}s "
                f"(AI: {ai_time:.3f}s | Persist seed: {persist_time:.3f}s)",
                request_id,
            )

            return response

        except Exception as e:
            total_time = time.perf_counter() - request_start

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
                "method": method,
                "strategy": strategy,
                "debug_mode": response["debug_mode"],
                "debug_chunk_id": response["debug_chunk_id"],
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
                "method": "error",
                "strategy": "error",
            },
        }