from __future__ import annotations

import time
from typing import Dict, Any, Optional

from modules.observability.high_end_tracer import tracer
from modules.decorators.trace_decorator import trace_entrypoint
from modules.orchestrators.base_orchestrator import BaseOrchestrator

from modules.services.ai_steward_manager_service import AIStewardManagerService
from modules.services.qanda_service import QandAService

from modules.configuration.log_config import (
    with_request_id,
    info_id,
    warning_id,
    error_id,
    debug_id,
)


class ChatPayloadOrchestrator(BaseOrchestrator):
    """
    Supporting-payload orchestrator.

    Responsibilities:
        - Load answer/RAG seed data from Q&A storage or request payload
        - Build documents/images/parts/drawings UI payload
        - Return only supporting UI payload

    Does NOT:
        - Generate the text answer
        - Persist a new Q&A interaction
        - Render frontend HTML
        - Read Flask request objects directly
    """

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
        name="chat_payload_pipeline",
        deep_profile=False,
        capture_args=False,
        capture_return=False,
    )
    def load_payload(
        self,
        *,
        request_id: Optional[str],
        payload_seed: Optional[Dict[str, Any]] = None,
        client_type: str = "web",
    ) -> Dict[str, Any]:

        request_start = time.perf_counter()
        payload_time = 0.0

        normalized_client_type = (client_type or "web").strip().lower() or "web"

        try:
            # --------------------------------------------------
            # 1. Resolve payload seed
            # --------------------------------------------------
            seed_result = payload_seed

            if not isinstance(seed_result, dict) or not seed_result:
                seed_result = self._load_seed_result_from_qanda(
                    request_id=request_id,
                )

            if not isinstance(seed_result, dict) or not seed_result:
                warning_id(
                    "[ChatPayloadOrchestrator] No payload seed available.",
                    request_id,
                )
                return self._empty_payload_response(
                    request_id=request_id,
                    status="not_found",
                    payload_status="unavailable",
                    message="No payload seed was found for this request.",
                    total_time=time.perf_counter() - request_start,
                    payload_time=payload_time,
                )

            # --------------------------------------------------
            # 2. Project payload
            # --------------------------------------------------
            payload_start = time.perf_counter()

            with self.transaction() as session:
                with tracer.span(
                    "chat_payload_project",
                    meta={
                        "request_id": request_id,
                        "client_type": normalized_client_type,
                    },
                ):
                    projected_result = self.ai_service.project_payload(
                        session=session,
                        result=seed_result,
                        request_id=request_id,
                    )

            payload_time = time.perf_counter() - payload_start

            if not isinstance(projected_result, dict):
                raise ValueError("AI service returned invalid payload format")

            # --------------------------------------------------
            # 3. Return payload-only response
            # --------------------------------------------------
            total_time = time.perf_counter() - request_start

            response = self._payload_response(
                result=projected_result,
                request_id=request_id,
                client_type=normalized_client_type,
                total_time=total_time,
                payload_time=payload_time,
            )

            info_id(
                f"Chat payload processed in {total_time:.3f}s "
                f"(Payload: {payload_time:.3f}s | "
                f"docs={len(response.get('documents') or [])} | "
                f"images={len(response.get('images') or [])} | "
                f"parts={len(response.get('parts') or [])} | "
                f"drawings={len(response.get('drawings') or [])})",
                request_id,
            )

            return response

        except Exception as e:
            total_time = time.perf_counter() - request_start

            error_id(
                f"ChatPayloadOrchestrator failure after {total_time:.3f}s: {e}",
                request_id,
                exc_info=True,
            )

            return self._empty_payload_response(
                request_id=request_id,
                status="error",
                payload_status="error",
                message="An unexpected error occurred while loading supporting payload.",
                total_time=total_time,
                payload_time=payload_time,
            )

    def _load_seed_result_from_qanda(
        self,
        *,
        request_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Loads the raw AI/RAG result saved by ChatOrchestrator.

        Preferred QandAService method:
            get_raw_response_by_request_id(session=session, request_id=request_id)

        Fallback supported method:
            get_interaction_by_request_id(session=session, request_id=request_id)

        If neither method exists yet, add one to QandAService.
        """

        if not request_id:
            return {}

        with self.transaction() as session:
            if hasattr(self.qanda_service, "get_raw_response_by_request_id"):
                raw_response = self.qanda_service.get_raw_response_by_request_id(
                    session=session,
                    request_id=request_id,
                )

                if isinstance(raw_response, dict):
                    debug_id(
                        "[ChatPayloadOrchestrator] Loaded raw_response by request_id.",
                        request_id,
                    )
                    return raw_response

                return {}

            if hasattr(self.qanda_service, "get_interaction_by_request_id"):
                interaction = self.qanda_service.get_interaction_by_request_id(
                    session=session,
                    request_id=request_id,
                )

                raw_response = getattr(interaction, "raw_response", None)

                if isinstance(raw_response, dict):
                    debug_id(
                        "[ChatPayloadOrchestrator] Loaded interaction.raw_response by request_id.",
                        request_id,
                    )
                    return raw_response

                if isinstance(interaction, dict):
                    raw_response = interaction.get("raw_response")
                    if isinstance(raw_response, dict):
                        debug_id(
                            "[ChatPayloadOrchestrator] Loaded dict raw_response by request_id.",
                            request_id,
                        )
                        return raw_response

                return {}

        warning_id(
            "[ChatPayloadOrchestrator] QandAService does not provide a seed lookup method. "
            "Add get_raw_response_by_request_id() or pass payload_seed directly.",
            request_id,
        )

        return {}

    def _payload_response(
        self,
        *,
        result: Dict[str, Any],
        request_id: Optional[str],
        client_type: str,
        total_time: float,
        payload_time: float,
    ) -> Dict[str, Any]:

        documents = result.get("documents") or []
        parts = result.get("parts") or []
        images = result.get("images") or []
        drawings = result.get("drawings") or []

        blocks = {
            "documents-container": documents,
            "parts-container": parts,
            "images-container": images,
            "drawings-container": drawings,
        }

        response: Dict[str, Any] = {
            "status": "success",
            "request_id": request_id,
            "payload_status": "complete",
            "message": "Supporting payload loaded.",
            "blocks": blocks,
            "documents": documents,
            "parts": parts,
            "images": images,
            "drawings": drawings,
            "relationship_summary": result.get("relationship_summary"),
            "used_chunks_count": len(result.get("used_chunks") or []),
            "retriever_top_k": result.get("retriever_top_k"),
            "response_time": total_time,
            "performance": {
                "total_time": total_time,
                "payload_time": payload_time,
                "documents": len(documents),
                "parts": len(parts),
                "images": len(images),
                "drawings": len(drawings),
            },
        }

        if client_type == "debug":
            response["debug"] = {
                "method": result.get("method"),
                "strategy": result.get("strategy"),
                "debug_mode": bool(result.get("debug_mode", False)),
                "debug_chunk_id": result.get("debug_chunk_id"),
                "has_relationship_map": isinstance(result.get("relationship_map"), dict),
                "chunks_count": len(result.get("chunks") or []),
                "used_chunks_count": len(result.get("used_chunks") or []),
            }

        return response

    def _empty_payload_response(
        self,
        *,
        request_id: Optional[str],
        status: str,
        payload_status: str,
        message: str,
        total_time: float,
        payload_time: float,
    ) -> Dict[str, Any]:

        return {
            "status": status,
            "request_id": request_id,
            "payload_status": payload_status,
            "message": message,
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
            "relationship_summary": None,
            "used_chunks_count": 0,
            "retriever_top_k": None,
            "response_time": total_time,
            "performance": {
                "total_time": total_time,
                "payload_time": payload_time,
                "documents": 0,
                "parts": 0,
                "images": 0,
                "drawings": 0,
            },
        }