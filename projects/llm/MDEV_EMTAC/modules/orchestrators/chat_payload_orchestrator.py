from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

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

    Performance notes:
        - This orchestrator should stay thin.
        - Heavy database relationship work should live in services.
        - ai_service.project_payload() is the main delegated payload projection step.
    """

    EMPTY_BLOCKS = {
        "documents-container": [],
        "parts-container": [],
        "images-container": [],
        "drawings-container": [],
    }

    VALID_CLIENT_TYPES = {"web", "debug", "api"}

    def __init__(
        self,
        *,
        ai_service: Optional[AIStewardManagerService] = None,
        qanda_service: Optional[QandAService] = None,
    ) -> None:
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
        seed_time = 0.0
        payload_time = 0.0
        seed_source = "none"

        normalized_client_type = self._normalize_client_type(client_type)

        try:
            # --------------------------------------------------
            # 1. Resolve payload seed
            # --------------------------------------------------
            seed_start = time.perf_counter()

            seed_result, seed_source = self._resolve_payload_seed(
                request_id=request_id,
                payload_seed=payload_seed,
            )

            seed_time = time.perf_counter() - seed_start

            if not seed_result:
                warning_id(
                    "[ChatPayloadOrchestrator] No payload seed available. "
                    f"seed_source={seed_source} | seed_time={seed_time:.3f}s",
                    request_id,
                )

                return self._empty_payload_response(
                    request_id=request_id,
                    status="not_found",
                    payload_status="unavailable",
                    message="No payload seed was found for this request.",
                    total_time=time.perf_counter() - request_start,
                    seed_time=seed_time,
                    payload_time=payload_time,
                    seed_source=seed_source,
                )

            debug_id(
                "[ChatPayloadOrchestrator] Payload seed resolved. "
                f"seed_source={seed_source} | seed_keys={sorted(seed_result.keys())} | "
                f"seed_time={seed_time:.3f}s",
                request_id,
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
                        "seed_source": seed_source,
                    },
                ):
                    projected_result = self.ai_service.project_payload(
                        session=session,
                        result=seed_result,
                        request_id=request_id,
                    )

            payload_time = time.perf_counter() - payload_start

            if not isinstance(projected_result, dict):
                raise ValueError(
                    "AI service returned invalid payload format. "
                    f"type={type(projected_result).__name__}"
                )

            # --------------------------------------------------
            # 3. Return payload-only response
            # --------------------------------------------------
            total_time = time.perf_counter() - request_start

            response = self._payload_response(
                result=projected_result,
                request_id=request_id,
                client_type=normalized_client_type,
                total_time=total_time,
                seed_time=seed_time,
                payload_time=payload_time,
                seed_source=seed_source,
            )

            info_id(
                f"Chat payload processed in {total_time:.3f}s "
                f"(Seed: {seed_time:.3f}s | Payload: {payload_time:.3f}s | "
                f"seed_source={seed_source} | "
                f"docs={len(response.get('documents') or [])} | "
                f"images={len(response.get('images') or [])} | "
                f"parts={len(response.get('parts') or [])} | "
                f"drawings={len(response.get('drawings') or [])})",
                request_id,
            )

            return response

        except Exception as exc:
            total_time = time.perf_counter() - request_start

            error_id(
                f"ChatPayloadOrchestrator failure after {total_time:.3f}s: {exc}",
                request_id,
                exc_info=True,
            )

            return self._empty_payload_response(
                request_id=request_id,
                status="error",
                payload_status="error",
                message="An unexpected error occurred while loading supporting payload.",
                total_time=total_time,
                seed_time=seed_time,
                payload_time=payload_time,
                seed_source=seed_source,
            )

    def _resolve_payload_seed(
        self,
        *,
        request_id: Optional[str],
        payload_seed: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Resolve the seed payload.

        Priority:
            1. Use payload_seed passed directly by caller.
            2. Load raw response from Q&A storage by request_id.

        Returns:
            (seed_result, seed_source)
        """

        if isinstance(payload_seed, dict) and payload_seed:
            return payload_seed, "request_payload"

        seed_result = self._load_seed_result_from_qanda(
            request_id=request_id,
        )

        if isinstance(seed_result, dict) and seed_result:
            return seed_result, "qanda"

        return {}, "none"

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

        This method also supports raw_response values stored as JSON strings.
        """

        if not request_id:
            warning_id(
                "[ChatPayloadOrchestrator] Cannot load seed from Q&A without request_id.",
                request_id,
            )
            return {}

        with self.transaction() as session:
            if hasattr(self.qanda_service, "get_raw_response_by_request_id"):
                raw_response = self.qanda_service.get_raw_response_by_request_id(
                    session=session,
                    request_id=request_id,
                )

                decoded = self._coerce_raw_response_to_dict(raw_response)

                if decoded:
                    debug_id(
                        "[ChatPayloadOrchestrator] Loaded raw_response by request_id.",
                        request_id,
                    )
                    return decoded

                debug_id(
                    "[ChatPayloadOrchestrator] raw_response lookup returned no usable dict.",
                    request_id,
                )
                return {}

            if hasattr(self.qanda_service, "get_interaction_by_request_id"):
                interaction = self.qanda_service.get_interaction_by_request_id(
                    session=session,
                    request_id=request_id,
                )

                raw_response = getattr(interaction, "raw_response", None)

                decoded = self._coerce_raw_response_to_dict(raw_response)

                if decoded:
                    debug_id(
                        "[ChatPayloadOrchestrator] Loaded interaction.raw_response by request_id.",
                        request_id,
                    )
                    return decoded

                if isinstance(interaction, dict):
                    decoded = self._coerce_raw_response_to_dict(
                        interaction.get("raw_response")
                    )

                    if decoded:
                        debug_id(
                            "[ChatPayloadOrchestrator] Loaded dict raw_response by request_id.",
                            request_id,
                        )
                        return decoded

                debug_id(
                    "[ChatPayloadOrchestrator] interaction lookup returned no usable raw_response.",
                    request_id,
                )
                return {}

        warning_id(
            "[ChatPayloadOrchestrator] QandAService does not provide a seed lookup method. "
            "Add get_raw_response_by_request_id() or pass payload_seed directly.",
            request_id,
        )

        return {}

    @staticmethod
    def _coerce_raw_response_to_dict(raw_response: Any) -> Dict[str, Any]:
        """
        Convert a stored raw_response value into a dictionary.

        Supports:
            - dict
            - JSON string containing an object

        Returns:
            dict if usable, otherwise {}
        """

        if isinstance(raw_response, dict):
            return raw_response

        if isinstance(raw_response, str):
            text = raw_response.strip()

            if not text:
                return {}

            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                return {}

            if isinstance(decoded, dict):
                return decoded

        return {}

    def _payload_response(
        self,
        *,
        result: Dict[str, Any],
        request_id: Optional[str],
        client_type: str,
        total_time: float,
        seed_time: float,
        payload_time: float,
        seed_source: str,
    ) -> Dict[str, Any]:

        documents = self._safe_list(result.get("documents"))
        parts = self._safe_list(result.get("parts"))
        images = self._safe_list(result.get("images"))
        drawings = self._safe_list(result.get("drawings"))

        blocks = self._build_blocks(
            documents=documents,
            parts=parts,
            images=images,
            drawings=drawings,
        )

        used_chunks = self._safe_list(result.get("used_chunks"))
        chunks = self._safe_list(result.get("chunks"))

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
            "used_chunks_count": len(used_chunks),
            "retriever_top_k": result.get("retriever_top_k"),
            "response_time": total_time,
            "performance": {
                "total_time": total_time,
                "seed_time": seed_time,
                "payload_time": payload_time,
                "seed_source": seed_source,
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
                "chunks_count": len(chunks),
                "used_chunks_count": len(used_chunks),
                "seed_source": seed_source,
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
        seed_time: float,
        payload_time: float,
        seed_source: str,
    ) -> Dict[str, Any]:

        return {
            "status": status,
            "request_id": request_id,
            "payload_status": payload_status,
            "message": message,
            "blocks": self._empty_blocks(),
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
                "seed_time": seed_time,
                "payload_time": payload_time,
                "seed_source": seed_source,
                "documents": 0,
                "parts": 0,
                "images": 0,
                "drawings": 0,
            },
        }

    @classmethod
    def _normalize_client_type(cls, client_type: Optional[str]) -> str:
        normalized = (client_type or "web").strip().lower() or "web"

        if normalized not in cls.VALID_CLIENT_TYPES:
            return "web"

        return normalized

    @staticmethod
    def _safe_list(value: Any) -> list:
        if isinstance(value, list):
            return value

        if value is None:
            return []

        return [value]

    @classmethod
    def _empty_blocks(cls) -> Dict[str, list]:
        return {
            key: list(value)
            for key, value in cls.EMPTY_BLOCKS.items()
        }

    @staticmethod
    def _build_blocks(
        *,
        documents: list,
        parts: list,
        images: list,
        drawings: list,
    ) -> Dict[str, list]:

        return {
            "documents-container": documents,
            "parts-container": parts,
            "images-container": images,
            "drawings-container": drawings,
        }