from __future__ import annotations

from typing import Dict, Any, Optional

from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    error_id,
)
from modules.orchestrators.enrichment_probe_orchestrator import (
    EnrichmentProbeOrchestrator,
)


class EnrichmentProbeCoordinator:
    """
    Application-layer coordinator for enrichment probe workflows.

    Responsibilities:
        - normalize probe inputs
        - delegate to enrichment probe orchestrator
        - normalize response contract
        - provide stable application-facing entrypoints

    Does NOT:
        - open sessions
        - commit / rollback
        - query ORM directly
        - perform domain logic directly
    """

    def __init__(self):
        self.orchestrator = EnrichmentProbeOrchestrator()

    # ------------------------------------------------------------------
    # MAIN PROBE ENTRY
    # ------------------------------------------------------------------

    @with_request_id
    def run_probe(
        self,
        *,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        debug_id("[EnrichmentProbeCoordinator] Starting run_probe", request_id)

        try:
            result = self.orchestrator.run_probe(
                request_id=request_id,
            )

            normalized = self._normalize_probe_response(result)

            info_id(
                "[EnrichmentProbeCoordinator] Completed run_probe",
                request_id,
            )
            return normalized

        except Exception as e:
            error_id(
                f"[EnrichmentProbeCoordinator] run_probe failed: {e}",
                request_id,
                exc_info=True,
            )
            return self._normalize_probe_response(
                {
                    "status": "error",
                    "message": "Enrichment probe failed.",
                    "chunk": None,
                    "graph_summary": {
                        "image_count": 0,
                        "embedding_count": 0,
                        "position_count": 0,
                        "part_count": 0,
                        "drawing_count": 0,
                        "has_trigger_chunk": False,
                    },
                    "ui_payload": {
                        "documents-container": [],
                        "summary": {},
                    },
                    "ui_payload_summary": {
                        "document_count": 0,
                        "documents": [],
                    },
                }
            )

    # ------------------------------------------------------------------
    # SPECIFIC CHUNK PROBE ENTRY
    # ------------------------------------------------------------------

    @with_request_id
    def probe_chunk(
        self,
        *,
        chunk_id: int,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        debug_id(
            f"[EnrichmentProbeCoordinator] Starting probe_chunk chunk_id={chunk_id}",
            request_id,
        )

        try:
            if not isinstance(chunk_id, int):
                try:
                    chunk_id = int(chunk_id)
                except (TypeError, ValueError):
                    return self._normalize_probe_response(
                        {
                            "status": "invalid_input",
                            "message": "chunk_id must be an integer.",
                            "chunk": None,
                            "graph_summary": {
                                "image_count": 0,
                                "embedding_count": 0,
                                "position_count": 0,
                                "part_count": 0,
                                "drawing_count": 0,
                                "has_trigger_chunk": False,
                            },
                            "ui_payload": {
                                "documents-container": [],
                                "summary": {},
                            },
                            "ui_payload_summary": {
                                "document_count": 0,
                                "documents": [],
                            },
                        }
                    )

            if chunk_id <= 0:
                return self._normalize_probe_response(
                    {
                        "status": "invalid_input",
                        "message": "chunk_id must be greater than 0.",
                        "chunk": None,
                        "graph_summary": {
                            "image_count": 0,
                            "embedding_count": 0,
                            "position_count": 0,
                            "part_count": 0,
                            "drawing_count": 0,
                            "has_trigger_chunk": False,
                        },
                        "ui_payload": {
                            "documents-container": [],
                            "summary": {},
                        },
                        "ui_payload_summary": {
                            "document_count": 0,
                            "documents": [],
                        },
                    }
                )

            result = self.orchestrator.probe_chunk(
                chunk_id=chunk_id,
                request_id=request_id,
            )

            normalized = self._normalize_probe_response(result)

            info_id(
                f"[EnrichmentProbeCoordinator] Completed probe_chunk chunk_id={chunk_id}",
                request_id,
            )
            return normalized

        except Exception as e:
            error_id(
                f"[EnrichmentProbeCoordinator] probe_chunk failed: {e}",
                request_id,
                exc_info=True,
            )
            return self._normalize_probe_response(
                {
                    "status": "error",
                    "message": "Chunk probe failed.",
                    "chunk": None,
                    "graph_summary": {
                        "image_count": 0,
                        "embedding_count": 0,
                        "position_count": 0,
                        "part_count": 0,
                        "drawing_count": 0,
                        "has_trigger_chunk": False,
                    },
                    "ui_payload": {
                        "documents-container": [],
                        "summary": {},
                    },
                    "ui_payload_summary": {
                        "document_count": 0,
                        "documents": [],
                    },
                }
            )

    # ------------------------------------------------------------------
    # TIER SUMMARY ENTRY
    # ------------------------------------------------------------------

    @with_request_id
    def scan_tier_summary(
        self,
        *,
        request_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        debug_id("[EnrichmentProbeCoordinator] Starting scan_tier_summary", request_id)

        try:
            result = self.orchestrator.scan_tier_summary(
                request_id=request_id,
                limit=limit,
            )

            normalized = self._normalize_tier_summary_response(result)

            info_id(
                "[EnrichmentProbeCoordinator] Completed scan_tier_summary",
                request_id,
            )
            return normalized

        except Exception as e:
            error_id(
                f"[EnrichmentProbeCoordinator] scan_tier_summary failed: {e}",
                request_id,
                exc_info=True,
            )
            return self._normalize_tier_summary_response(
                {
                    "status": "error",
                    "message": "Tier summary scan failed.",
                    "tier_summary": {},
                }
            )

    # ------------------------------------------------------------------
    # NORMALIZATION - PROBE RESPONSES
    # ------------------------------------------------------------------

    def _normalize_probe_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(response, dict):
            response = {
                "status": "error",
                "message": "Invalid enrichment probe response format.",
                "chunk": None,
                "graph_summary": {
                    "image_count": 0,
                    "embedding_count": 0,
                    "position_count": 0,
                    "part_count": 0,
                    "drawing_count": 0,
                    "has_trigger_chunk": False,
                },
                "ui_payload": {
                    "documents-container": [],
                    "summary": {},
                },
                "ui_payload_summary": {
                    "document_count": 0,
                    "documents": [],
                },
            }

        response.setdefault("status", "success")
        response.setdefault("message", "")
        response.setdefault("chunk", None)

        if "graph_summary" not in response or not isinstance(response["graph_summary"], dict):
            response["graph_summary"] = {
                "image_count": 0,
                "embedding_count": 0,
                "position_count": 0,
                "part_count": 0,
                "drawing_count": 0,
                "has_trigger_chunk": False,
            }

        response["graph_summary"].setdefault("image_count", 0)
        response["graph_summary"].setdefault("embedding_count", 0)
        response["graph_summary"].setdefault("position_count", 0)
        response["graph_summary"].setdefault("part_count", 0)
        response["graph_summary"].setdefault("drawing_count", 0)
        response["graph_summary"].setdefault("has_trigger_chunk", False)

        if "ui_payload" not in response or not isinstance(response["ui_payload"], dict):
            response["ui_payload"] = {
                "documents-container": [],
                "summary": {},
            }

        response["ui_payload"].setdefault("documents-container", [])
        response["ui_payload"].setdefault("summary", {})

        if (
            "ui_payload_summary" not in response
            or not isinstance(response["ui_payload_summary"], dict)
        ):
            response["ui_payload_summary"] = {
                "document_count": 0,
                "documents": [],
            }

        response["ui_payload_summary"].setdefault("document_count", 0)
        response["ui_payload_summary"].setdefault("documents", [])

        return response

    # ------------------------------------------------------------------
    # NORMALIZATION - TIER SUMMARY
    # ------------------------------------------------------------------

    def _normalize_tier_summary_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(response, dict):
            response = {
                "status": "error",
                "message": "Invalid tier summary response format.",
                "tier_summary": {},
            }

        response.setdefault("status", "success")
        response.setdefault("message", "")

        if "tier_summary" not in response or not isinstance(response["tier_summary"], dict):
            response["tier_summary"] = {}

        ts = response["tier_summary"]

        ts.setdefault("total_chunks_scanned", 0)
        ts.setdefault("chunks_with_chunk_level_images", 0)
        ts.setdefault("chunks_with_document_level_images", 0)
        ts.setdefault("chunks_with_any_images", 0)
        ts.setdefault("chunks_with_positions", 0)
        ts.setdefault("chunks_with_parts", 0)
        ts.setdefault("chunks_with_drawings", 0)
        ts.setdefault("chunks_with_images_and_parts", 0)
        ts.setdefault("chunks_with_images_and_drawings", 0)
        ts.setdefault("chunks_with_parts_and_drawings", 0)
        ts.setdefault("chunks_with_full_payload", 0)
        ts.setdefault(
            "sample_chunk_ids",
            {
                "chunk_level_images": [],
                "document_level_images": [],
                "parts": [],
                "drawings": [],
                "full_payload": [],
            },
        )

        ts["sample_chunk_ids"].setdefault("chunk_level_images", [])
        ts["sample_chunk_ids"].setdefault("document_level_images", [])
        ts["sample_chunk_ids"].setdefault("parts", [])
        ts["sample_chunk_ids"].setdefault("drawings", [])
        ts["sample_chunk_ids"].setdefault("full_payload", [])

        return response