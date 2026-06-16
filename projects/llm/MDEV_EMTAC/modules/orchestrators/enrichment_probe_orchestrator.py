from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.orchestrators.chunk_graph_orchestrator import ChunkGraphOrchestrator
from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    error_id,
)
from modules.emtacdb.emtacdb_fts import Document
from modules.services.document_ui_projection_service import DocumentUIProjectionService


class EnrichmentProbeOrchestrator(BaseOrchestrator):
    """
    Orchestrates enrichment probing for document chunks.

    Responsibilities:
        - Find a trigger chunk that produces enrichment
        - Build a UI payload from that chunk
        - Return structured diagnostics
        - Scan tier-level enrichment counts across chunks

    Does NOT:
        - Print to console
        - Own CLI/bootstrap behavior
        - Embed bootstrap or sys.path logic
    """

    def __init__(self):
        super().__init__()
        self.chunk_graph_orchestrator = ChunkGraphOrchestrator()

    # ----------------------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------------------

    @staticmethod
    def _safe_rollback(session: Session, request_id: Optional[str] = None) -> None:
        """
        Safely rollback a poisoned or failed SQLAlchemy transaction.
        """
        try:
            session.rollback()
            debug_id("[ENRICHMENT_PROBE] Session rollback completed", request_id)
        except Exception as rollback_error:
            error_id(
                f"[ENRICHMENT_PROBE] Session rollback failed: {rollback_error}",
                request_id,
                exc_info=True,
            )

    @staticmethod
    def _chunk_preview(chunk: Document, preview_len: int = 300) -> Dict[str, Any]:
        """
        Return a compact preview payload for the chunk.
        """
        content = (getattr(chunk, "content", None) or "").strip()

        return {
            "chunk_id": getattr(chunk, "id", None),
            "chunk_name": getattr(chunk, "name", None),
            "complete_document_id": getattr(chunk, "complete_document_id", None),
            "preview": content[:preview_len],
            "content_length": len(content),
        }

    @staticmethod
    def _ui_payload_summary(ui_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a compact summary of the UI payload structure.
        """
        documents = ui_payload.get("documents-container", []) or []

        doc_summaries: List[Dict[str, Any]] = []

        for doc in documents:
            doc_summaries.append(
                {
                    "complete_document_id": doc.get("complete_document_id"),
                    "chunk_count": len(doc.get("chunks", []) or []),
                    "image_count": len(doc.get("images", []) or []),
                    "position_count": len(doc.get("positions", []) or []),
                    "part_count": len(doc.get("parts", []) or []),
                    "has_drawing_navigation": bool(doc.get("drawing_navigation")),
                }
            )

        return {
            "document_count": len(documents),
            "documents": doc_summaries,
        }

    @staticmethod
    def _graph_summary(graph_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize graph summary into a stable shape for probe responses.
        """
        summary = graph_result.get("summary", {}) or {}

        image_count = int(summary.get("image_count", 0) or 0)
        embedding_count = int(summary.get("embedding_count", 0) or 0)
        position_count = int(summary.get("position_count", 0) or 0)
        part_count = int(summary.get("part_count", 0) or 0)
        drawing_count = int(summary.get("drawing_count", 0) or 0)

        return {
            "image_count": image_count,
            "embedding_count": embedding_count,
            "position_count": position_count,
            "part_count": part_count,
            "drawing_count": drawing_count,
            "has_trigger_chunk": (
                image_count > 0 or part_count > 0 or drawing_count > 0
            ),
        }

    @staticmethod
    def _is_trigger_graph(graph_result: Dict[str, Any]) -> bool:
        """
        A trigger chunk is one that yields UI-relevant enrichment.
        """
        summary = graph_result.get("summary", {}) or {}

        image_count = int(summary.get("image_count", 0) or 0)
        part_count = int(summary.get("part_count", 0) or 0)
        drawing_count = int(summary.get("drawing_count", 0) or 0)

        return image_count > 0 or part_count > 0 or drawing_count > 0

    # ----------------------------------------------------------------------
    # CORE SEARCH
    # ----------------------------------------------------------------------

    @with_request_id
    def find_trigger_chunk(
        self,
        *,
        session: Session,
        request_id: Optional[str] = None,
    ) -> Optional[Document]:
        """
        Returns ONE Document chunk that can trigger UI enrichment.

        Trigger conditions:
        - image_count > 0
        - part_count > 0
        - drawing_count > 0
        """
        info_id("[ENRICHMENT_PROBE] Starting trigger chunk search", request_id)

        processed = 0
        trigger_count = 0

        query = session.query(Document).yield_per(50)

        for chunk in query:
            processed += 1

            try:
                graph_result = self.chunk_graph_orchestrator.build_graph(
                    chunk_id=chunk.id,
                    include_embeddings=False,
                    include_2nd_tier=True,
                    request_id=request_id,
                )
            except Exception as exc:
                error_id(
                    f"[ENRICHMENT_PROBE] build_graph failed for chunk_id={getattr(chunk, 'id', None)}: {exc}",
                    request_id,
                    exc_info=True,
                )
                self._safe_rollback(session, request_id)
                continue

            if not isinstance(graph_result, dict):
                debug_id(
                    f"[ENRICHMENT_PROBE] Skipping chunk_id={chunk.id}: graph_result was not a dict",
                    request_id,
                )
                self._safe_rollback(session, request_id)
                continue

            if graph_result.get("error"):
                debug_id(
                    f"[ENRICHMENT_PROBE] Skipping chunk_id={chunk.id}: graph_result contained error={graph_result.get('error')}",
                    request_id,
                )
                self._safe_rollback(session, request_id)
                continue

            summary = graph_result.get("summary", {}) or {}
            image_count = int(summary.get("image_count", 0) or 0)
            part_count = int(summary.get("part_count", 0) or 0)
            drawing_count = int(summary.get("drawing_count", 0) or 0)

            if self._is_trigger_graph(graph_result):
                trigger_count += 1

                debug_id(
                    (
                        f"[ENRICHMENT_PROBE] Trigger chunk found: "
                        f"chunk_id={chunk.id} "
                        f"images={image_count} "
                        f"parts={part_count} "
                        f"drawings={drawing_count}"
                    ),
                    request_id,
                )

                return chunk

            if processed % 100 == 0:
                debug_id(
                    f"[ENRICHMENT_PROBE] Processed {processed} chunks so far; triggers_found={trigger_count}",
                    request_id,
                )

        info_id(
            f"[ENRICHMENT_PROBE] Completed search. Processed={processed}, triggers_found={trigger_count}",
            request_id,
        )
        return None

    # ----------------------------------------------------------------------
    # TIER SUMMARY SCAN
    # ----------------------------------------------------------------------

    @with_request_id
    def scan_tier_summary(
        self,
        *,
        request_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Scan chunks and return tier-level enrichment counts.

        Metrics:
            - chunk-level images
            - document-level images
            - any images
            - positions
            - parts
            - drawings
            - common combinations
            - sample chunk IDs
        """
        info_id("[ENRICHMENT_PROBE] Starting tier summary scan", request_id)

        stats = {
            "total_chunks_scanned": 0,
            "chunks_with_chunk_level_images": 0,
            "chunks_with_document_level_images": 0,
            "chunks_with_any_images": 0,
            "chunks_with_positions": 0,
            "chunks_with_parts": 0,
            "chunks_with_drawings": 0,
            "chunks_with_images_and_parts": 0,
            "chunks_with_images_and_drawings": 0,
            "chunks_with_parts_and_drawings": 0,
            "chunks_with_full_payload": 0,
            "sample_chunk_ids": {
                "chunk_level_images": [],
                "document_level_images": [],
                "parts": [],
                "drawings": [],
                "full_payload": [],
            },
        }

        def _maybe_add_sample(bucket: str, chunk_id: int, max_items: int = 10) -> None:
            items = stats["sample_chunk_ids"][bucket]
            if chunk_id not in items and len(items) < max_items:
                items.append(chunk_id)

        try:
            with self.transaction() as session:
                query = session.query(Document).order_by(Document.id.asc())

                if limit is not None:
                    query = query.limit(int(limit))

                for chunk in query.yield_per(50):
                    stats["total_chunks_scanned"] += 1

                    try:
                        graph_result = self.chunk_graph_orchestrator.build_graph(
                            chunk_id=chunk.id,
                            include_embeddings=False,
                            include_2nd_tier=True,
                            request_id=request_id,
                        )
                    except Exception as exc:
                        error_id(
                            f"[ENRICHMENT_PROBE] build_graph failed for chunk_id={chunk.id}: {exc}",
                            request_id,
                            exc_info=True,
                        )
                        self._safe_rollback(session, request_id)
                        continue

                    if not isinstance(graph_result, dict) or graph_result.get("error"):
                        continue

                    first = graph_result.get("1st_tier", {}) or {}
                    images = first.get("images", {}) or {}
                    summary = graph_result.get("summary", {}) or {}

                    chunk_level_images = images.get("chunk_level", []) or []
                    document_level_images = images.get("document_level", []) or []

                    has_chunk_level_images = bool(chunk_level_images)
                    has_document_level_images = bool(document_level_images)
                    has_any_images = has_chunk_level_images or has_document_level_images

                    position_count = int(summary.get("position_count", 0) or 0)
                    part_count = int(summary.get("part_count", 0) or 0)
                    drawing_count = int(summary.get("drawing_count", 0) or 0)

                    has_positions = position_count > 0
                    has_parts = part_count > 0
                    has_drawings = drawing_count > 0
                    has_full_payload = has_any_images and has_parts and has_drawings

                    if has_chunk_level_images:
                        stats["chunks_with_chunk_level_images"] += 1
                        _maybe_add_sample("chunk_level_images", chunk.id)

                    if has_document_level_images:
                        stats["chunks_with_document_level_images"] += 1
                        _maybe_add_sample("document_level_images", chunk.id)

                    if has_any_images:
                        stats["chunks_with_any_images"] += 1

                    if has_positions:
                        stats["chunks_with_positions"] += 1

                    if has_parts:
                        stats["chunks_with_parts"] += 1
                        _maybe_add_sample("parts", chunk.id)

                    if has_drawings:
                        stats["chunks_with_drawings"] += 1
                        _maybe_add_sample("drawings", chunk.id)

                    if has_any_images and has_parts:
                        stats["chunks_with_images_and_parts"] += 1

                    if has_any_images and has_drawings:
                        stats["chunks_with_images_and_drawings"] += 1

                    if has_parts and has_drawings:
                        stats["chunks_with_parts_and_drawings"] += 1

                    if has_full_payload:
                        stats["chunks_with_full_payload"] += 1
                        _maybe_add_sample("full_payload", chunk.id)

                    if stats["total_chunks_scanned"] % 100 == 0:
                        debug_id(
                            (
                                f"[ENRICHMENT_PROBE] Tier scan progress: "
                                f"scanned={stats['total_chunks_scanned']} "
                                f"images={stats['chunks_with_any_images']} "
                                f"parts={stats['chunks_with_parts']} "
                                f"drawings={stats['chunks_with_drawings']} "
                                f"full={stats['chunks_with_full_payload']}"
                            ),
                            request_id,
                        )

            info_id(
                (
                    f"[ENRICHMENT_PROBE] Tier summary completed: "
                    f"scanned={stats['total_chunks_scanned']} "
                    f"images={stats['chunks_with_any_images']} "
                    f"parts={stats['chunks_with_parts']} "
                    f"drawings={stats['chunks_with_drawings']} "
                    f"full={stats['chunks_with_full_payload']}"
                ),
                request_id,
            )

            return {
                "status": "success",
                "message": "Tier summary scan completed.",
                "tier_summary": stats,
            }

        except Exception as exc:
            error_id(
                f"[ENRICHMENT_PROBE] Tier summary failed: {exc}",
                request_id,
                exc_info=True,
            )
            raise

    # ----------------------------------------------------------------------
    # MAIN PROBE
    # ----------------------------------------------------------------------

    @with_request_id
    def run_probe(
        self,
        *,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the enrichment probe workflow and return structured results.
        """
        info_id("[ENRICHMENT_PROBE] Probe started", request_id)

        try:
            with self.transaction() as session:
                chunk = self.find_trigger_chunk(
                    session=session,
                    request_id=request_id,
                )

                if not chunk:
                    info_id("[ENRICHMENT_PROBE] No trigger chunk found", request_id)
                    return {
                        "status": "not_found",
                        "message": "No trigger chunk found.",
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

                chunk_preview = self._chunk_preview(chunk)

                try:
                    graph_result = self.chunk_graph_orchestrator.build_graph(
                        chunk_id=chunk.id,
                        include_embeddings=False,
                        include_2nd_tier=True,
                        request_id=request_id,
                    )
                    graph_summary = self._graph_summary(graph_result)
                except Exception as exc:
                    error_id(
                        f"[ENRICHMENT_PROBE] build_graph failed for selected chunk_id={chunk.id}: {exc}",
                        request_id,
                        exc_info=True,
                    )
                    self._safe_rollback(session, request_id)
                    graph_summary = {
                        "image_count": 0,
                        "embedding_count": 0,
                        "position_count": 0,
                        "part_count": 0,
                        "drawing_count": 0,
                        "has_trigger_chunk": True,
                    }

                try:
                    ui_service = DocumentUIProjectionService(session=session)
                    ui_payload = ui_service.build_from_chunk(chunk.id)
                except Exception as exc:
                    error_id(
                        f"[ENRICHMENT_PROBE] build_from_chunk failed for chunk_id={chunk.id}: {exc}",
                        request_id,
                        exc_info=True,
                    )
                    self._safe_rollback(session, request_id)

                    return {
                        "status": "error",
                        "message": "Failed to build UI payload.",
                        "chunk": chunk_preview,
                        "graph_summary": graph_summary,
                        "ui_payload": {
                            "documents-container": [],
                            "summary": {},
                        },
                        "ui_payload_summary": {
                            "document_count": 0,
                            "documents": [],
                        },
                    }

                if not isinstance(ui_payload, dict):
                    error_id(
                        f"[ENRICHMENT_PROBE] Unexpected ui_payload type: {type(ui_payload).__name__}",
                        request_id,
                    )
                    return {
                        "status": "error",
                        "message": "UI payload was not a dictionary.",
                        "chunk": chunk_preview,
                        "graph_summary": graph_summary,
                        "ui_payload": {
                            "documents-container": [],
                            "summary": {},
                        },
                        "ui_payload_summary": {
                            "document_count": 0,
                            "documents": [],
                        },
                    }

                ui_payload_summary = self._ui_payload_summary(ui_payload)

                info_id("[ENRICHMENT_PROBE] Probe completed successfully", request_id)

                return {
                    "status": "success",
                    "message": "UI payload built successfully.",
                    "chunk": chunk_preview,
                    "graph_summary": graph_summary,
                    "ui_payload": ui_payload,
                    "ui_payload_summary": ui_payload_summary,
                }

        except Exception as exc:
            error_id(
                f"[ENRICHMENT_PROBE] Fatal error: {exc}",
                request_id,
                exc_info=True,
            )
            raise

    # ----------------------------------------------------------------------
    # PROBE SPECIFIC CHUNK
    # ----------------------------------------------------------------------

    @with_request_id
    def probe_chunk(
        self,
        *,
        chunk_id: int,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Probe a specific chunk ID and return the projected UI payload.
        """
        info_id(f"[ENRICHMENT_PROBE] Probing specific chunk_id={chunk_id}", request_id)

        try:
            with self.transaction() as session:
                chunk = session.get(Document, chunk_id)

                if not chunk:
                    return {
                        "status": "not_found",
                        "message": f"Chunk {chunk_id} not found.",
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

                chunk_preview = self._chunk_preview(chunk)

                try:
                    graph_result = self.chunk_graph_orchestrator.build_graph(
                        chunk_id=chunk_id,
                        include_embeddings=False,
                        include_2nd_tier=True,
                        request_id=request_id,
                    )
                    graph_summary = self._graph_summary(graph_result)
                except Exception as exc:
                    error_id(
                        f"[ENRICHMENT_PROBE] build_graph failed for chunk_id={chunk_id}: {exc}",
                        request_id,
                        exc_info=True,
                    )
                    self._safe_rollback(session, request_id)
                    graph_summary = {
                        "image_count": 0,
                        "embedding_count": 0,
                        "position_count": 0,
                        "part_count": 0,
                        "drawing_count": 0,
                        "has_trigger_chunk": False,
                    }

                ui_service = DocumentUIProjectionService(session=session)
                ui_payload = ui_service.build_from_chunk(chunk_id)

                if not isinstance(ui_payload, dict):
                    return {
                        "status": "error",
                        "message": "UI payload was not a dictionary.",
                        "chunk": chunk_preview,
                        "graph_summary": graph_summary,
                        "ui_payload": {
                            "documents-container": [],
                            "summary": {},
                        },
                        "ui_payload_summary": {
                            "document_count": 0,
                            "documents": [],
                        },
                    }

                return {
                    "status": "success",
                    "message": "Chunk probed successfully.",
                    "chunk": chunk_preview,
                    "graph_summary": graph_summary,
                    "ui_payload": ui_payload,
                    "ui_payload_summary": self._ui_payload_summary(ui_payload),
                }

        except Exception as exc:
            error_id(
                f"[ENRICHMENT_PROBE] probe_chunk failed for chunk_id={chunk_id}: {exc}",
                request_id,
                exc_info=True,
            )
            raise