from __future__ import annotations

from typing import Dict, Any

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    debug_id,
    get_request_id,
)
from modules.orchestrators.chunk_graph_orchestrator import ChunkGraphOrchestrator
from modules.services.document_service import DocumentService
from modules.ai.search_pathway.rag_core.document_ui_payload import DocumentUIPayload
from modules.services.drawing_navigation_projection import DrawingNavigationProjection


class DocumentUIProjectionService:
    """
    Orchestrates UI payload construction for documents starting from a chunk.

    Responsibilities:
    - Resolve the chunk
    - Build forward graph relationships
    - Project those relationships into a UI-ready payload

    NOTE:
    - This version no longer depends on ChunkAssociationSearchService
    - It reuses ChunkGraphOrchestrator for graph resolution
    """

    def __init__(self, session: Session):
        self.session = session
        self.document_service = DocumentService()
        self.chunk_graph_orchestrator = ChunkGraphOrchestrator()

    def build_from_chunk(
        self,
        chunk_id: int,
        *,
        include_embeddings: bool = False,
        include_reverse: bool = True,  # kept for signature stability
    ) -> Dict[str, Any]:

        rid = get_request_id()

        # --------------------------------------------------
        # 1. Resolve chunk
        # --------------------------------------------------
        chunk = self.document_service.get(
            session=self.session,
            doc_id=chunk_id,
            request_id=rid,
        )

        if not chunk:
            return {
                "documents-container": [],
                "summary": {},
            }

        # --------------------------------------------------
        # 2. Build graph from chunk
        # --------------------------------------------------
        graph = self.chunk_graph_orchestrator.build_graph(
            chunk_id=chunk_id,
            include_embeddings=include_embeddings,
            include_2nd_tier=True,
            request_id=rid,
        )

        if not isinstance(graph, dict) or graph.get("error"):
            return {
                "documents-container": [],
                "summary": graph.get("summary", {}) if isinstance(graph, dict) else {},
            }

        first = graph.get("1st_tier", {}) or {}
        second = graph.get("2nd_tier", {}) or {}
        summary = graph.get("summary", {}) or {}

        complete_doc = first.get("complete_document") or {}

        # --------------------------------------------------
        # 3. Build base document payload from chunk
        # --------------------------------------------------
        base_chunks = [
            {
                "chunk_id": chunk.id,
                "document_id": chunk.id,
                "content": chunk.content or "",
                "file_path": getattr(chunk, "file_path", None),
                "complete_document_id": chunk.complete_document_id,
                "complete_document_title": complete_doc.get("title"),
            }
        ]

        payload = DocumentUIPayload()
        payload.aggregate_from_chunks(base_chunks, request_id=rid)

        # --------------------------------------------------
        # 4. Add enrichment to each projected document
        # --------------------------------------------------
        chunk_level_images = (
            first.get("images", {}) or {}
        ).get("chunk_level", []) or []

        document_level_images = (
            first.get("images", {}) or {}
        ).get("document_level", []) or []

        all_images = chunk_level_images + document_level_images

        raw_position_ids = second.get("positions", []) or []
        position_ids = [
            int(pid) for pid in raw_position_ids
            if isinstance(pid, int) or (isinstance(pid, str) and pid.isdigit())
        ]

        serialized_positions = [{"id": pid} for pid in position_ids]

        parts = second.get("parts", []) or []
        drawings = second.get("drawings", []) or []

        for doc in payload._documents.values():
            # -----------------------------
            # Images
            # -----------------------------
            doc["images"] = all_images

            # -----------------------------
            # Positions
            # -----------------------------
            doc["positions"] = serialized_positions
            doc["position_ids"] = position_ids

            # -----------------------------
            # Drawing Navigation
            # -----------------------------
            debug_id(
                "[UI PROJECTION] entering drawing_navigation block",
                rid,
            )

            if doc["position_ids"]:
                drawing_nav = DrawingNavigationProjection(session=self.session)

                nav = drawing_nav.build_navigation(
                    complete_document_id=doc.get("complete_document_id"),
                    position_ids=doc["position_ids"],
                )

                if not nav:
                    nav = {
                        "complete_document_id": doc.get("complete_document_id"),
                        "areas": [],
                        "meta": {
                            "area_count": 0,
                            "model_count": 0,
                            "asset_count": 0,
                            "drawing_count": 0,
                        },
                    }

                doc["drawing_navigation"] = nav

                debug_id(
                    f"[UI PROJECTION] drawing_navigation={doc['drawing_navigation']}",
                    rid,
                )
            else:
                doc["drawing_navigation"] = {
                    "complete_document_id": doc.get("complete_document_id"),
                    "areas": [],
                    "meta": {
                        "area_count": 0,
                        "model_count": 0,
                        "asset_count": 0,
                        "drawing_count": 0,
                    },
                }

            # -----------------------------
            # Parts
            # -----------------------------
            doc["parts"] = parts

            # -----------------------------
            # Optional panels
            # -----------------------------
            if drawings:
                doc["drawings"] = drawings

            for key in ("tasks", "tools", "problems"):
                if key in second:
                    doc[key] = second.get(key, [])

        return {
            "documents-container": payload.build(),
            "summary": summary,
        }