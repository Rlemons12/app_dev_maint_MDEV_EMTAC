from typing import Dict, Any
from sqlalchemy.orm import Session

from modules.services.chunk_search_service import ChunkAssociationSearchService
from modules.emtac_ai.search.rag_core.chunk_relationship_projection import (
    ChunkRelationshipProjection,
)
from modules.services.drawing_part_association_service import (DrawingPartAssociationService)
from modules.services.drawing_navigation_projection import DrawingNavigationProjection
from modules.configuration.log_config import debug_id, get_request_id

class DocumentUIProjectionService:
    """
    Orchestrates UI payload construction for documents starting from a chunk.

    Responsibilities:
    - Execute chunk search
    - Project result into UI-ready payload

    NOTE:
    - Relationship resolution now lives in the search layer
    - UI projection is document-centric
    """

    def __init__(self, session: Session):
        self.session = session
        self.search_service = ChunkAssociationSearchService(session=session)
        self.projector = ChunkRelationshipProjection(session=session)

    def build_from_chunk(
        self,
        chunk_id: int,
        *,
        include_embeddings: bool = False,
        include_reverse: bool = True,
    ) -> Dict[str, Any]:

        search_result = self.search_service.search_from_chunk(
            chunk_id=chunk_id,
            include_embeddings=include_embeddings,
            include_2nd_tier=True,
        )

        documents = search_result.get("documents", [])
        if not documents:
            return {
                "documents-container": [],
                "summary": search_result.get("summary", {}),
            }

        tier1 = search_result.get("1st_tier", {})
        tier2 = search_result.get("2nd_tier", {})

        images = tier1.get("images", {})
        chunk_images = images.get("chunk_level", [])
        document_images = images.get("document_level", [])

        for doc in documents:
            # -----------------------------
            # Images
            # -----------------------------
            doc["images"] = chunk_images + document_images

            # -----------------------------
            # Positions
            # -----------------------------
            positions = tier2.get("positions", [])
            doc["positions"] = positions
            doc["position_ids"] = [
                p["id"] for p in positions
                if isinstance(p, dict) and "id" in p
            ]

            # -----------------------------
            # Drawing Navigation (DOCUMENT → POSITION → DRAWINGS)
            # -----------------------------
            debug_id(
                "[UI PROJECTION] entering drawing_navigation block",
                get_request_id(),
            )
            if doc["position_ids"]:
                drawing_nav = DrawingNavigationProjection(session=self.session)

                doc["drawing_navigation"] = drawing_nav.build_navigation(
                    complete_document_id=doc.get("complete_document_id"),
                    position_ids=doc["position_ids"],
                )

                if not doc.get("drawing_navigation"):
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

                debug_id(
                    f"[UI PROJECTION] drawing_navigation={doc['drawing_navigation']}",
                    get_request_id(),
                )

            # -----------------------------
            # Parts (no drawing enrichment here)
            # -----------------------------
            doc["parts"] = tier2.get("parts", [])

            # -----------------------------
            # Optional panels
            # -----------------------------
            for key in ("tasks", "tools", "problems"):
                if key in tier2:
                    doc[key] = tier2.get(key, [])

        return {
            "documents-container": documents,
            "summary": search_result.get("summary", {}),
        }


