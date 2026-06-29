from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.configuration.log_config import with_request_id

from modules.services.document_service import DocumentService
from modules.services.complete_document_service import CompleteDocumentService
from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)
from modules.services.document_embedding_service import (
    DocumentEmbeddingService,
)
from modules.services.completed_document_position_service import (
    CompletedDocumentPositionService,
)
from modules.services.parts_position_image_service import (
    PartsPositionImageService,
)
from modules.services.drawing_position_association_service import (
    DrawingPositionAssociationService,
)


class ChunkGraphOrchestrator(BaseOrchestrator):
    """
    Forward traversal:

        Document (Chunk)
            → CompleteDocument
            → Images
            → Embeddings
            → Positions
            → Parts
            → Drawings
    """

    def __init__(self):
        super().__init__()

        self.document_service = DocumentService()
        self.complete_document_service = CompleteDocumentService()
        self.image_completed_document_association_service = (
            ImageCompletedDocumentAssociationService()
        )
        self.document_embedding_service = DocumentEmbeddingService()
        self.completed_document_position_service = (
            CompletedDocumentPositionService()
        )
        self.parts_position_image_service = PartsPositionImageService()
        self.drawing_position_association_service = (
            DrawingPositionAssociationService()
        )

    # ============================================================
    # PRIMARY ENTRY
    # ============================================================

    @with_request_id
    def build_graph(
        self,
        *,
        chunk_id: int,
        include_embeddings: bool = True,
        include_2nd_tier: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        with self._timed("ChunkGraphOrchestrator.build_graph"):
            with self.transaction() as session:

                chunk = self.document_service.get(
                    session=session,
                    doc_id=chunk_id,
                )

                if not chunk:
                    return {"error": "Chunk not found", "chunk_id": chunk_id}

                result = {
                    "chunk_id": chunk.id,
                    "chunk": self._serialize_chunk(chunk),
                    "1st_tier": {},
                    "2nd_tier": {},
                    "summary": {},
                }

                result["1st_tier"] = self._resolve_1st_tier(
                    session=session,
                    chunk=chunk,
                    include_embeddings=include_embeddings,
                )

                if include_2nd_tier:
                    result["2nd_tier"] = self._resolve_2nd_tier(
                        session=session,
                        chunk=chunk,
                    )

                result["summary"] = self._generate_summary(result)

                return result

    # ============================================================
    # TIER 1
    # ============================================================

    def _resolve_1st_tier(
        self,
        *,
        session: Session,
        chunk,
        include_embeddings: bool,
    ) -> Dict[str, Any]:

        result = {
            "complete_document": None,
            "images": {
                "chunk_level": [],
                "document_level": [],
            },
            "embeddings": [],
        }

        if not chunk.complete_document_id:
            return result

        cd_id = chunk.complete_document_id

        # CompleteDocument
        complete_doc = self.complete_document_service.get(
            session=session,
            document_id=cd_id,
        )

        result["complete_document"] = (
            {"id": complete_doc.id, "title": complete_doc.title}
            if complete_doc else None
        )

        # Chunk-level images
        chunk_images = (
            self.image_completed_document_association_service
            .resolve_related_entities(
                session=session,
                document_id=chunk.id,
            )
            .get("images", [])
        )

        result["images"]["chunk_level"] = [
            {
                "id": img.id,
                "file_path": img.file_path,
                "description": img.description,
            }
            for img in chunk_images
        ]

        # Document-level images
        doc_images = (
            self.image_completed_document_association_service
            .resolve_related_entities(
                session=session,
                complete_document_id=cd_id,
            )
            .get("images", [])
        )

        result["images"]["document_level"] = [
            {
                "id": img.id,
                "file_path": img.file_path,
                "description": img.description,
            }
            for img in doc_images
        ]

        # Embeddings
        if include_embeddings:
            embeddings = self.document_embedding_service.get_by_document(
                session=session,
                document_id=chunk.id,
            )

            result["embeddings"] = [
                {
                    "id": e.id,
                    "model_name": e.model_name,
                }
                for e in embeddings
            ]

        return result

    # ============================================================
    # TIER 2
    # ============================================================

    def _resolve_2nd_tier(
            self,
            *,
            session: Session,
            chunk,
    ) -> Dict[str, Any]:

        result = {
            "documents": [],  # preserve graph structure
            "positions": [],
            "parts": [],
            "drawings": [],
        }

        if not chunk.complete_document_id:
            return result

        # -------------------------------------------------
        # Positions
        # -------------------------------------------------
        position_ids = (
            self.completed_document_position_service
            .get_position_ids_for_document(
                session=session,
                complete_document_id=chunk.complete_document_id,
            )
        )

        result["positions"] = position_ids

        if not position_ids:
            return result

        # -------------------------------------------------
        # Parts (via PartsPositionImageAssociation)
        # -------------------------------------------------
        associations = (
            self.parts_position_image_service
            .search_by_positions(
                session=session,
                position_ids=position_ids,
            )
        )

        # Deduplicate Part objects
        parts = list({
            assoc.part
            for assoc in associations
            if assoc.part is not None
        })

        result["parts"] = [
            {
                "id": p.id,
                "part_number": p.part_number,
                "name": p.name,
            }
            for p in parts
        ]

        # -------------------------------------------------
        # Drawings
        # -------------------------------------------------
        drawings = (
            self.drawing_position_association_service
            .get_drawings_for_positions(
                session=session,
                position_ids=position_ids,
            )
        )

        result["drawings"] = [
            {
                "id": d.id,
                "drw_number": d.drw_number,
                "drw_name": d.drw_name,
            }
            for d in drawings
        ]

        return result

    # ============================================================
    # SUMMARY
    # ============================================================

    def _generate_summary(self, result: Dict[str, Any]) -> Dict[str, int]:
        return {
            "image_count": (
                len(result["1st_tier"]["images"]["chunk_level"])
                + len(result["1st_tier"]["images"]["document_level"])
            ),
            "embedding_count": len(result["1st_tier"]["embeddings"]),
            "position_count": len(result["2nd_tier"].get("positions", [])),
            "part_count": len(result["2nd_tier"].get("parts", [])),
            "drawing_count": len(result["2nd_tier"].get("drawings", [])),
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def _serialize_chunk(self, chunk):
        return {
            "id": chunk.id,
            "name": chunk.name,
            "content": chunk.content,
            "complete_document_id": chunk.complete_document_id,
        }