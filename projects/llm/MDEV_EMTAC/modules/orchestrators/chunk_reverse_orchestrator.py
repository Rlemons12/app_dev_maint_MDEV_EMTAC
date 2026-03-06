from typing import Dict, Any, List, Optional

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.configuration.log_config import with_request_id

# Explicit service imports
from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)
from modules.services.parts_position_image_service import (
    PartsPositionImageService,
)
from modules.services.completed_document_position_service import (
    CompletedDocumentPositionService,
)
from modules.services.document_service import DocumentService


class ChunkReverseOrchestrator(BaseOrchestrator):

    # =========================================================
    # INIT – Explicit Service Injection
    # =========================================================
    def __init__(self):
        super().__init__()

        self.image_completed_document_association_service = (
            ImageCompletedDocumentAssociationService()
        )
        self.parts_position_image_service = PartsPositionImageService()
        self.completed_document_position_service = (
            CompletedDocumentPositionService()
        )
        self.document_service = DocumentService()

    # =========================================================
    # CHUNKS BY IMAGE
    # =========================================================
    @with_request_id
    def find_chunks_by_image(
        self,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        with self.transaction() as session:

            resolved = (
                self.image_completed_document_association_service
                .resolve_related_entities(
                    session=session,
                    image_id=image_id,
                )
            )

            documents = resolved.get("documents", [])

            return [
                {
                    "chunk_id": d.id,
                    "content_preview": d.content[:200] if d.content else None,
                    "complete_document_id": d.complete_document_id,
                }
                for d in documents
            ]

    # =========================================================
    # CHUNKS BY PART
    # =========================================================
    @with_request_id
    def find_chunks_by_part(
        self,
        *,
        part_id: int,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        with self.transaction() as session:

            position_ids = (
                self.parts_position_image_service
                .get_position_ids_for_part(
                    session=session,
                    part_id=part_id,
                )
            )

            complete_doc_ids = (
                self.completed_document_position_service
                .get_complete_document_ids_for_positions(
                    session=session,
                    position_ids=position_ids,
                )
            )

            chunks = (
                self.document_service
                .get_by_complete_document_ids(
                    session=session,
                    complete_document_ids=complete_doc_ids,
                )
            )

            return [
                {
                    "chunk_id": c.id,
                    "content_preview": c.content[:200] if c.content else None,
                    "complete_document_id": c.complete_document_id,
                }
                for c in chunks
            ]