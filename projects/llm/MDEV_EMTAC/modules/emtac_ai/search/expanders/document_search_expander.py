from typing import Dict, List

from modules.emtac_ai.search.expanders.base import BaseSearchExpander
from modules.emtac_ai.search.expanders.search_expansion_result import SearchExpansionResult

from modules.services.complete_document_service import CompleteDocumentService
from modules.services.image_service import ImageService
from modules.services.position_service import PositionService


class DocumentSearchExpander(BaseSearchExpander):
    """
    Document-centric expander.

    GIVEN a CompletedDocument ID:
      → return EVERYTHING structurally associated with it.
    """

    intent = "Document"

    def __init__(
        self,
        document_service=None,
        image_service=None,
        position_service=None,
    ):
        self.document_service = document_service or CompleteDocumentService()
        self.image_service = image_service or ImageService()
        self.position_service = position_service or PositionService()

    # --------------------------------------------------
    # ENTRY POINT 1: Query → Document IDs → Expand
    # --------------------------------------------------
    def expand(self, query: str, entities: Dict) -> SearchExpansionResult:
        documents = self.document_service.search_text(query, limit=10)

        if not documents:
            return SearchExpansionResult(intent=self.intent)

        return self.expand_from_document_ids([doc.id for doc in documents])

    # --------------------------------------------------
    # ENTRY POINT 2: Document ID(s) → Full Graph Expansion
    # --------------------------------------------------
    def expand_from_document_ids(self, document_ids: List[int]) -> SearchExpansionResult:
        result = SearchExpansionResult(intent=self.intent)

        documents = []
        images = []
        positions = []

        for doc_id in document_ids:
            doc = self.document_service.get(doc_id)
            if not doc:
                continue

            documents.append(doc)

            related = self.document_service.find_related(doc_id)
            if not related:
                continue

            positions.extend(related["downward"].get("positions", []))
            images.extend(related["downward"].get("images", []))

        # --------------------------------------------------
        # Deduplicate
        # --------------------------------------------------
        documents = list({d.id: d for d in documents}.values())
        positions = list({p.id: p for p in positions}.values())
        images = list({i.id: i for i in images}.values())

        # --------------------------------------------------
        # Assemble result
        # --------------------------------------------------
        result.add_primary("documents", documents)
        result.add_context("positions", positions)
        result.add_context("images", images)

        result.metadata = {
            "document_ids": document_ids,
            "document_count": len(documents),
            "position_count": len(positions),
            "image_count": len(images),
        }

        return result
