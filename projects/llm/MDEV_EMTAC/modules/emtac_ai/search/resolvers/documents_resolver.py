# modules/emtac_ai/search/resolvers/document_resolver.py

from typing import List, Dict, Any, Optional

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    with_request_id,
    get_request_id,
)

from modules.services.complete_document_service import CompleteDocumentService


class DocumentResolver:
    """
    DOCUMENT RESOLVER

    Purpose:
        Resolve CompleteDocument IDs from a query + document-focused NER output.

    Pipeline position:
        Intent → NER (documents) → THIS CLASS → Expander

    This class:
        - DOES resolve IDs
        - DOES call document services
        - DOES NOT expand associations
        - DOES NOT build context
        - DOES NOT call RAG
    """

    def __init__(self, document_service: Optional[CompleteDocumentService] = None):
        self.document_service = document_service or CompleteDocumentService()

    # --------------------------------------------------
    # PRIMARY RESOLUTION ENTRY POINT
    # --------------------------------------------------
    @with_request_id
    def resolve(
        self,
        query: Optional[str],
        entities: Dict[str, Any],
        limit: int = 10,
        request_id: Optional[str] = None,
    ) -> List[int]:
        """
        Resolve CompleteDocument IDs using:

            1. Explicit document IDs (highest precision)
            2. Entity-driven lookups (title / filename)
            3. Text search (FTS)
            4. Embedding similarity (semantic fallback)

        Returns:
            List[int] → CompleteDocument.id
        """

        request_id = request_id or get_request_id()
        entities = entities or {}

        debug_id(
            f"[DocumentResolver] resolve | query='{query}' | entity_keys={list(entities.keys())}",
            request_id,
        )

        resolved_ids: List[int] = []

        # --------------------------------------------------
        # 0. EXPLICIT DOCUMENT ID (short-circuit)
        # --------------------------------------------------
        explicit_ids = entities.get("DOCUMENT_ID", [])
        if explicit_ids:
            try:
                resolved_ids.extend(
                    int(i) for i in explicit_ids if str(i).isdigit()
                )
                info_id(
                    f"[DocumentResolver] resolved via explicit DOCUMENT_ID ({len(resolved_ids)})",
                    request_id,
                )
                return sorted(set(resolved_ids))
            except Exception as e:
                warning_id(
                    f"[DocumentResolver] DOCUMENT_ID parsing failed: {e}",
                    request_id,
                )

        # --------------------------------------------------
        # 1. ENTITY-DRIVEN LOOKUPS (high precision)
        # --------------------------------------------------
        title_candidates = entities.get("DOCUMENT_TITLE", [])
        file_candidates = entities.get("FILE_NAME", [])

        for title in title_candidates:
            try:
                docs = self.document_service.find(title=title, limit=limit)
                resolved_ids.extend(d.id for d in docs)
            except Exception as e:
                warning_id(
                    f"[DocumentResolver] title lookup failed: {e}",
                    request_id,
                )

        for fname in file_candidates:
            try:
                docs = self.document_service.find(file_path=fname, limit=limit)
                resolved_ids.extend(d.id for d in docs)
            except Exception as e:
                warning_id(
                    f"[DocumentResolver] filename lookup failed: {e}",
                    request_id,
                )

        # --------------------------------------------------
        # 2. TEXT SEARCH (FTS)
        # --------------------------------------------------
        if query:
            try:
                docs = self.document_service.search_text(query, limit=limit)
                resolved_ids.extend(d.id for d in docs)
            except Exception as e:
                warning_id(
                    f"[DocumentResolver] text search failed: {e}",
                    request_id,
                )

        # --------------------------------------------------
        # 3. EMBEDDING SEARCH (semantic fallback)
        # --------------------------------------------------
        if query:
            try:
                docs = self.document_service.search_embedding(
                    query_text=query,
                    limit=limit,
                )
                resolved_ids.extend(d.id for d in docs)
            except Exception as e:
                warning_id(
                    f"[DocumentResolver] embedding search failed: {e}",
                    request_id,
                )

        # --------------------------------------------------
        # 4. DE-DUPLICATE & RETURN
        # --------------------------------------------------
        unique_ids = sorted(set(resolved_ids))

        info_id(
            f"[DocumentResolver] resolved {len(unique_ids)} CompleteDocument IDs",
            request_id,
        )

        return unique_ids

    # --------------------------------------------------
    # DIRECT-ID PASS THROUGH (UI / API)
    # --------------------------------------------------
    def resolve_from_ids(self, ids: List[int]) -> List[int]:
        """
        Validate and normalize externally supplied document IDs.
        """
        return sorted({i for i in ids if isinstance(i, int) and i > 0})
