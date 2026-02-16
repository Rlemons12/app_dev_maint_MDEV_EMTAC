"""
Documents intent router (new BaseRouter version)

Routing priority:
    1. doc_id (direct lookup)
    2. title
    3. file_name
    4. position → documents
    5. FTS on CompleteDocument
    6. FTS on Document chunks
    7. fallback → no_results
"""

from typing import Any, Dict, List

from modules.services import DBServices
from modules.emtac_ai.intent_ner.routers.base_router import BaseRouter

DB = DBServices()


# ----------------------------------------------------------------------
# MAIN ROUTER (NEW BASE ROUTER VERSION)
# ----------------------------------------------------------------------
def documents_router(*, text: str, intent: str, confidence: float, entities: Dict[str, Any]) -> Dict[str, Any]:

    router = BaseRouter(
        service=DB.complete_documents,
        serializer=_serialize_complete_document
    )

    # --------------------------------------------
    # PRIORITY HANDLER FUNCTIONS
    # --------------------------------------------

    def by_doc_id(entities):
        doc_id = entities.get("doc_id")
        if not doc_id:
            return None
        doc = DB.complete_documents.get(doc_id)
        if doc:
            return ([doc], "doc_id")        # tests expect EXACT string
        return None

    def by_title(entities):
        title = entities.get("title")
        if not title:
            return None
        hits = DB.complete_documents.find(title=title)
        if hits:
            return (hits, "title")
        return None

    def by_file_name(entities):
        file_name = entities.get("file_name")
        if not file_name:
            return None
        hits = DB.complete_documents.find(file_path=file_name)
        if hits:
            return (hits, "file_name")
        return None

    def by_position(entities):
        position_id = entities.get("position")
        if not position_id:
            return None
        hits = DB.complete_documents.find_by_position(position_id)
        if hits:
            return (hits, "position")
        return None

    # --------------------------------------------
    # FALLBACK HANDLER (FTS → serializer_override)
    # --------------------------------------------

    def fallback_handler(q: str):
        # 1) CompleteDocument FTS
        fts_hits = DB.complete_documents.search_text(q)
        if fts_hits:
            return (fts_hits, "complete_document_fts", _serialize_fts_row)

        # 2) Chunk-level FTS
        chunk_hits = DB.documents.search_fts(q)
        if chunk_hits:
            return (chunk_hits, "document_chunk_fts", _serialize_chunk_row)

        return None

    # --------------------------------------------
    # EXECUTE ROUTER
    # --------------------------------------------
    return router.route(
        text=text,
        intent=intent,
        confidence=confidence,
        entities=entities,
        priority_handlers=[
            by_doc_id,
            by_title,
            by_file_name,
            by_position,
        ],
        fallback=fallback_handler
    )


# ----------------------------------------------------------------------
# SERIALIZERS
# ----------------------------------------------------------------------
def _serialize_complete_document(doc):
    """Handles ORM objects and dict-like FTS rows gracefully."""
    if isinstance(doc, dict):
        # Allow dict rows in edge cases
        return {
            "id": doc.get("id"),
            "title": doc.get("title") or doc.get("name"),
            "file_path": doc.get("file_path"),
            "has_embedding": doc.get("embedding") is not None
        }

    # Normal ORM object
    return {
        "id": doc.id,
        "title": getattr(doc, "title", None) or getattr(doc, "name", None),
        "file_path": getattr(doc, "file_path", None),
        "has_embedding": getattr(doc, "embedding", None) is not None,
    }


def _serialize_fts_row(row: dict):
    """Serialize a single CompleteDocument FTS row."""
    return {
        "id": row.get("document_id")
              or row.get("complete_document_id")
              or row.get("id"),
        "title": row.get("title") or row.get("name"),
        "rank": float(row.get("rank", 0)),
    }


def _serialize_chunk_row(row: dict):
    """Serialize a single chunk-level FTS row."""
    return {
        "chunk_id": row.get("id"),
        "document_id": row.get("document_id"),
        "snippet": row.get("snippet"),
        "rank": float(row.get("rank", 0)),
    }

