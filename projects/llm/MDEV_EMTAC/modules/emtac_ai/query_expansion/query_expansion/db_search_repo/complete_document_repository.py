from __future__ import annotations
from typing import List, Optional
from .base_repository import BaseRepository
from modules.configuration.log_config import get_request_id, with_request_id
from modules.configuration.log_config import debug_id, error_id, log_timed_operation
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session


# ---- Robust imports (match your repo style) ----
try:
    from modules.emtacdb.emtacdb_fts import (
        CompleteDocument,
        Document,  # chunks
        Image,
        ImageCompletedDocumentAssociation,
        DocumentEmbedding,
    )  # type: ignore
except Exception:
    try:
        from modules.emtacdb.emtacdb_fts import (
            CompleteDocument,
            Document,
            Image,
            ImageCompletedDocumentAssociation,
            DocumentEmbedding,
        )  # type: ignore
    except Exception:
        # Minimum fallback so imports won't explode during early boot
        from emtacdb_fts import CompleteDocument, Document  # type: ignore
        Image = None  # type: ignore
        ImageCompletedDocumentAssociation = None  # type: ignore
        DocumentEmbedding = None  # type: ignore


class CompleteDocumentRepository(BaseRepository):
    """
    Repository for CompleteDocument domain (top-level docs).
    - Thin wrappers over CompleteDocument classmethods (FTS, similarity, serving files)
    - Simple SQLAlchemy reads
    - Keeps signature parity with UnifiedSearch expectations (limit/threshold/session/with_links)
    """

    # ---------------- Basic fetches ----------------

    @with_request_id
    def get_by_id(self, doc_id: int, request_id: Optional[str] = None) -> Optional[CompleteDocument]:
        rid = request_id or get_request_id()
        sess: Session = self._session()
        debug_id(f"[CompleteDocumentRepository.get_by_id] id={doc_id}", rid)
        try:
            with log_timed_operation("CompleteDocumentRepository.get_by_id", rid):
                return sess.query(CompleteDocument).filter(CompleteDocument.id == doc_id).first()
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def list(
        self,
        *,
        title_ilike: Optional[str] = None,
        rev: Optional[str] = None,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> List[CompleteDocument]:
        """Simple filters against CompleteDocument columns."""
        rid = request_id or get_request_id()
        sess: Session = self._session()
        debug_id(f"[CompleteDocumentRepository.list] title={title_ilike}, rev={rev}, limit={limit}", rid)
        try:
            with log_timed_operation("CompleteDocumentRepository.list", rid):
                q = sess.query(CompleteDocument)
                if title_ilike:
                    q = q.filter(CompleteDocument.title.ilike(f"%{title_ilike}%"))
                if rev:
                    q = q.filter(CompleteDocument.rev == rev)
                q = q.limit(limit)
                rows = q.all()
                debug_id(f"[CompleteDocumentRepository.list] -> {len(rows)} row(s)", rid)
                return rows
        finally:
            if self._owns_session():
                sess.close()

    # ---------------- Search (FTS) ----------------
    # Matches UnifiedSearch call style: search_by_text(query, session=?, limit=?, threshold=?, with_links=?)

    @with_request_id
    def search_by_text(
        self,
        query: str,
        *,
        session: Optional[Session] = None,
        limit: int = 25,
        threshold: Optional[float] = None,  # accepted for compatibility; ignored by FTS
        similarity_threshold: int = 70,     # accepted for compatibility; ignored by FTS
        with_links: bool = False,
        request_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Delegate to CompleteDocument.search_by_text which already implements:
        - PostgreSQL FTS with highlight
        - graceful fallback ILIKE
        - returns list[CompleteDocument] or HTML links (if with_links=True)
        """
        rid = request_id or get_request_id()
        debug_id(f"[CompleteDocumentRepository.search_by_text] q='{query}', limit={limit}, with_links={with_links}", rid)
        # Prefer using the model method so we keep a single FTS implementation.
        return CompleteDocument.search_by_text(
            query,
            session=session,
            limit=limit,
            threshold=threshold,
            similarity_threshold=similarity_threshold,
            with_links=with_links,
            request_id=rid,
            **kwargs,
        )

    @with_request_id
    def search_documents(
        self,
        query: str,
        *,
        limit: int = 50,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Thin wrapper over CompleteDocument.search_documents (FTS results as dicts)."""
        rid = request_id or get_request_id()
        debug_id(f"[CompleteDocumentRepository.search_documents] q='{query}', limit={limit}", rid)
        return CompleteDocument.search_documents(query, limit=limit, request_id=rid)

    # ---------------- Similarity (pg_trgm) ----------------

    @with_request_id
    def find_similar(
        self,
        document_id: int,
        *,
        threshold: float = 0.3,
        limit: int = 10,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Thin wrapper over CompleteDocument.find_similar (pg_trgm)."""
        rid = request_id or get_request_id()
        debug_id(f"[CompleteDocumentRepository.find_similar] id={document_id}, thr={threshold}, limit={limit}", rid)
        return CompleteDocument.find_similar(document_id, threshold=threshold, limit=limit, request_id=rid)

    # ---------------- Chunks & image conveniences ----------------

    @with_request_id
    def chunks_for_complete_document(
        self, complete_document_id: int, *, limit: int = 500, request_id: Optional[str] = None
    ) -> List[Document]:
        """Return chunk rows for a CompleteDocument (pairs with DocumentRepository.chunks_for_complete_document_id)."""
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("CompleteDocumentRepository.chunks_for_complete_document", rid):
                q = (
                    sess.query(Document)
                    .filter(Document.complete_document_id == complete_document_id)
                    .limit(limit)
                )
                rows = q.all()
                debug_id(f"[CompleteDocumentRepository.chunks_for_complete_document] -> {len(rows)}", rid)
                return rows
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def get_images_with_chunk_context(
        self, complete_document_id: int, request_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Delegate to CompleteDocument.get_images_with_chunk_context()."""
        rid = request_id or get_request_id()
        return CompleteDocument.get_images_with_chunk_context(complete_document_id, request_id=rid)

    @with_request_id
    def search_images_by_chunk_text(
        self,
        search_text: str,
        *,
        complete_document_id: Optional[int] = None,
        confidence_threshold: float = 0.5,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Delegate to CompleteDocument.search_images_by_chunk_text()."""
        rid = request_id or get_request_id()
        return CompleteDocument.search_images_by_chunk_text(
            search_text,
            document_id=complete_document_id,
            confidence_threshold=confidence_threshold,
            request_id=rid,
        )

    # ---------------- File serving ----------------

    @with_request_id
    def serve_file(
        self, complete_document_id: int, *, download: Optional[bool] = None, request_id: Optional[str] = None
    ):
        """
        Delegate to CompleteDocument.serve_file() which returns (success, response, status_code).
        Keep AIST/chatbot route unchanged—controller just calls repo.serve_file(...).
        """
        rid = request_id or get_request_id()
        return CompleteDocument.serve_file(complete_document_id, download=download, request_id=rid)

    # ---------------- Optional: embedding similarity passthroughs ----------------

    @with_request_id
    def search_similar_by_embedding(
        self,
        query_text: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rid = request_id or get_request_id()
        return CompleteDocument.search_similar_by_embedding(
            query_text, limit=limit, threshold=threshold, request_id=rid
        )

    @with_request_id
    def embedding_stats(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        rid = request_id or get_request_id()
        return CompleteDocument.get_embedding_statistics(request_id=rid)

    @with_request_id
    def ensure_pgvector_indexes(self, request_id: Optional[str] = None) -> bool:
        rid = request_id or get_request_id()
        return bool(CompleteDocument.create_pgvector_indexes(request_id=rid))
