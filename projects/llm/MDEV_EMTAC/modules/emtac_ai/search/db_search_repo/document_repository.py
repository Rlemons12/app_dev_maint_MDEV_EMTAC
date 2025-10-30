#modules/emtac_ai/query_expansion/db_search_repo/document_repository.py
from __future__ import annotations
from .base_repository import BaseRepository
from modules.configuration.log_config import get_request_id,with_request_id
from modules.configuration.log_config import debug_id, error_id, log_timed_operation
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from typing import List, Optional, Dict, Any, Sequence


# ---- Robust imports for your tree layout ----
try:
    from modules.emtacdb.emtacdb_fts import (
        Document,
        DocumentEmbedding,
        Image,
        ImageCompletedDocumentAssociation,
    )  # type: ignore
except Exception:
    try:
        from modules.emtacdb.emtacdb_fts import (
            Document,
            DocumentEmbedding,
            Image,
            ImageCompletedDocumentAssociation,
    )  # type: ignore
    except Exception:
        # Minimum viable import fallback: Document must exist
        from emtacdb_fts import Document  # type: ignore
        DocumentEmbedding = None  # type: ignore
        Image = None  # type: ignore
        ImageCompletedDocumentAssociation = None  # type: ignore


class DocumentRepository(BaseRepository):
    """
    Repository for the Document domain:
      - Thin wrappers around your Document classmethods
      - Straight SQLAlchemy queries for common needs
      - FTS helpers over documents_fts
    """

    # ---------------- Baseline CRUD-ish helpers ----------------

    @with_request_id
    def get_by_id(self, doc_id: int, request_id: Optional[str] = None) -> Optional[Document]:
        rid = request_id or get_request_id()
        sess: Session = self._session()
        debug_id(f"[DocumentRepository.get_by_id] id={doc_id}", rid)
        try:
            with log_timed_operation("DocumentRepository.get_by_id", rid):
                return sess.query(Document).filter(Document.id == doc_id).first()
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def search(
        self,
        *,
        name: Optional[str] = None,
        content_text: Optional[str] = None,
        complete_document_id: Optional[int] = None,
        rev: Optional[str] = None,
        has_images: Optional[bool] = None,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Simple SQLAlchemy search against Document table, with optional image existence via join.
        (This is not the FTS—see fts_search() for the documents_fts helper.)
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        debug_id(f"[DocumentRepository.search] args="
                 f"name={name}, content_text={bool(content_text)}, complete_document_id={complete_document_id}, "
                 f"rev={rev}, has_images={has_images}, limit={limit}", rid)
        try:
            with log_timed_operation("DocumentRepository.search", rid):
                q = sess.query(Document)

                # Text filters
                if name:
                    q = q.filter(Document.name.ilike(f"%{name}%"))
                if content_text:
                    q = q.filter(Document.content.ilike(f"%{content_text}%"))

                # Other filters
                if complete_document_id is not None:
                    q = q.filter(Document.complete_document_id == complete_document_id)
                if rev:
                    q = q.filter(Document.rev == rev)

                # has_images via association existence
                if has_images is True:
                    if ImageCompletedDocumentAssociation is None:
                        raise RuntimeError("ImageCompletedDocumentAssociation is not importable.")
                    q = q.join(
                        ImageCompletedDocumentAssociation,
                        ImageCompletedDocumentAssociation.document_id == Document.id
                    ).distinct()
                elif has_images is False:
                    if ImageCompletedDocumentAssociation is None:
                        raise RuntimeError("ImageCompletedDocumentAssociation is not importable.")
                    # Left join + filter for null association
                    q = (
                        q.outerjoin(
                            ImageCompletedDocumentAssociation,
                            ImageCompletedDocumentAssociation.document_id == Document.id,
                        )
                        .filter(ImageCompletedDocumentAssociation.id.is_(None))
                    )

                q = q.limit(limit)
                rows = q.all()
                debug_id(f"[DocumentRepository.search] -> {len(rows)} row(s)", rid)
                return rows
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def documents_by_ids(
        self, ids: Sequence[int], request_id: Optional[str] = None
    ) -> List[Document]:
        rid = request_id or get_request_id()
        if not ids:
            return []
        sess: Session = self._session()
        try:
            with log_timed_operation("DocumentRepository.documents_by_ids", rid):
                rows = sess.query(Document).filter(Document.id.in_(list(ids))).all()
                debug_id(f"documents_by_ids -> {len(rows)}", rid)
                return rows
        finally:
            if self._owns_session():
                sess.close()

    # ---------------- Chunk ↔ image helpers (wrap your model classmethods) ----------------

    @with_request_id
    def get_images_for_chunk(
        self, chunk_id: int, request_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Use your existing Document.get_images_for_chunk classmethod.
        If the association or Image model isn't importable, returns [].
        """
        rid = request_id or get_request_id()
        debug_id(f"[DocumentRepository.get_images_for_chunk] chunk_id={chunk_id}", rid)
        try:
            # Prefer to call the model classmethod (it already manages a session)
            return Document.get_images_for_chunk(chunk_id=chunk_id, request_id=rid)
        except Exception as e:
            error_id(f"get_images_for_chunk failed via model: {e}", rid)

        # Manual fallback (if you prefer to keep it entirely within the repo):
        if Image is None or ImageCompletedDocumentAssociation is None:
            return []
        sess: Session = self._session()
        try:
            with log_timed_operation("DocumentRepository.get_images_for_chunk (fallback)", rid):
                rows = (
                    sess.query(Image, ImageCompletedDocumentAssociation)
                    .join(
                        ImageCompletedDocumentAssociation,
                        Image.id == ImageCompletedDocumentAssociation.image_id,
                    )
                    .filter(ImageCompletedDocumentAssociation.document_id == chunk_id)
                    .all()
                )
                result = []
                for img, assoc in rows:
                    file_path = getattr(img, "file_path", getattr(img, "filepath", None))
                    result.append(
                        {
                            "image_id": img.id,
                            "image_title": getattr(img, "title", None),
                            "image_filepath": file_path,
                            "association_confidence": getattr(assoc, "confidence_score", None),
                            "page_number": getattr(assoc, "page_number", None),
                            "context_metadata": getattr(assoc, "context_metadata", None),
                        }
                    )
                return result
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def find_chunks_with_images(
        self, complete_document_id: int, request_id: Optional[str] = None
    ) -> List[Document]:
        """
        Thin wrapper over your model classmethod.
        """
        rid = request_id or get_request_id()
        debug_id(f"[DocumentRepository.find_chunks_with_images] complete_document_id={complete_document_id}", rid)
        try:
            return Document.find_chunks_with_images(complete_document_id=complete_document_id, request_id=rid)
        except Exception as e:
            error_id(f"find_chunks_with_images failed via model: {e}", rid)
            return []

    # ---------------- Chunk helpers ----------------

    @with_request_id
    def chunks_for_complete_document_id(
        self, complete_document_id: int, limit: int = 500, request_id: Optional[str] = None
    ) -> List[Document]:
        """
        Return Document rows (chunks) that belong to a given CompleteDocument.
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("DocumentRepository.chunks_for_complete_document_id", rid):
                q = (
                    sess.query(Document)
                    .filter(Document.complete_document_id == complete_document_id)
                    .limit(limit)
                )
                rows = q.all()
                debug_id(f"chunks_for_complete_document_id -> {len(rows)}", rid)
                return rows
        finally:
            if self._owns_session():
                sess.close()

    # ---------------- FTS helpers over documents_fts ----------------

    @with_request_id
    def create_enhanced_fts_table(self, request_id: Optional[str] = None) -> bool:
        """
        Thin wrapper over your Document.create_fts_table()
        """
        rid = request_id or get_request_id()
        debug_id("[DocumentRepository.create_enhanced_fts_table] invoked", rid)
        try:
            return bool(Document.create_fts_table())
        except Exception as e:
            error_id(f"create_enhanced_fts_table failed: {e}", rid)
            return False

    @with_request_id
    def fts_search(
        self,
        query_text: str,
        *,
        require_has_images: Optional[bool] = None,
        complete_document_id: Optional[int] = None,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the enhanced FTS table (documents_fts) using PostgreSQL FTS.
        Returns rows with rank and flags.

        Output schema (list of dicts):
          {
            'title': str,
            'content': str | None,
            'chunk_id': int | None,
            'complete_document_id': int | None,
            'has_images': bool,
            'rank': float
          }
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        debug_id(f"[DocumentRepository.fts_search] q='{query_text}', has_images={require_has_images}, "
                 f"complete_document_id={complete_document_id}, limit={limit}", rid)

        try:
            with log_timed_operation("DocumentRepository.fts_search", rid):
                tsq = func.plainto_tsquery("english", query_text)

                base_sql = """
                    SELECT
                        title,
                        content,
                        chunk_id,
                        complete_document_id,
                        has_images,
                        ts_rank(search_vector, plainto_tsquery('english', :q)) AS rank
                    FROM documents_fts
                    WHERE search_vector @@ plainto_tsquery('english', :q)
                """

                where_clauses = []
                params = {"q": query_text}

                if complete_document_id is not None:
                    where_clauses.append("complete_document_id = :cdid")
                    params["cdid"] = complete_document_id

                if require_has_images is True:
                    where_clauses.append("has_images = TRUE")
                elif require_has_images is False:
                    where_clauses.append("has_images = FALSE")

                if where_clauses:
                    base_sql += " AND " + " AND ".join(where_clauses)

                base_sql += " ORDER BY rank DESC LIMIT :limit"
                params["limit"] = limit

                rows = sess.execute(text(base_sql), params).fetchall()

                out: List[Dict[str, Any]] = []
                for r in rows:
                    out.append(
                        {
                            "title": r[0],
                            "content": r[1],
                            "chunk_id": r[2],
                            "complete_document_id": r[3],
                            "has_images": bool(r[4]),
                            "rank": float(r[5]),
                        }
                    )
                debug_id(f"[DocumentRepository.fts_search] -> {len(out)} row(s)", rid)
                return out
        finally:
            if self._owns_session():
                sess.close()

    # ---------------- Convenience: map FTS hits back to Document rows ----------------

    @with_request_id
    def fts_search_chunks_to_documents(
        self,
        query_text: str,
        *,
        require_has_images: Optional[bool] = None,
        complete_document_id: Optional[int] = None,
        limit: int = 50,
        request_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Run FTS and then hydrate the results back into Document chunks (when chunk_id is present).
        Keeps the original ranking order as best as possible.
        """
        rid = request_id or get_request_id()
        fts_hits = self.fts_search(
            query_text,
            require_has_images=require_has_images,
            complete_document_id=complete_document_id,
            limit=limit,
            request_id=rid,
        )

        # Pull chunk_ids in order
        chunk_ids: List[int] = [hit["chunk_id"] for hit in fts_hits if hit.get("chunk_id") is not None]
        if not chunk_ids:
            return []

        # Fetch all chunks; preserve the original FTS order
        chunks = self.documents_by_ids(chunk_ids, request_id=rid)
        by_id = {c.id: c for c in chunks}
        ordered: List[Document] = [by_id[cid] for cid in chunk_ids if cid in by_id]
        return ordered
