from __future__ import annotations

from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import text

from modules.configuration.log_config import (
    info_id,
    warning_id,
    debug_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    Document,
    Image,
    ImageCompletedDocumentAssociation,
)


class DocumentService:
    """
    Pure domain service for Document (text chunk).

    HARD RULES:
    - NEVER open sessions
    - NEVER close sessions
    - NEVER commit
    - NEVER rollback
    - Orchestrator owns transactions

    Notes:
    - Supports both legacy flat-text chunking and page-aware chunking.
    - Page-aware chunking stores page_number in doc_metadata so later
      image->chunk association by page can work deterministically.
    """

    # ----------------------------------------------------------------------
    # CREATE / UPDATE
    # ----------------------------------------------------------------------

    @with_request_id
    def save(
        self,
        session: Session,
        *,
        name: str,
        file_path: str,
        content: Optional[str] = None,
        complete_document_id: Optional[int] = None,
        rev: str = "R0",
        doc_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Document:
        if session is None:
            raise RuntimeError("Session required for DocumentService.save")

        if not name or not str(name).strip():
            raise ValueError("name is required")

        if not file_path or not str(file_path).strip():
            raise ValueError("file_path is required")

        if doc_id is not None:
            chunk = session.get(Document, doc_id)
            if not chunk:
                raise ValueError(f"Document id={doc_id} not found")

            chunk.name = name
            chunk.file_path = file_path
            chunk.content = content
            chunk.rev = rev

            if doc_metadata is not None:
                chunk.doc_metadata = doc_metadata

            session.flush()

            debug_id(
                f"[DocumentService] Updated chunk id={chunk.id} "
                f"complete_document_id={chunk.complete_document_id}",
                request_id,
            )
            return chunk

        chunk = Document(
            name=name,
            file_path=file_path,
            content=content,
            complete_document_id=complete_document_id,
            rev=rev,
            doc_metadata=doc_metadata or {},
        )

        session.add(chunk)
        session.flush()

        debug_id(
            f"[DocumentService] Created chunk id={chunk.id} "
            f"complete_document_id={complete_document_id}",
            request_id,
        )
        return chunk

    # ----------------------------------------------------------------------
    # CREATE CHUNKS
    # ----------------------------------------------------------------------

    @with_request_id
    def create_chunks(
            self,
            session: Session,
            *,
            complete_document_id: int,
            text: Optional[str] = None,
            pages: Optional[List[Dict[str, Any]]] = None,
            file_path: Optional[str] = None,
            chunk_size: int = 1000,
            overlap: int = 100,
            request_id: Optional[str] = None,
    ) -> List[int]:
        """
        Create chunks for a CompleteDocument.

        Supported modes:

        1) Page-aware mode
           Preferred when `pages` is provided and can be normalized into:
               {
                   "page_number": <int>,
                   "text": <str>
               }

        2) Legacy flat-text mode
           Used only when page-aware input is missing or unusable.

        HARD RULES:
        - NEVER open sessions here
        - NEVER close sessions here
        - NEVER commit here
        - NEVER rollback here
        - Caller/orchestrator owns the transaction
        """
        if session is None:
            raise RuntimeError("Session required for DocumentService.create_chunks")

        if not complete_document_id:
            raise ValueError("complete_document_id is required")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        if overlap < 0:
            raise ValueError("overlap must be >= 0")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        # Helpful debug so you can verify what arrived from extraction
        debug_id(
            f"[DocumentService] create_chunks input | "
            f"complete_document_id={complete_document_id} | "
            f"pages_count={len(pages) if pages else 0} | "
            f"text_len={len(text) if text else 0}",
            request_id,
        )

        if pages:
            debug_id(
                f"[DocumentService] raw pages sample={pages[:2]}",
                request_id,
            )

        normalized_pages = self._normalize_pages(pages)

        if normalized_pages:
            debug_id(
                f"[DocumentService] normalized page-aware input detected | "
                f"pages_count={len(normalized_pages)} | "
                f"sample={normalized_pages[:2]}",
                request_id,
            )

            return self._create_page_aware_chunks(
                session=session,
                complete_document_id=complete_document_id,
                pages=normalized_pages,
                file_path=file_path,
                chunk_size=chunk_size,
                overlap=overlap,
                request_id=request_id,
            )

        if pages and not normalized_pages:
            warning_id(
                f"[DocumentService] pages were provided but could not be normalized; "
                f"falling back to flat chunking | complete_document_id={complete_document_id}",
                request_id,
            )

        if not text or not text.strip():
            warning_id(
                f"[DocumentService] No text/pages provided for complete_document_id={complete_document_id}",
                request_id,
            )
            return []

        debug_id(
            f"[DocumentService] using flat chunking fallback | complete_document_id={complete_document_id}",
            request_id,
        )

        return self._create_flat_chunks(
            session=session,
            complete_document_id=complete_document_id,
            text=text,
            file_path=file_path,
            chunk_size=chunk_size,
            overlap=overlap,
            request_id=request_id,
        )
    # ----------------------------------------------------------------------
    # INTERNAL: PAGE-AWARE CHUNKING
    # ----------------------------------------------------------------------

    def _create_page_aware_chunks(
        self,
        session: Session,
        *,
        complete_document_id: int,
        pages: List[Dict[str, Any]],
        file_path: Optional[str],
        chunk_size: int,
        overlap: int,
        request_id: Optional[str],
    ) -> List[int]:
        created_ids: List[int] = []
        global_chunk_index = 0
        global_char_cursor = 0
        step = chunk_size - overlap

        for page_idx, page in enumerate(pages):
            page_text = str(page.get("text") or "").strip()
            if not page_text:
                continue

            raw_page_number = page.get("page_number")
            try:
                page_number = int(raw_page_number) if raw_page_number is not None else (page_idx + 1)
            except Exception:
                page_number = page_idx + 1

            page_length = len(page_text)
            start = 0
            page_chunk_index = 0

            while start < page_length:
                end = min(start + chunk_size, page_length)
                chunk_text = page_text[start:end]

                if not chunk_text.strip():
                    start += step
                    page_chunk_index += 1
                    continue

                metadata = {
                    "chunk_index": global_chunk_index,
                    "page_chunk_index": page_chunk_index,
                    "page_number": page_number,
                    "char_start": global_char_cursor + start,
                    "char_end": global_char_cursor + end,
                    "page_char_start": start,
                    "page_char_end": end,
                    "chunking_strategy": "page_aware",
                }

                chunk = Document(
                    name=f"chunk_{complete_document_id}_{global_chunk_index}",
                    file_path=file_path,
                    content=chunk_text,
                    complete_document_id=complete_document_id,
                    rev="R0",
                    doc_metadata=metadata,
                )

                session.add(chunk)
                session.flush()

                created_ids.append(chunk.id)

                global_chunk_index += 1
                page_chunk_index += 1
                start += step

            global_char_cursor += page_length + 1

        debug_id(
            f"[DocumentService] Page-aware chunks created={len(created_ids)} "
            f"complete_document_id={complete_document_id}",
            request_id,
        )

        return created_ids

    # ----------------------------------------------------------------------
    # INTERNAL: LEGACY FLAT CHUNKING
    # ----------------------------------------------------------------------

    def _create_flat_chunks(
        self,
        session: Session,
        *,
        complete_document_id: int,
        text: str,
        file_path: Optional[str],
        chunk_size: int,
        overlap: int,
        request_id: Optional[str],
    ) -> List[int]:
        created_ids: List[int] = []
        text = str(text)
        text_length = len(text)
        index = 0
        start = 0
        step = chunk_size - overlap

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]

            if not chunk_text.strip():
                start += step
                index += 1
                continue

            chunk = Document(
                name=f"chunk_{complete_document_id}_{index}",
                file_path=file_path,
                content=chunk_text,
                complete_document_id=complete_document_id,
                rev="R0",
                doc_metadata={
                    "chunk_index": index,
                    "char_start": start,
                    "char_end": end,
                    "chunking_strategy": "flat",
                },
            )

            session.add(chunk)
            session.flush()

            created_ids.append(chunk.id)

            index += 1
            start += step

        debug_id(
            f"[DocumentService] Flat chunks created={len(created_ids)} "
            f"complete_document_id={complete_document_id}",
            request_id,
        )

        return created_ids

    # ----------------------------------------------------------------------
    # INTERNAL: PAGE NORMALIZATION
    # ----------------------------------------------------------------------

    @staticmethod
    def _normalize_pages(
        pages: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Normalize incoming page data into:
            [{"page_number": int, "text": str}, ...]

        Silently skips invalid/empty pages.
        """
        if not pages:
            return []

        normalized: List[Dict[str, Any]] = []

        for idx, page in enumerate(pages):
            if not isinstance(page, dict):
                continue

            text_value = page.get("text")
            if text_value is None:
                continue

            text_str = str(text_value).strip()
            if not text_str:
                continue

            raw_page_number = page.get("page_number", idx + 1)

            try:
                page_number = int(raw_page_number)
            except Exception:
                page_number = idx + 1

            normalized.append(
                {
                    "page_number": page_number,
                    "text": text_str,
                }
            )

        return normalized

    # ----------------------------------------------------------------------
    # GET
    # ----------------------------------------------------------------------

    @with_request_id
    def get(
        self,
        session: Session,
        *,
        doc_id: Optional[int] = None,
        document_id: Optional[int] = None,
        id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Document]:
        if session is None:
            raise RuntimeError("Session required for DocumentService.get")

        resolved_id = doc_id or document_id or id

        if resolved_id is None:
            raise ValueError(
                "DocumentService.get() requires one of: doc_id, document_id, or id"
            )

        doc = session.get(Document, resolved_id)

        if not doc:
            warning_id(f"[DocumentService] Document id={resolved_id} not found", request_id)

        return doc

    # ----------------------------------------------------------------------
    # GET MULTIPLE
    # ----------------------------------------------------------------------

    @with_request_id
    def get_by_ids(
        self,
        session: Session,
        *,
        ids: List[int],
        request_id: Optional[str] = None,
    ) -> List[Document]:
        if session is None:
            raise RuntimeError("Session required for DocumentService.get_by_ids")

        if not ids:
            return []

        results = (
            session.query(Document)
            .filter(Document.id.in_(ids))
            .order_by(Document.id.asc())
            .all()
        )

        debug_id(
            f"[DocumentService] get_by_ids requested={len(ids)} returned={len(results)}",
            request_id,
        )

        return results

    # ----------------------------------------------------------------------
    # DELETE
    # ----------------------------------------------------------------------

    @with_request_id
    def remove(
        self,
        session: Session,
        *,
        doc_id: int,
        request_id: Optional[str] = None,
    ) -> bool:
        if session is None:
            raise RuntimeError("Session required for DocumentService.remove")

        doc = session.get(Document, doc_id)
        if not doc:
            warning_id(f"[DocumentService] Document id={doc_id} not found", request_id)
            return False

        session.delete(doc)
        session.flush()

        info_id(f"[DocumentService] Document staged for deletion id={doc_id}", request_id)
        return True

    # ----------------------------------------------------------------------
    # FIND
    # ----------------------------------------------------------------------

    @with_request_id
    def find(
        self,
        session: Session,
        *,
        name: Optional[str] = None,
        file_path: Optional[str] = None,
        complete_document_id: Optional[int] = None,
        has_images: Optional[bool] = None,
        limit: int = 50,
        request_id: Optional[str] = None,
    ) -> List[Document]:
        if session is None:
            raise RuntimeError("Session required for DocumentService.find")

        limit = max(1, min(int(limit or 50), 1000))

        query = session.query(Document)

        if name:
            query = query.filter(Document.name.ilike(f"%{name}%"))

        if file_path:
            query = query.filter(Document.file_path.ilike(f"%{file_path}%"))

        if complete_document_id is not None:
            query = query.filter(Document.complete_document_id == complete_document_id)

        if has_images is True:
            query = query.join(
                ImageCompletedDocumentAssociation,
                ImageCompletedDocumentAssociation.document_id == Document.id,
            ).distinct()

        results = query.order_by(Document.id.asc()).limit(limit).all()

        debug_id(
            f"[DocumentService] find returned={len(results)} "
            f"complete_document_id={complete_document_id}",
            request_id,
        )

        return results

    # ----------------------------------------------------------------------
    # IMAGE RELATIONSHIPS
    # ----------------------------------------------------------------------

    @with_request_id
    def get_images_for_chunk(
        self,
        session: Session,
        *,
        chunk_id: int,
        request_id: Optional[str] = None,
    ) -> List[Image]:
        if session is None:
            raise RuntimeError("Session required for DocumentService.get_images_for_chunk")

        results = (
            session.query(Image)
            .join(
                ImageCompletedDocumentAssociation,
                ImageCompletedDocumentAssociation.image_id == Image.id,
            )
            .filter(ImageCompletedDocumentAssociation.document_id == chunk_id)
            .order_by(Image.id.asc())
            .all()
        )

        debug_id(
            f"[DocumentService] chunk_id={chunk_id} images_found={len(results)}",
            request_id,
        )

        return results

    # ----------------------------------------------------------------------
    # FULL-TEXT SEARCH
    # ----------------------------------------------------------------------

    @with_request_id
    def search_fts(
        self,
        session: Session,
        *,
        search_text: str,
        limit: int = 20,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if session is None:
            raise RuntimeError("Session required for DocumentService.search_fts")

        if not search_text or not str(search_text).strip():
            return []

        limit = max(1, min(int(limit or 20), 1000))

        sql = text(
            """
            SELECT
                id,
                name,
                file_path,
                rev,
                ts_rank(
                    to_tsvector('english', COALESCE(content, '')),
                    plainto_tsquery('english', :query)
                ) AS rank
            FROM document
            WHERE to_tsvector('english', COALESCE(content, ''))
                  @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC, id ASC
            LIMIT :limit
            """
        )

        rows = session.execute(
            sql,
            {"query": search_text, "limit": limit},
        ).fetchall()

        results = [
            {
                "id": r[0],
                "name": r[1],
                "file_path": r[2],
                "rev": r[3],
                "rank": float(r[4]),
            }
            for r in rows
        ]

        info_id(
            f"[DocumentService] FTS returned {len(results)} results for query='{search_text}'",
            request_id,
        )

        return results