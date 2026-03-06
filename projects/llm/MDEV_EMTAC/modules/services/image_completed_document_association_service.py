# modules/services/image_completed_document_association_service.py

from __future__ import annotations

from typing import Optional, List, Dict, Any, Iterable
import json

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    info_id,
    warning_id,
    debug_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    ImageCompletedDocumentAssociation,
    Image,
    Document,
)


class ImageCompletedDocumentAssociationService:
    """
    Pure domain service for ImageCompletedDocumentAssociation.

    HARD RULES:
    - NEVER open sessions
    - NEVER close sessions
    - NEVER commit
    - NEVER rollback
    - Orchestrator owns transactions
    """

    # ---------------------------------------------------------
    # CREATE (WRITE SIDE)
    # ---------------------------------------------------------

    @with_request_id
    def create_association(
        self,
        session: Session,
        *,
        complete_document_id: int,
        image_id: int,
        document_id: Optional[int] = None,
        page_number: Optional[int] = None,
        chunk_index: Optional[int] = None,
        association_method: str = "page",
        confidence_score: Optional[float] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> ImageCompletedDocumentAssociation:
        if session is None:
            raise RuntimeError("Session required for create_association")

        assoc = ImageCompletedDocumentAssociation(
            complete_document_id=complete_document_id,
            image_id=image_id,
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
            association_method=association_method,
            confidence_score=confidence_score,
            context_metadata=context_metadata,
        )

        session.add(assoc)
        session.flush()

        debug_id(
            f"[ImgAssocService] Association staged id={assoc.id} "
            f"doc={complete_document_id} img={image_id} page={page_number} chunk={document_id}",
            request_id,
        )
        return assoc

    # ---------------------------------------------------------
    # READ OPERATIONS
    # ---------------------------------------------------------

    @with_request_id
    def get_images_with_chunk_context(
        self,
        session: Session,
        *,
        complete_document_id: int,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return ImageCompletedDocumentAssociation.get_images_with_chunk_context(
            session=session,
            complete_document_id=complete_document_id,
        )

    @with_request_id
    def search_by_chunk_text(
        self,
        session: Session,
        *,
        search_text: str,
        complete_document_id: Optional[int] = None,
        confidence_threshold: float = 0.5,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return ImageCompletedDocumentAssociation.search_by_chunk_text(
            session=session,
            search_text=search_text,
            complete_document_id=complete_document_id,
            confidence_threshold=confidence_threshold,
        )

    @with_request_id
    def get_association_statistics(
        self,
        session: Session,
        *,
        complete_document_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return ImageCompletedDocumentAssociation.get_association_statistics(
            session=session,
            complete_document_id=complete_document_id,
        )

    # ---------------------------------------------------------
    # UPDATE OPERATIONS
    # ---------------------------------------------------------

    @with_request_id
    def update_association_confidence(
        self,
        session: Session,
        *,
        association_id: int,
        new_confidence: float,
        request_id: Optional[str] = None,
    ) -> bool:
        assoc = session.get(ImageCompletedDocumentAssociation, association_id)

        if not assoc:
            warning_id(f"[ImgAssocService] Association id={association_id} not found", request_id)
            return False

        assoc.confidence_score = new_confidence
        session.flush()

        debug_id(f"[ImgAssocService] Updated confidence id={association_id} -> {new_confidence}", request_id)
        return True

    @with_request_id
    def bulk_update_associations(
        self,
        session: Session,
        *,
        complete_document_id: int,
        association_method: str = "bulk_update",
        confidence_score: float = 0.7,
        request_id: Optional[str] = None,
    ) -> int:
        associations = (
            session.query(ImageCompletedDocumentAssociation)
            .filter(ImageCompletedDocumentAssociation.complete_document_id == complete_document_id)
            .all()
        )

        count = 0
        for assoc in associations:
            assoc.association_method = association_method
            assoc.confidence_score = confidence_score
            count += 1

        session.flush()

        info_id(
            f"[ImgAssocService] Bulk updated {count} associations for complete_document_id={complete_document_id}",
            request_id,
        )
        return count

    # ---------------------------------------------------------
    # PAGE-FIRST ASSOCIATION HELPERS
    # ---------------------------------------------------------

    @with_request_id
    def associate_images_by_page(
        self,
        session: Session,
        *,
        complete_document_id: int,
        images: Iterable[Image],
        request_id: Optional[str] = None,
        association_method: str = "page",
        default_confidence: float = 0.85,
    ) -> int:
        """
        Creates/ensures associations that at minimum link:
            complete_document_id + image_id + page_number

        It does NOT invent page numbers.
        It uses image.img_metadata['page_number'] if present.

        Returns count of created associations.
        """
        created = 0

        for img in images:
            page_number = None
            try:
                md = img.img_metadata or {}
                page_number = md.get("page_number")
            except Exception:
                page_number = None

            if page_number is None:
                warning_id(
                    f"[ImgAssocService] image_id={img.id} missing img_metadata.page_number; skipping page association",
                    request_id,
                )
                continue

            existing = (
                session.query(ImageCompletedDocumentAssociation)
                .filter(
                    ImageCompletedDocumentAssociation.complete_document_id == complete_document_id,
                    ImageCompletedDocumentAssociation.image_id == img.id,
                )
                .first()
            )

            if existing:
                # keep existing; optionally fill in page_number if missing
                if existing.page_number is None:
                    existing.page_number = int(page_number)
                    existing.association_method = existing.association_method or association_method
                    if existing.confidence_score is None:
                        existing.confidence_score = float(default_confidence)
                    session.flush()
                    debug_id(
                        f"[ImgAssocService] Filled page_number for existing assoc id={existing.id} page={page_number}",
                        request_id,
                    )
                continue

            self.create_association(
                session,
                complete_document_id=complete_document_id,
                image_id=img.id,
                document_id=None,
                page_number=int(page_number),
                chunk_index=None,
                association_method=association_method,
                confidence_score=float(default_confidence),
                context_metadata={
                    "strategy": "page_first",
                    "source": "image.img_metadata.page_number",
                },
                request_id=request_id,
            )
            created += 1

        info_id(
            f"[ImgAssocService] Page-first associations created={created} complete_document_id={complete_document_id}",
            request_id,
        )
        return created

    @with_request_id
    def associate_images_to_chunks_by_page(
        self,
        session: Session,
        *,
        complete_document_id: int,
        request_id: Optional[str] = None,
        association_method: str = "page_chunk",
        default_confidence: float = 0.8,
    ) -> int:
        """
        Optional enrichment step:

        For each image association with a page_number, try to find a chunk (Document)
        whose metadata/doc_metadata has the same page_number, then set document_id.

        This is deterministic and does NOT distribute chunks or guess.
        """
        # Build page -> first_chunk_id map (deterministic)
        chunks: List[Document] = (
            session.query(Document)
            .filter(Document.complete_document_id == complete_document_id)
            .order_by(Document.id.asc())
            .all()
        )

        page_to_chunk_id: Dict[int, int] = {}
        for ch in chunks:
            p = self._extract_chunk_page_number(ch)
            if p is None:
                continue
            # keep first chunk for that page (deterministic)
            if int(p) not in page_to_chunk_id:
                page_to_chunk_id[int(p)] = ch.id

        if not page_to_chunk_id:
            warning_id(
                f"[ImgAssocService] No chunks with page_number metadata for complete_document_id={complete_document_id}",
                request_id,
            )
            return 0

        assocs: List[ImageCompletedDocumentAssociation] = (
            session.query(ImageCompletedDocumentAssociation)
            .filter(ImageCompletedDocumentAssociation.complete_document_id == complete_document_id)
            .all()
        )

        updated = 0

        for assoc in assocs:
            if assoc.document_id is not None:
                continue
            if assoc.page_number is None:
                continue

            chunk_id = page_to_chunk_id.get(int(assoc.page_number))
            if not chunk_id:
                continue

            assoc.document_id = chunk_id
            assoc.association_method = association_method
            if assoc.confidence_score is None:
                assoc.confidence_score = float(default_confidence)

            # You can track why/what happened:
            ctx = assoc.context_metadata or {}
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except Exception:
                    ctx = {"raw": ctx}
            ctx.update({"linked_by": "page_number", "page_number": int(assoc.page_number)})
            assoc.context_metadata = ctx

            updated += 1

        if updated:
            session.flush()

        info_id(
            f"[ImgAssocService] Page->chunk enrichment updated={updated} complete_document_id={complete_document_id}",
            request_id,
        )
        return updated

    # ---------------------------------------------------------
    # RESOLVER
    # ---------------------------------------------------------

    @with_request_id
    def resolve_related_entities(
        self,
        session: Session,
        *,
        image_id: Optional[int] = None,
        document_id: Optional[int] = None,
        complete_document_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        images, documents, complete_document, associations = (
            ImageCompletedDocumentAssociation.resolve_related_orm(
                session=session,
                image_id=image_id,
                document_id=document_id,
                complete_document_id=complete_document_id,
            )
        )

        debug_id(
            f"[ImgAssocService] Resolved entities: images={len(images)}, "
            f"documents={len(documents)}, associations={len(associations)}",
            request_id,
        )

        return {
            "images": images,
            "documents": documents,
            "complete_document": complete_document,
            "associations": associations,
        }

    # ---------------------------------------------------------
    # INTERNAL: extract page_number from chunk metadata
    # ---------------------------------------------------------

    @staticmethod
    def _extract_chunk_page_number(chunk: Document) -> Optional[int]:
        """
        Looks for page_number in chunk.doc_metadata or chunk.metadata.
        Supports dict or JSON string.
        """
        for attr in ("doc_metadata", "metadata"):
            if not hasattr(chunk, attr):
                continue
            raw = getattr(chunk, attr)
            if not raw:
                continue

            if isinstance(raw, dict):
                pn = raw.get("page_number")
                if pn is not None:
                    return int(pn)

            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                    pn = data.get("page_number")
                    if pn is not None:
                        return int(pn)
                except Exception:
                    continue

        return None