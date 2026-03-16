from __future__ import annotations

from typing import Optional, List, Dict, Any, Iterable
import json
import os
import tempfile
from datetime import datetime

import fitz
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    info_id,
    warning_id,
    debug_id,
    error_id,
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
    # CREATE / UPSERT (WRITE SIDE)
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
        """
        Create a new association row.

        Use ensure_association() in most workflows unless you explicitly want
        to force a brand new row.
        """
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
            f"complete_document_id={complete_document_id} "
            f"image_id={image_id} "
            f"page_number={page_number} "
            f"document_id={document_id} "
            f"chunk_index={chunk_index} "
            f"method={association_method}",
            request_id,
        )
        return assoc

    @with_request_id
    def ensure_association(
        self,
        session: Session,
        *,
        complete_document_id: int,
        image_id: int,
        document_id: Optional[int] = None,
        page_number: Optional[int] = None,
        chunk_index: Optional[int] = None,
        association_method: str = "explicit",
        confidence_score: Optional[float] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
        overwrite_document_id: bool = False,
        overwrite_page_number: bool = False,
        overwrite_chunk_index: bool = False,
        overwrite_method: bool = False,
        overwrite_confidence: bool = False,
        request_id: Optional[str] = None,
    ) -> ImageCompletedDocumentAssociation:
        """
        Idempotent association creator/updater.

        Guarantees that an association exists for:

            (complete_document_id, image_id)

        If an association already exists, it fills missing fields and optionally
        overwrites existing values when overwrite_* flags are set.
        """
        if session is None:
            raise RuntimeError("Session required for ensure_association")

        existing = (
            session.query(ImageCompletedDocumentAssociation)
            .filter(
                ImageCompletedDocumentAssociation.complete_document_id == complete_document_id,
                ImageCompletedDocumentAssociation.image_id == image_id,
            )
            .first()
        )

        if existing is None:
            assoc = self.create_association(
                session=session,
                complete_document_id=complete_document_id,
                image_id=image_id,
                document_id=document_id,
                page_number=page_number,
                chunk_index=chunk_index,
                association_method=association_method,
                confidence_score=confidence_score,
                context_metadata=context_metadata,
                request_id=request_id,
            )
            debug_id(
                f"[ImgAssocService] ensure_association created id={assoc.id} "
                f"complete_document_id={complete_document_id} image_id={image_id}",
                request_id,
            )
            return assoc

        changed = False

        if document_id is not None and (existing.document_id is None or overwrite_document_id):
            existing.document_id = document_id
            changed = True

        if page_number is not None and (existing.page_number is None or overwrite_page_number):
            existing.page_number = int(page_number)
            changed = True

        if chunk_index is not None and (existing.chunk_index is None or overwrite_chunk_index):
            existing.chunk_index = int(chunk_index)
            changed = True

        if association_method and (not existing.association_method or overwrite_method):
            existing.association_method = association_method
            changed = True

        if confidence_score is not None and (
            existing.confidence_score is None or overwrite_confidence
        ):
            existing.confidence_score = float(confidence_score)
            changed = True

        if context_metadata:
            merged_context = self._merge_context_metadata(
                existing.context_metadata,
                context_metadata,
            )
            if merged_context != self._coerce_context_metadata(existing.context_metadata):
                existing.context_metadata = merged_context
                changed = True

        if changed:
            session.flush()
            debug_id(
                f"[ImgAssocService] ensure_association updated id={existing.id} "
                f"complete_document_id={complete_document_id} image_id={image_id} "
                f"page_number={existing.page_number} document_id={existing.document_id} "
                f"chunk_index={existing.chunk_index} method={existing.association_method}",
                request_id,
            )
        else:
            debug_id(
                f"[ImgAssocService] ensure_association reused existing id={existing.id} "
                f"complete_document_id={complete_document_id} image_id={image_id}",
                request_id,
            )

        return existing

    @with_request_id
    def ensure_chunk_link(
        self,
        session: Session,
        *,
        complete_document_id: int,
        image_id: int,
        document_id: int,
        page_number: Optional[int] = None,
        chunk_index: Optional[int] = None,
        association_method: str = "guided_chunk",
        confidence_score: float = 0.95,
        context_metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> ImageCompletedDocumentAssociation:
        """
        Convenience helper for guided extraction workflows.

        Best method to call right after an extracted image is saved
        and mapped to a chunk/page.
        """
        return self.ensure_association(
            session=session,
            complete_document_id=complete_document_id,
            image_id=image_id,
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
            association_method=association_method,
            confidence_score=confidence_score,
            context_metadata=context_metadata,
            overwrite_document_id=True,
            overwrite_page_number=False,
            overwrite_chunk_index=True,
            overwrite_method=True,
            overwrite_confidence=True,
            request_id=request_id,
        )

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

    @with_request_id
    def get_by_complete_document(
        self,
        session: Session,
        *,
        complete_document_id: int,
        request_id: Optional[str] = None,
    ) -> List[ImageCompletedDocumentAssociation]:
        assocs = (
            session.query(ImageCompletedDocumentAssociation)
            .filter(ImageCompletedDocumentAssociation.complete_document_id == complete_document_id)
            .order_by(ImageCompletedDocumentAssociation.id.asc())
            .all()
        )
        debug_id(
            f"[ImgAssocService] get_by_complete_document complete_document_id={complete_document_id} "
            f"found={len(assocs)}",
            request_id,
        )
        return assocs

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
            warning_id(
                f"[ImgAssocService] Association id={association_id} not found",
                request_id,
            )
            return False

        assoc.confidence_score = float(new_confidence)
        session.flush()

        debug_id(
            f"[ImgAssocService] Updated confidence id={association_id} -> {new_confidence}",
            request_id,
        )
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
            assoc.confidence_score = float(confidence_score)
            count += 1

        if count:
            session.flush()

        info_id(
            f"[ImgAssocService] Bulk updated {count} associations "
            f"for complete_document_id={complete_document_id}",
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
        Creates or ensures associations using image.img_metadata['page_number'].

        Returns the number of NEW associations created.
        Existing rows may still be enriched in-place.
        """
        created = 0

        for img in images:
            page_number = None
            try:
                md = img.img_metadata or {}
                if isinstance(md, str):
                    md = json.loads(md)
                if isinstance(md, dict):
                    page_number = md.get("page_number")
            except Exception:
                page_number = None

            if page_number is None:
                warning_id(
                    f"[ImgAssocService] image_id={img.id} missing img_metadata.page_number; "
                    f"skipping page association",
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

            if existing is None:
                self.create_association(
                    session=session,
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
                continue

            changed = False

            if existing.page_number is None:
                existing.page_number = int(page_number)
                changed = True

            if not existing.association_method:
                existing.association_method = association_method
                changed = True

            if existing.confidence_score is None:
                existing.confidence_score = float(default_confidence)
                changed = True

            merged_context = self._merge_context_metadata(
                existing.context_metadata,
                {
                    "strategy": "page_first",
                    "source": "image.img_metadata.page_number",
                },
            )
            if merged_context != self._coerce_context_metadata(existing.context_metadata):
                existing.context_metadata = merged_context
                changed = True

            if changed:
                session.flush()
                debug_id(
                    f"[ImgAssocService] Filled existing association id={existing.id} "
                    f"with page_number={page_number}",
                    request_id,
                )

        info_id(
            f"[ImgAssocService] Page-first associations created={created} "
            f"complete_document_id={complete_document_id}",
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
        For each image association with a page_number, find the first chunk
        on that same page and set document_id if missing.

        Returns number of associations updated.
        """
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
            p = int(p)
            if p not in page_to_chunk_id:
                page_to_chunk_id[p] = ch.id

        if not page_to_chunk_id:
            warning_id(
                f"[ImgAssocService] No chunks with page_number metadata "
                f"for complete_document_id={complete_document_id}",
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

            assoc.context_metadata = self._merge_context_metadata(
                assoc.context_metadata,
                {
                    "linked_by": "page_number",
                    "page_number": int(assoc.page_number),
                    "strategy": "page_to_first_chunk",
                },
            )

            updated += 1

        if updated:
            session.flush()

        info_id(
            f"[ImgAssocService] Page->chunk enrichment updated={updated} "
            f"complete_document_id={complete_document_id}",
            request_id,
        )
        return updated

    # ---------------------------------------------------------
    # GUIDED EXTRACTION WORKFLOW
    # ---------------------------------------------------------

    @with_request_id
    def guided_extraction_with_mapping(
        self,
        *,
        session: Session,
        file_path: str,
        metadata: Dict[str, Any],
        request_id: Optional[str] = None,
    ):
        """
        Guided extraction using the caller-owned session.

        HARD RULES:
        - NEVER open a session here
        - NEVER commit here
        - NEVER rollback here
        - Caller/orchestrator owns the transaction

        Alignment:
        - Uses caller-owned session so newly-created, uncommitted chunks are visible
        - Uses 1-based page numbering everywhere stored in metadata/associations
        - Stores JSON columns as Python dicts, not json.dumps strings
        - Uses ensure_chunk_link() instead of raw ORM row creation
        """
        doc = None
        rid = request_id

        try:
            if session is None:
                raise RuntimeError("Session required for guided_extraction_with_mapping")

            if not metadata:
                return False, {"error": "metadata required"}, 400

            complete_document_id = metadata.get("complete_document_id")
            position_id = metadata.get("position_id")

            if not complete_document_id:
                return False, {"error": "complete_document_id required"}, 400

            doc = fitz.open(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            associations_created = 0

            all_chunks: List[Document] = (
                session.query(Document)
                .filter(Document.complete_document_id == complete_document_id)
                .order_by(Document.id.asc())
                .all()
            )

            if not all_chunks:
                warning_id(
                    f"No chunks found for complete_document_id {complete_document_id}",
                    rid,
                )
                return False, {"error": "No chunks found"}, 400

            info_id(
                f"Found {len(all_chunks)} total chunks for document {complete_document_id}",
                rid,
            )

            chunk_page_map = self._create_enhanced_chunk_page_mapping(all_chunks, rid)

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_number = page_idx + 1
                img_list = page.get_images(full=True)

                if not img_list:
                    debug_id(f"No images found on page {page_number}", rid)
                    continue

                page_chunks = self._get_page_chunks_enhanced(
                    chunk_page_map,
                    page_number,
                    all_chunks,
                    rid,
                )

                if not page_chunks:
                    warning_id(
                        f"No chunks found for page {page_number}, skipping image association",
                        rid,
                    )
                    continue

                info_id(
                    f"Page {page_number}: Processing {len(img_list)} images with {len(page_chunks)} available chunks",
                    rid,
                )

                for img_index, img in enumerate(img_list):
                    temp_path = None

                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        ext = base_image.get("ext", "jpg")

                        with tempfile.NamedTemporaryFile(
                            suffix=f".{ext}",
                            delete=False,
                        ) as tmp:
                            tmp.write(image_bytes)
                            temp_path = tmp.name

                        title = f"{file_name} - Page {page_number} Image {img_index + 1}"

                        selected_chunk_info = self._select_best_chunk_for_image(
                            page_chunks,
                            img_index,
                            page_number,
                            img,
                            rid,
                        )

                        if not selected_chunk_info:
                            warning_id(
                                f"Could not select appropriate chunk for image {img_index + 1} on page {page_number}",
                                rid,
                            )
                            continue

                        chunk_index = int(selected_chunk_info["chunk_index"])
                        nearest_chunk = selected_chunk_info["chunk"]
                        association_method = selected_chunk_info["method"]
                        confidence = float(selected_chunk_info["confidence"])

                        debug_id(
                            f"Page {page_number}, Image {img_index + 1}: "
                            f"Selected chunk_index={chunk_index}, "
                            f"chunk_id={nearest_chunk.id}, "
                            f"method={association_method}",
                            rid,
                        )

                        image_metadata = {
                            "page_number": page_number,
                            "image_index": img_index,
                            "extraction_method": "structure_guided_enhanced",
                            "structure_guided": True,
                            "association_method": association_method,
                            "confidence_score": confidence,
                        }

                        image_id = Image.add_to_db(
                            session=session,
                            title=title,
                            file_path=temp_path,
                            description=f"Enhanced guided extraction from {os.path.basename(file_path)}",
                            position_id=position_id,
                            complete_document_id=complete_document_id,
                            metadata=image_metadata,
                            request_id=rid,
                        )

                        if image_id is None:
                            warning_id(f"Failed to save image {title}", rid)
                            continue

                        assoc = self.ensure_chunk_link(
                            session=session,
                            complete_document_id=complete_document_id,
                            image_id=image_id,
                            document_id=nearest_chunk.id,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            association_method=association_method,
                            confidence_score=confidence,
                            context_metadata={
                                "extraction_method": "enhanced_guided",
                                "selection_strategy": association_method,
                                "page_total_images": len(img_list),
                                "page_total_chunks": len(page_chunks),
                                "created_at": datetime.now().isoformat(),
                            },
                            request_id=rid,
                        )

                        associations_created += 1

                        info_id(
                            f"Associated image {image_id} with chunk {nearest_chunk.id} "
                            f"(assoc_id={assoc.id}, page {page_number}, chunk_index {chunk_index})",
                            rid,
                        )

                    except Exception as e:
                        error_id(
                            f"Error processing image {img_index + 1} on page {page_number}: {e}",
                            rid,
                            exc_info=True,
                        )
                        continue

                    finally:
                        if temp_path:
                            try:
                                os.unlink(temp_path)
                            except Exception:
                                pass

            info_id(
                f"Enhanced guided extraction completed: {associations_created} associations created",
                rid,
            )
            return True, {"associations_created": associations_created}, 200

        except Exception as e:
            error_id(f"Enhanced guided extraction failed: {e}", rid, exc_info=True)
            return False, {"error": str(e)}, 500

        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

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
    # INTERNAL HELPERS
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

    @staticmethod
    def _coerce_context_metadata(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return dict(value)

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
                return {"raw": parsed}
            except Exception:
                return {"raw": value}

        return {"raw": value}

    @classmethod
    def _merge_context_metadata(
        cls,
        existing: Any,
        incoming: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base = cls._coerce_context_metadata(existing)

        if incoming:
            for key, value in incoming.items():
                base[key] = value

        return base

    @classmethod
    def _create_enhanced_chunk_page_mapping(
        cls,
        all_chunks: List[Document],
        request_id: Optional[str] = None,
    ) -> Dict[int, List[Document]]:
        """
        Build page_number -> [chunks] mapping using chunk metadata.
        """
        chunk_page_map: Dict[int, List[Document]] = {}
        no_page_count = 0

        for chunk in all_chunks:
            page_number = cls._extract_chunk_page_number(chunk)
            if page_number is None:
                no_page_count += 1
                continue

            page_number = int(page_number)
            chunk_page_map.setdefault(page_number, []).append(chunk)

        info_id(
            f"Chunk page mapping: {len(chunk_page_map)} pages mapped, {no_page_count} chunks without page info",
            request_id,
        )

        for page_number, chunks in chunk_page_map.items():
            debug_id(
                f"Page {page_number}: {len(chunks)} chunks",
                request_id,
            )

        return chunk_page_map

    @classmethod
    def _get_page_chunks_enhanced(
        cls,
        chunk_page_map: Dict[int, List[Document]],
        page_number: int,
        all_chunks: List[Document],
        request_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Resolve chunks for a given page number.

        Strategy:
        1) direct page match from metadata
        2) fallback empty list
        """
        direct_chunks = chunk_page_map.get(int(page_number), [])
        if direct_chunks:
            debug_id(
                f"Strategy 1 - Direct mapping: Found {len(direct_chunks)} chunks for page {page_number}",
                request_id,
            )
            return direct_chunks

        warning_id(
            f"No direct chunk mapping found for page {page_number}",
            request_id,
        )
        return []

    @classmethod
    def _select_best_chunk_for_image(
        cls,
        page_chunks: List[Document],
        img_index: int,
        page_number: int,
        img: Any,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Deterministic chunk selector for a page image.

        Current strategy:
        - If one chunk exists on page, use it
        - If multiple chunks exist, distribute by image index
        """
        if not page_chunks:
            return None

        if len(page_chunks) == 1:
            debug_id(
                f"Single chunk selection: img {img_index} -> chunk 0 (only option)",
                request_id,
            )
            return {
                "chunk_index": 0,
                "chunk": page_chunks[0],
                "method": "single_chunk",
                "confidence": 0.95,
            }

        selected_index = min(img_index, len(page_chunks) - 1)

        debug_id(
            f"Multi-chunk selection: img {img_index} -> chunk {selected_index} (page {page_number})",
            request_id,
        )

        return {
            "chunk_index": selected_index,
            "chunk": page_chunks[selected_index],
            "method": "sequential_page_chunk",
            "confidence": 0.85,
        }