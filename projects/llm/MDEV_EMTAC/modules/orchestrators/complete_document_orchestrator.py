# modules/orchestrators/complete_document_orchestrator.py

from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Optional

from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    warning_id,
    error_id,
)

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.emtacdb.emtacdb_fts import Position, Image, ImageCompletedDocumentAssociation

from modules.services.complete_document_service import CompleteDocumentService
from modules.services.completed_document_position_service import CompletedDocumentPositionService
from modules.services.document_service import DocumentService
from modules.services.ai_models_embedding_service import AIModelsEmbeddingService
from modules.services.document_embedding_service import DocumentEmbeddingService

from modules.services.file_storage_service import FileStorageService
from modules.services.content_extraction_service import ContentExtractionService
from modules.services.search_index_service import SearchIndexService

from modules.services.document_conversion_service import DocumentConversionService
from modules.services.image_guided_association_service import ImageGuidedAssociationService
from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)
from modules.services.upload_idempotency_service import UploadIdempotencyService
from modules.services.batch_embedding_optimization_service import BatchEmbeddingOptimizationService
from modules.services.concurrent_processing_service import ConcurrentProcessingService
from modules.services.ai_model_image_service import AIModelImageService


class CompleteDocumentOrchestrator(BaseOrchestrator):
    """
    CompleteDocument upload orchestrator.

    Key updates:
      - VLM "structured_pages" visuals are stored WITHOUT page_number association (per your latest direction).
      - Native PDF extraction continues to store image.img_metadata.page_number (from extractor),
        and then we associate images to the document using page_number (page-first, deterministic),
        optionally enriching to chunks if chunk metadata contains page_number.

    HARD RULES:
      - Orchestrator owns transaction boundaries
      - Services do not open sessions
    """

    def __init__(self):
        super().__init__()

        self.complete_document_service = CompleteDocumentService()
        self.completed_document_position_service = CompletedDocumentPositionService()
        self.document_service = DocumentService()
        self.embedding_model_service = AIModelsEmbeddingService()
        self.document_embedding_service = DocumentEmbeddingService()

        self.file_storage_service = FileStorageService()
        self.content_extraction_service = ContentExtractionService()
        self.search_index_service = SearchIndexService()

        self.conversion_service = DocumentConversionService()
        self.image_guided_service = ImageGuidedAssociationService()
        self.image_assoc_service = ImageCompletedDocumentAssociationService()

        self.idempotency_service = UploadIdempotencyService()
        self.batch_embedding_service = BatchEmbeddingOptimizationService()
        self.concurrent_service = ConcurrentProcessingService()
        self.image_model_service = AIModelImageService()

    # =========================================================
    # MAIN ENTRY
    # =========================================================
    @with_request_id
    def process_upload(
        self,
        files: List[Any],
        metadata: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not files:
            raise ValueError("No files provided")

        valid_files = [
            f for f in files
            if getattr(f, "filename", None) or isinstance(f, str)
        ]
        if not valid_files:
            raise ValueError("No valid files provided")

        total_chunks = 0
        total_embeddings = 0
        total_images = 0
        deduped = 0
        document_ids: List[int] = []

        # ---------------------------------------
        # PHASE 1 — FILE SAVE + EXTRACTION (NO DB)
        # ---------------------------------------
        prepared_docs: List[Dict[str, Any]] = []

        for file in valid_files:
            original_filename = getattr(file, "filename", None) or str(file)

            stored_path = self.file_storage_service.save(file)

            conversion = self.conversion_service.ensure_pdf(stored_path)
            effective_path = conversion.pdf_path or stored_path

            content_info = self.content_extraction_service.extract(
                effective_path,
                request_id=request_id,
            )

            if not content_info or not content_info.get("text"):
                warning_id(
                    f"[CompleteDocumentOrchestrator] No extractable text; skipping | file={effective_path}",
                    request_id,
                )
                continue

            filename = os.path.basename(original_filename)

            title = (
                metadata.get("title")
                if metadata and metadata.get("title")
                else self._clean_filename(filename)
            )

            prepared_docs.append(
                {
                    "title": title,
                    "file_path": effective_path,
                    "text": content_info.get("text") or "",
                    # May be present if your extraction pipeline returns structured pages
                    "pages": content_info.get("pages"),
                    "conversion": conversion,
                    "original_filename": original_filename,
                }
            )

        # ---------------------------------------
        # PHASE 2 — DATABASE PERSISTENCE (TXN)
        # ---------------------------------------
        with self.transaction() as session:

            position_id = self._resolve_position(
                metadata=metadata,
                session=session,
            )

            for doc_data in prepared_docs:
                title = doc_data["title"]
                effective_path = doc_data["file_path"]
                extracted_text = doc_data["text"]
                structured_pages = doc_data.get("pages")
                conversion = doc_data["conversion"]

                # Idempotency
                signature = self.idempotency_service.compute_signature(
                    file_path=effective_path
                )

                existing_id = self.idempotency_service.find_existing_complete_document_id(
                    session=session,
                    file_sha256=signature.get("file_sha256"),
                )

                if existing_id:
                    document_ids.append(existing_id)
                    deduped += 1
                    info_id(
                        f"[CompleteDocumentOrchestrator] Deduped upload -> existing complete_document_id={existing_id}",
                        request_id,
                    )
                    # Cleanup temp conversion dir if it exists
                    if conversion.temp_dir:
                        self._safe_cleanup_temp_dir(conversion.temp_dir, request_id=request_id)
                    continue

                # Upsert CompleteDocument
                doc = self.complete_document_service.upsert(
                    session=session,
                    title=title,
                    file_path=effective_path,
                    content=extracted_text,
                )

                if position_id:
                    self.completed_document_position_service.associate(
                        session=session,
                        position_id=position_id,
                        complete_document_id=doc.id,
                    )

                document_ids.append(doc.id)

                # -------------------------
                # CHUNKING + EMBEDDINGS
                # -------------------------
                chunk_ids = self.document_service.create_chunks(
                    session=session,
                    complete_document_id=doc.id,
                    text=extracted_text,
                    file_path=effective_path,
                )

                if chunk_ids:
                    total_chunks += len(chunk_ids)

                    chunks = self.document_service.get_by_ids(
                        session=session,
                        ids=chunk_ids,
                    )

                    total_embeddings += self.batch_embedding_service.embed_and_store(
                        session=session,
                        chunks=chunks,
                        embedding_model_service=self.embedding_model_service,
                        document_embedding_service=self.document_embedding_service,
                        complete_document_id=doc.id,
                    )

                # FTS index
                self.search_index_service.index_complete_document(
                    session=session,
                    title=title,
                    content=extracted_text,
                )

                # -------------------------
                # VLM STRUCTURED VISUALS
                # -------------------------
                # Your latest direction: do NOT associate visuals with page number here.
                # We store them as synthetic "images" (records) but no page association.
                if structured_pages:
                    created = self._store_structured_visuals_no_page_assoc(
                        session=session,
                        structured_pages=structured_pages,
                        complete_document_id=doc.id,
                        position_id=position_id,
                        request_id=request_id,
                    )
                    total_images += created

                # -------------------------
                # Native PDF image extraction + page-first association
                # -------------------------
                if (not structured_pages) and effective_path.lower().endswith(".pdf"):
                    extracted = self.image_guided_service.extract_and_associate(
                        session=session,
                        file_path=effective_path,
                        complete_document_id=doc.id,
                        position_id=position_id,
                        embedding_model_service=self.image_model_service,
                        request_id=request_id,
                    )
                    total_images += int(extracted or 0)

                    # Page-first association: uses Image.img_metadata["page_number"]
                    # Deterministic, no guessing.
                    try:
                        # Find images for this complete_document that are not yet associated
                        unassociated_images = (
                            session.query(Image)
                            .join(
                                ImageCompletedDocumentAssociation,
                                Image.id == ImageCompletedDocumentAssociation.image_id,
                                isouter=True,
                            )
                            .filter(ImageCompletedDocumentAssociation.id.is_(None))
                            .all()
                        )

                        # Note: if your extractor already creates associations, this may be empty.
                        created_assocs = self.image_assoc_service.associate_images_by_page(
                            session=session,
                            complete_document_id=doc.id,
                            images=unassociated_images,
                            request_id=request_id,
                        )

                        # Optional enrichment: link to first chunk on the same page (only if chunks have page_number).
                        self.image_assoc_service.associate_images_to_chunks_by_page(
                            session=session,
                            complete_document_id=doc.id,
                            request_id=request_id,
                        )

                        debug_id(
                            f"[CompleteDocumentOrchestrator] Page-first image associations created={created_assocs}",
                            request_id,
                        )

                    except Exception as e:
                        warning_id(
                            f"[CompleteDocumentOrchestrator] Page-first association step failed: {e}",
                            request_id,
                        )

                # Cleanup conversion temp dir
                if conversion.temp_dir:
                    self._safe_cleanup_temp_dir(conversion.temp_dir, request_id=request_id)

        return {
            "status": "success",
            "documents_processed": len(document_ids),
            "document_ids": document_ids,
            "deduped": deduped,
            "chunks_created": total_chunks,
            "embeddings_created": total_embeddings,
            "images_extracted": total_images,
            "position_id": position_id,
        }

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _store_structured_visuals_no_page_assoc(
        self,
        *,
        session,
        structured_pages: List[Dict[str, Any]],
        complete_document_id: int,
        position_id: Optional[int],
        request_id: Optional[str],
    ) -> int:
        """
        Stores VLM "visual elements" as synthetic image records WITHOUT page_number association.

        Assumptions:
          - ImageGuidedAssociationService has create_from_description() as in your orchestrator snippet.
          - That method creates an Image record (and optionally embedding), and may create association rows.
            For this path, we want NO page_number. If create_from_description creates associations,
            it should create them with page_number=None.
        """
        created = 0

        for page in structured_pages:
            visuals = page.get("visual_elements", []) or []
            for visual in visuals:
                description = (visual.get("description") or "").strip()
                label = (visual.get("label") or "Visual").strip()
                visual_type = (visual.get("type") or "Unknown").strip()

                if not description:
                    continue

                try:
                    created += int(
                        self.image_guided_service.create_from_description(
                            session=session,
                            complete_document_id=complete_document_id,
                            position_id=position_id,
                            label=label,
                            visual_type=visual_type,
                            description=description,
                            embedding_model_service=self.image_model_service,
                            request_id=request_id,
                        )
                        or 0
                    )
                except Exception as e:
                    warning_id(
                        f"[CompleteDocumentOrchestrator] Failed to store structured visual '{label}': {e}",
                        request_id,
                    )

        return created

    def _safe_cleanup_temp_dir(self, temp_dir: str, *, request_id: Optional[str]) -> None:
        try:
            self.conversion_service._safe_rmtree(temp_dir, self._rid())
        except Exception as e:
            debug_id(f"[CompleteDocumentOrchestrator] Temp cleanup failed: {e}", request_id)

    # =========================================================
    # UTILITIES
    # =========================================================

    @staticmethod
    def _clean_filename(filename: str) -> str:
        if not filename:
            return "Untitled Document"

        title = os.path.splitext(filename)[0]
        title = re.sub(r"[_\-]+", " ", title)
        title = " ".join(word.capitalize() for word in title.split() if word)

        return title or "Untitled Document"

    # =========================================================
    # POSITION RESOLUTION
    # =========================================================

    def _resolve_position(
        self,
        *,
        metadata: Dict[str, Any],
        session,
    ) -> Optional[int]:

        filters = {
            k: metadata.get(k)
            for k in [
                "site_location_id",
                "area_id",
                "equipment_group_id",
                "model_id",
                "asset_number_id",
                "location_id",
            ]
            if metadata.get(k)
        }

        if not filters:
            return None

        position = session.query(Position).filter_by(**filters).first()
        if position:
            return position.id

        position = Position(**filters)
        session.add(position)
        session.flush()
        return position.id