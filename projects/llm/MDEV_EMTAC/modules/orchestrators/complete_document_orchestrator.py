# modules/orchestrators/complete_document_orchestrator.py

from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Optional

from modules.configuration.config import DATABASE_DIR
from modules.configuration.log_config import (
    with_request_id,
    debug_id,
    info_id,
    warning_id,
    error_id,
)

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.emtacdb.emtacdb_fts import (
    Position,
    Image,
    ImageCompletedDocumentAssociation,
)

from modules.services.complete_document_service import CompleteDocumentService
from modules.services.completed_document_position_service import (
    CompletedDocumentPositionService,
)
from modules.services.document_service import DocumentService
from modules.services.ai_models_embedding_service import AIModelsEmbeddingService
from modules.services.document_embedding_service import DocumentEmbeddingService
from modules.services.file_storage_service import FileStorageService
from modules.services.content_extraction_service import ContentExtractionService
from modules.services.search_index_service import SearchIndexService
from modules.services.document_conversion_service import DocumentConversionService
from modules.services.image_guided_association_service import (
    ImageGuidedAssociationService,
)
from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)
from modules.services.upload_idempotency_service import UploadIdempotencyService
from modules.services.batch_embedding_optimization_service import (
    BatchEmbeddingOptimizationService,
)
from modules.services.concurrent_processing_service import ConcurrentProcessingService
from modules.services.ai_model_image_service import AIModelImageService


class CompleteDocumentOrchestrator(BaseOrchestrator):
    """
    CompleteDocument upload orchestrator.

    Behavior:
      - Saves incoming files
      - Converts to PDF when needed
      - Extracts text / structured pages
      - Creates or reuses a Position from metadata
      - Persists CompleteDocument rows
      - Ensures CompletedDocumentPositionAssociation exists
      - Creates chunks + embeddings
      - Indexes the document
      - Stores VLM structured visuals
      - Extracts PDF-native images
      - Performs page-first image association / enrichment

    HARD RULES:
      - Orchestrator owns transaction boundaries
      - Services do not open sessions
      - Services do not commit/rollback
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
        position_id: Optional[int] = None

        # ---------------------------------------
        # PHASE 1 — FILE SAVE + EXTRACTION (NO DB)
        # ---------------------------------------
        prepared_docs: List[Dict[str, Any]] = []

        for file in valid_files:
            original_filename = getattr(file, "filename", None) or str(file)
            conversion = None

            try:
                stored_path = self.file_storage_service.save(file)

                conversion = self.conversion_service.ensure_pdf(
                    stored_path,
                    request_id=request_id,
                )
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

                    if conversion and conversion.temp_dir:
                        self._safe_cleanup_temp_dir(
                            conversion.temp_dir,
                            request_id=request_id,
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
                        "stored_path": stored_path,
                        "effective_path": effective_path,
                        "text": content_info.get("text") or "",
                        "pages": content_info.get("pages") or [],
                        "conversion": conversion,
                        "original_filename": original_filename,
                        "source_type": content_info.get("source_type"),
                        "method": content_info.get("method"),
                        "scanned": bool(content_info.get("scanned", False)),
                    }
                )

                debug_id(
                    f"[CompleteDocumentOrchestrator] Prepared doc | "
                    f"title='{title}' | stored_path='{stored_path}' | "
                    f"effective_path='{effective_path}' | "
                    f"pages={len(content_info.get('pages') or [])} | "
                    f"scanned={bool(content_info.get('scanned', False))}",
                    request_id,
                )

            except Exception as e:
                error_id(
                    f"[CompleteDocumentOrchestrator] Preparation failed for '{original_filename}': {e}",
                    request_id,
                    exc_info=True,
                )

                if conversion and conversion.temp_dir:
                    self._safe_cleanup_temp_dir(
                        conversion.temp_dir,
                        request_id=request_id,
                    )
                continue

        # ---------------------------------------
        # EARLY EXIT — ALL FILES SKIPPED
        # ---------------------------------------
        if not prepared_docs:
            warning_id(
                "[CompleteDocumentOrchestrator] No documents prepared for persistence; all files skipped",
                request_id,
            )
            return {
                "status": "skipped",
                "message": "No extractable text found for uploaded file(s)",
                "documents_processed": 0,
                "document_ids": [],
                "deduped": 0,
                "chunks_created": 0,
                "embeddings_created": 0,
                "images_extracted": 0,
                "position_id": None,
                "skip_reason": "unsupported_or_no_extractable_text",
            }

        # ---------------------------------------
        # PHASE 2 — DATABASE PERSISTENCE (TXN)
        # ---------------------------------------
        with self.transaction() as session:
            debug_id(
                f"[CompleteDocumentOrchestrator] Incoming metadata for position resolution: {metadata}",
                request_id,
            )

            position_id = self._resolve_position(
                metadata=metadata,
                session=session,
                request_id=request_id,
            )

            debug_id(
                f"[CompleteDocumentOrchestrator] Resolved position_id={position_id}",
                request_id,
            )

            for doc_data in prepared_docs:
                title = doc_data["title"]
                stored_path = doc_data["stored_path"]
                effective_path = doc_data["effective_path"]
                extracted_text = doc_data["text"]
                structured_pages = doc_data.get("pages") or []
                conversion = doc_data["conversion"]

                try:
                    signature = self.idempotency_service.compute_signature(
                        file_path=effective_path
                    )

                    file_sha256 = signature.get("file_sha256")
                    existing_id = self.idempotency_service.find_existing_complete_document_id(
                        session=session,
                        file_sha256=file_sha256,
                    )

                    debug_id(
                        f"[CompleteDocumentOrchestrator] Signature lookup | "
                        f"file='{effective_path}' | file_sha256='{file_sha256}' | "
                        f"existing_id={existing_id}",
                        request_id,
                    )

                    # -------------------------------------------------
                    # DEDUPE PATH
                    # IMPORTANT:
                    # Even if document already exists, still ensure the
                    # CompletedDocumentPositionAssociation exists.
                    # -------------------------------------------------
                    if existing_id:
                        document_ids.append(existing_id)
                        deduped += 1

                        debug_id(
                            f"[CompleteDocumentOrchestrator] Deduped upload detected | "
                            f"existing complete_document_id={existing_id} | "
                            f"position_id={position_id}",
                            request_id,
                        )

                        if position_id:
                            assoc = self.completed_document_position_service.associate(
                                session=session,
                                position_id=position_id,
                                complete_document_id=existing_id,
                                request_id=request_id,
                            )

                            debug_id(
                                f"[CompleteDocumentOrchestrator] Deduped association result | "
                                f"complete_document_id={existing_id} | "
                                f"position_id={position_id} | "
                                f"assoc_id={getattr(assoc, 'id', None)}",
                                request_id,
                            )
                        else:
                            warning_id(
                                f"[CompleteDocumentOrchestrator] Deduped doc {existing_id} "
                                f"has no resolved position_id; skipping "
                                f"completed_document_position_association",
                                request_id,
                            )

                        info_id(
                            f"[CompleteDocumentOrchestrator] Deduped upload -> existing complete_document_id={existing_id}",
                            request_id,
                        )

                        continue

                    db_relative_path = os.path.normpath(
                        os.path.relpath(stored_path, DATABASE_DIR)
                    )

                    debug_id(
                        f"[CompleteDocumentOrchestrator] Saving complete_document path | "
                        f"stored='{stored_path}' | effective='{effective_path}' | "
                        f"db_relative='{db_relative_path}'",
                        request_id,
                    )

                    doc = self.complete_document_service.upsert(
                        session=session,
                        title=title,
                        file_path=db_relative_path,
                        content=extracted_text,
                        request_id=request_id,
                    )

                    if position_id:
                        assoc = self.completed_document_position_service.associate(
                            session=session,
                            position_id=position_id,
                            complete_document_id=doc.id,
                            request_id=request_id,
                        )

                        debug_id(
                            f"[CompleteDocumentOrchestrator] New document association result | "
                            f"complete_document_id={doc.id} | "
                            f"position_id={position_id} | "
                            f"assoc_id={getattr(assoc, 'id', None)}",
                            request_id,
                        )
                    else:
                        warning_id(
                            f"[CompleteDocumentOrchestrator] No position resolved for "
                            f"complete_document_id={doc.id}; skipping "
                            f"completed_document_position_association",
                            request_id,
                        )

                    document_ids.append(doc.id)

                    # -------------------------
                    # CHUNKING + EMBEDDINGS
                    # -------------------------
                    debug_id(
                        f"[CompleteDocumentOrchestrator] structured_pages count={len(structured_pages)} "
                        f"sample={structured_pages[:2] if structured_pages else []}",
                        request_id,
                    )

                    chunk_ids = self.document_service.create_chunks(
                        session=session,
                        complete_document_id=doc.id,
                        text=extracted_text,
                        pages=structured_pages if structured_pages else None,
                        file_path=db_relative_path,
                        request_id=request_id,
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

                    # -------------------------
                    # SEARCH INDEX
                    # -------------------------
                    self.search_index_service.index_complete_document(
                        session=session,
                        title=title,
                        content=extracted_text,
                    )

                    # -------------------------
                    # VLM STRUCTURED VISUALS
                    # -------------------------
                    if structured_pages:
                        created = self._store_structured_visuals_no_page_assoc(
                            session=session,
                            structured_pages=structured_pages,
                            complete_document_id=doc.id,
                            position_id=position_id,
                            request_id=request_id,
                        )
                        total_images += int(created or 0)

                    # -------------------------
                    # PDF IMAGE EXTRACTION
                    # -------------------------
                    if effective_path.lower().endswith(".pdf"):
                        extracted = self.image_guided_service.extract_and_associate(
                            session=session,
                            file_path=effective_path,
                            complete_document_id=doc.id,
                            position_id=position_id,
                            embedding_model_service=self.image_model_service,
                            request_id=request_id,
                        )
                        total_images += int(extracted or 0)

                        # -------------------------------------------------
                        # PAGE-FIRST ASSOCIATION / ENRICHMENT
                        # Only scope to images already linked to THIS doc.
                        # -------------------------------------------------
                        try:
                            current_doc_image_id_rows = (
                                session.query(Image.id)
                                .join(
                                    ImageCompletedDocumentAssociation,
                                    Image.id == ImageCompletedDocumentAssociation.image_id,
                                )
                                .filter(
                                    ImageCompletedDocumentAssociation.complete_document_id == doc.id
                                )
                                .order_by(Image.id.asc())
                                .distinct()
                                .all()
                            )

                            current_doc_image_ids = [row[0] for row in current_doc_image_id_rows]

                            current_doc_images = []
                            if current_doc_image_ids:
                                current_doc_images = (
                                    session.query(Image)
                                    .filter(Image.id.in_(current_doc_image_ids))
                                    .all()
                                )

                            created_assocs = self.image_assoc_service.associate_images_by_page(
                                session=session,
                                complete_document_id=doc.id,
                                images=current_doc_images,
                                request_id=request_id,
                            )

                            enriched_assocs = self.image_assoc_service.associate_images_to_chunks_by_page(
                                session=session,
                                complete_document_id=doc.id,
                                request_id=request_id,
                            )

                            debug_id(
                                f"[CompleteDocumentOrchestrator] Page-first image associations "
                                f"created={created_assocs} enriched={enriched_assocs}",
                                request_id,
                            )

                        except Exception as e:
                            warning_id(
                                f"[CompleteDocumentOrchestrator] Page-first association step failed: {e}",
                                request_id,
                            )

                    info_id(
                        f"[CompleteDocumentOrchestrator] Persisted complete_document_id={doc.id}",
                        request_id,
                    )

                except Exception as e:
                    error_id(
                        f"[CompleteDocumentOrchestrator] Persistence failed for '{effective_path}': {e}",
                        request_id,
                        exc_info=True,
                    )
                    raise

                finally:
                    if conversion and conversion.temp_dir:
                        self._safe_cleanup_temp_dir(
                            conversion.temp_dir,
                            request_id=request_id,
                        )

        # ---------------------------------------
        # FINAL STATUS
        # ---------------------------------------
        if not document_ids:
            warning_id(
                "[CompleteDocumentOrchestrator] No documents persisted; returning skipped status",
                request_id,
            )
            return {
                "status": "skipped",
                "message": "No documents were persisted",
                "documents_processed": 0,
                "document_ids": [],
                "deduped": deduped,
                "chunks_created": total_chunks,
                "embeddings_created": total_embeddings,
                "images_extracted": total_images,
                "position_id": position_id,
                "skip_reason": "nothing_persisted",
            }

        return {
            "status": "success",
            "message": "Processed",
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
        Stores VLM visual elements as synthetic image records without
        page-number association.
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

    def _safe_cleanup_temp_dir(
        self,
        temp_dir: str,
        *,
        request_id: Optional[str],
    ) -> None:
        try:
            self.conversion_service._safe_rmtree(temp_dir, self._rid())
        except Exception as e:
            debug_id(
                f"[CompleteDocumentOrchestrator] Temp cleanup failed: {e}",
                request_id,
            )

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
        request_id: Optional[str] = None,
    ) -> Optional[int]:
        filters = {
            key: metadata.get(key)
            for key in [
                "site_location_id",
                "area_id",
                "equipment_group_id",
                "model_id",
                "asset_number_id",
                "location_id",
            ]
            if metadata.get(key)
        }

        debug_id(
            f"[CompleteDocumentOrchestrator] Position resolution filters={filters}",
            request_id,
        )

        if not filters:
            return None

        position = session.query(Position).filter_by(**filters).first()
        if position:
            debug_id(
                f"[CompleteDocumentOrchestrator] Reusing existing position id={position.id}",
                request_id,
            )
            return position.id

        position = Position(**filters)
        session.add(position)
        session.flush()

        info_id(
            f"[CompleteDocumentOrchestrator] Created new position id={position.id} from filters={filters}",
            request_id,
        )
        return position.id