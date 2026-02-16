# modules/services/document_processing_service.py

from typing import List, Optional, Dict, Tuple, Any

from modules.configuration.log_config import (
    info_id,
    debug_id,
    warning_id,
    error_id,
    with_request_id,
)

from modules.services.db_services import DBServices


class DocumentProcessingService:
    """
    Orchestration-only service for document ingestion.

    HARD RULES:
    - MUST NOT touch the database directly
    - MUST NOT import ORM models
    - MUST NOT open database sessions
    - MUST delegate ALL persistence to DBServices-backed services
    """

    def __init__(self, services: DBServices):
        self.services = services

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    @with_request_id
    def process_upload(
        self,
        files: List[Any],
        metadata: Dict[str, Any],
        *,
        request_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], int]:

        info_id("Starting document processing pipeline", request_id)

        # 1. Validate input files
        validation_error = self._validate_files(files)
        if validation_error:
            warning_id("File validation failed", request_id)
            return False, validation_error, 400

        # 2. Resolve position context
        try:
            position_id = self._resolve_position(metadata, request_id=request_id)
            debug_id(f"Resolved position_id={position_id}", request_id)
        except Exception as e:
            error_id(f"Position resolution failed: {e}", request_id)
            return False, {"error": "Failed to resolve position context"}, 500

        # 3. Persist CompleteDocument records
        try:
            complete_document_ids = self._persist_documents(
                files=files,
                metadata=metadata,
                position_id=position_id,
                request_id=request_id,
            )
        except Exception as e:
            error_id(f"Document persistence failed: {e}", request_id)
            return False, {"error": "Failed to persist documents"}, 500

        if not complete_document_ids:
            return False, {"error": "No documents processed"}, 500

        debug_id(
            f"Persisted {len(complete_document_ids)} CompleteDocuments",
            request_id,
        )

        # 4. Associate documents to position
        if position_id:
            try:
                self._associate_documents_to_position(
                    complete_document_ids,
                    position_id,
                    request_id=request_id,
                )
            except Exception as e:
                error_id(f"Document-position association failed: {e}", request_id)
                return False, {"error": "Document-position association failed"}, 500

        # 5. Per-document processing
        all_chunk_ids: List[int] = []

        for file, complete_document_id in zip(files, complete_document_ids):
            filename = getattr(file, "filename", "<unknown>")

            text = self._extract_text_content(file, request_id=request_id)
            if not text:
                warning_id(f"No text extracted from '{filename}'", request_id)
                continue

            chunk_ids = self._create_chunks(
                complete_document_id=complete_document_id,
                text=text,
                file_path=filename,
                request_id=request_id,
            )

            all_chunk_ids.extend(chunk_ids)

            images = self._extract_images(file, request_id=request_id)
            if images:
                image_ids = self._persist_images(
                    images=images,
                    complete_document_id=complete_document_id,
                    position_id=position_id,
                    request_id=request_id,
                )

                if image_ids and chunk_ids:
                    self._associate_images_to_chunks(
                        image_ids=image_ids,
                        chunk_ids=chunk_ids,
                        request_id=request_id,
                    )

        # 6. Deferred embedding generation
        if all_chunk_ids:
            self._request_embeddings(all_chunk_ids, request_id=request_id)

        # 7. Search / index update (non-fatal)
        try:
            self._update_search_indexes(
                complete_document_ids,
                request_id=request_id,
            )
        except Exception as e:
            warning_id(f"Search index update failed: {e}", request_id)

        info_id("Document processing completed successfully", request_id)

        return True, {
            "complete_document_ids": complete_document_ids,
            "documents_processed": len(complete_document_ids),
            "chunks_created": len(all_chunk_ids),
        }, 200

    # ------------------------------------------------------------------
    # VALIDATION (NO DB)
    # ------------------------------------------------------------------

    def _validate_files(self, files: List[Any]) -> Optional[Dict[str, Any]]:
        if not files or not isinstance(files, list):
            return {"error": "No files provided"}

        for idx, file in enumerate(files):
            filename = getattr(file, "filename", None)
            if not filename or not isinstance(filename, str) or not filename.strip():
                return {
                    "error": "Invalid file",
                    "details": f"File at index {idx} has no valid filename",
                }

        return None

    # ------------------------------------------------------------------
    # DOCUMENT CREATION (DBServices ONLY)
    # ------------------------------------------------------------------

    def _persist_documents(
        self,
        files: List[Any],
        metadata: Dict[str, Any],
        position_id: Optional[int],
        *,
        request_id: Optional[str] = None,
    ) -> List[int]:

        created_ids: List[int] = []

        title = metadata.get("title")
        description = metadata.get("description")
        rev = metadata.get("rev", "R0")

        for file in files:
            filename = getattr(file, "filename", None)

            debug_id(f"Creating CompleteDocument for '{filename}'", request_id)

            doc_id = self.services.complete_documents.create(
                title=title or filename,
                file_path=filename,
                description=description,
                rev=rev,
                position_id=position_id,
                metadata=metadata,
                request_id=request_id,
            )

            if not doc_id:
                raise RuntimeError(f"Failed to create CompleteDocument for '{filename}'")

            created_ids.append(doc_id)

        return created_ids

    # ------------------------------------------------------------------
    # REMAINING STUBS (INTENTIONAL)
    # ------------------------------------------------------------------

    def _resolve_position(
            self,
            metadata: Dict[str, Any],
            *,
            request_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Resolve or create Position context.

        Delegates to:
        - PositionService.add_to_db (via DBServices.positions)

        Returns:
            position_id or None
        """

        # Only pass known Position FK fields
        position_fields = {
            k: v
            for k, v in metadata.items()
            if k in self.services.positions.VALID_FIELDS
               and v not in (None, "", "null")
        }

        if not position_fields:
            debug_id("No position metadata provided; skipping position resolution", request_id)
            return None

        debug_id(
            f"Resolving position with fields: {position_fields}",
            request_id,
        )

        position_id = self.services.positions.add_to_db(
            **position_fields,
            request_id=request_id,
        )

        return position_id

    def _associate_documents_to_position(
            self,
            complete_document_ids: List[int],
            position_id: Optional[int],
            *,
            request_id: Optional[str] = None,
    ) -> None:
        """
        Create CompletedDocumentPositionAssociation records.

        Delegates to:
        - CompletedDocumentPositionService (via DBServices)
        """

        if not position_id:
            debug_id(
                "No position_id provided; skipping document-position association",
                request_id,
            )
            return

        for complete_document_id in complete_document_ids:
            debug_id(
                f"Associating complete_document_id={complete_document_id} "
                f"with position_id={position_id}",
                request_id,
            )

            assoc = self.services.completed_document_positions.associate(
                position_id=position_id,
                complete_document_id=complete_document_id,
                request_id=request_id,
            )

            if not assoc:
                raise RuntimeError(
                    f"Failed to associate document {complete_document_id} "
                    f"with position {position_id}"
                )

    def _extract_text_content(self, file: Any, *, request_id=None) -> Optional[str]:
        raise NotImplementedError

    def _create_chunks(self, complete_document_id: int, text: str, file_path: Optional[str], *, request_id=None) -> List[int]:
        raise NotImplementedError

    def _request_embeddings(self, chunk_ids: List[int], *, request_id=None) -> None:
        raise NotImplementedError

    def _extract_images(self, file: Any, *, request_id=None) -> List[Any]:
        raise NotImplementedError

    def _persist_images(self, images: List[Any], complete_document_id: int, position_id: Optional[int], *, request_id=None) -> List[int]:
        raise NotImplementedError

    def _associate_images_to_chunks(self, image_ids: List[int], chunk_ids: List[int], *, request_id=None) -> None:
        raise NotImplementedError

    def _update_search_indexes(self, complete_document_ids: List[int], *, request_id=None) -> None:
        raise NotImplementedError
