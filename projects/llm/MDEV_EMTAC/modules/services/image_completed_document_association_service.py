# modules/services/image_completed_document_association_service.py

from typing import Optional, List, Dict, Any, Tuple

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as SASession

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id,
    error_id,
    warning_id,
    debug_id,
    with_request_id,
    log_timed_operation,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    ImageCompletedDocumentAssociation,
    Image,
    Document,
    CompleteDocument,
)


class ImageCompletedDocumentAssociationService:
    """
    Service layer for ImageCompletedDocumentAssociation.

    Responsibilities:
    - Expose guided extraction as a clean API
    - Provide basic/fallback extraction entry points
    - Query images + chunk context
    - Search by chunk text
    - Bulk update association metadata
    - Provide statistics for dashboards
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    def _get_session(self, session: Optional[SASession]) -> Tuple[SASession, bool]:
        if session is not None:
            return session, False
        return self.db_config.get_main_session(), True

    # ---------------------------------------------------------
    # Guided extraction entry point
    # ---------------------------------------------------------
    @with_request_id
    def guided_extraction_with_mapping(
        self,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any], int]:
        """
        High-level wrapper for ImageCompletedDocumentAssociation.guided_extraction_with_mapping.

        Returns:
            (success, payload_dict, http_status_code)
        """
        rid = get_request_id()
        debug_id(
            f"[ImageCompletedDocumentAssociationService.guided_extraction_with_mapping] "
            f"file_path={file_path}, metadata_keys={list(metadata.keys())}",
            rid,
        )

        with log_timed_operation(
            "ImageCompletedDocumentAssociationService.guided_extraction_with_mapping", rid
        ):
            # The model method creates its own sessions; we just call it.
            success, payload, status = ImageCompletedDocumentAssociation.guided_extraction_with_mapping(
                file_path=file_path,
                metadata=metadata,
                request_id=rid,
            )
            return success, payload, status

    # ---------------------------------------------------------
    # Fallback extraction wrapper
    # ---------------------------------------------------------
    @with_request_id
    def fallback_basic_extraction(
        self,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any], int]:
        """
        Explicit wrapper invoking the fallback basic extraction.

        Useful if you want to bypass structure-guided path and just do basic image extraction.
        """
        rid = get_request_id()
        debug_id(
            f"[ImageCompletedDocumentAssociationService.fallback_basic_extraction] "
            f"file_path={file_path}, metadata_keys={list(metadata.keys())}",
            rid,
        )

        with log_timed_operation(
            "ImageCompletedDocumentAssociationService.fallback_basic_extraction", rid
        ):
            success, payload, status = ImageCompletedDocumentAssociation._fallback_basic_extraction(
                file_path=file_path,
                metadata=metadata,
                request_id=rid,
            )
            return success, payload, status

    # ---------------------------------------------------------
    # Chunk-context and searches
    # ---------------------------------------------------------
    @with_request_id
    def get_images_with_chunk_context(
        self,
        complete_document_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Get images plus their chunk context as a list of dicts for UI / API use.
        Delegates to ImageCompletedDocumentAssociation.get_images_with_chunk_context.
        """
        rid = get_request_id()
        debug_id(
            f"[ImageCompletedDocumentAssociationService.get_images_with_chunk_context] cd_id={complete_document_id}",
            rid,
        )

        with log_timed_operation(
            "ImageCompletedDocumentAssociationService.get_images_with_chunk_context", rid
        ):
            results = ImageCompletedDocumentAssociation.get_images_with_chunk_context(
                complete_document_id=complete_document_id,
                request_id=rid,
            )
            return results

    @with_request_id
    def search_by_chunk_text(
        self,
        search_text: str,
        complete_document_id: Optional[int] = None,
        confidence_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search images via their associated chunk text.
        """
        rid = get_request_id()
        debug_id(
            f"[ImageCompletedDocumentAssociationService.search_by_chunk_text] "
            f"text='{search_text[:40]}...', cd_id={complete_document_id}, "
            f"threshold={confidence_threshold}",
            rid,
        )

        with log_timed_operation(
            "ImageCompletedDocumentAssociationService.search_by_chunk_text", rid
        ):
            return ImageCompletedDocumentAssociation.search_by_chunk_text(
                search_text=search_text,
                complete_document_id=complete_document_id,
                confidence_threshold=confidence_threshold,
                request_id=rid,
            )

    @with_request_id
    def get_association_statistics(
        self,
        complete_document_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics for associations (for dashboards / debugging).
        """
        rid = get_request_id()
        debug_id(
            f"[ImageCompletedDocumentAssociationService.get_association_statistics] cd_id={complete_document_id}",
            rid,
        )

        with log_timed_operation(
            "ImageCompletedDocumentAssociationService.get_association_statistics", rid
        ):
            return ImageCompletedDocumentAssociation.get_association_statistics(
                complete_document_id=complete_document_id,
                request_id=rid,
            )

    # ---------------------------------------------------------
    # Update operations
    # ---------------------------------------------------------
    @with_request_id
    def update_association_confidence(
        self,
        association_id: int,
        new_confidence: float,
    ) -> bool:
        """
        Update confidence_score for a single association.
        """
        rid = get_request_id()
        debug_id(
            f"[ImageCompletedDocumentAssociationService.update_association_confidence] "
            f"assoc_id={association_id}, confidence={new_confidence}",
            rid,
        )

        with log_timed_operation(
            "ImageCompletedDocumentAssociationService.update_association_confidence", rid
        ):
            return ImageCompletedDocumentAssociation.update_association_confidence(
                association_id=association_id,
                new_confidence=new_confidence,
                request_id=rid,
            )

    @with_request_id
    def bulk_update_associations(
        self,
        complete_document_id: int,
        association_method: str = "bulk_update",
        confidence_score: float = 0.7,
    ) -> int:
        """
        Bulk-update method and confidence across all associations for a document.
        """
        rid = get_request_id()
        debug_id(
            f"[ImageCompletedDocumentAssociationService.bulk_update_associations] "
            f"cd_id={complete_document_id}, method={association_method}, "
            f"confidence={confidence_score}",
            rid,
        )

        with log_timed_operation(
            "ImageCompletedDocumentAssociationService.bulk_update_associations", rid
        ):
            return ImageCompletedDocumentAssociation.bulk_update_associations(
                document_id=complete_document_id,
                association_method=association_method,
                confidence_score=confidence_score,
                request_id=rid,
            )

    # ---------------------------------------------------------
    # Debug helpers
    # ---------------------------------------------------------
    @with_request_id
    def debug_chunk_distribution(
        self,
        complete_document_id: int,
    ) -> Dict[int, Any]:
        """
        Expose the debug_chunk_distribution as a service call so you can hit it from a route.
        """
        rid = get_request_id()
        debug_id(
            f"[ImageCompletedDocumentAssociationService.debug_chunk_distribution] cd_id={complete_document_id}",
            rid,
        )

        with log_timed_operation(
            "ImageCompletedDocumentAssociationService.debug_chunk_distribution", rid
        ):
            return ImageCompletedDocumentAssociation.debug_chunk_distribution(
                complete_document_id=complete_document_id,
                request_id=rid,
            )


    # ---------------------------------------------------------
    # Unified resolver (NEW)
    # ---------------------------------------------------------
    @with_request_id
    def resolve_related_entities(
            self,
            *,
            image_id: Optional[int] = None,
            document_id: Optional[int] = None,
            complete_document_id: Optional[int] = None,
            session: Optional[SASession] = None,
    ) -> Dict[str, Any]:
        """
        Service-layer resolver.

        Responsibilities:
        - session management
        - logging & timing
        - delegation to ORM resolver

        Returns ORM objects only.
        NO serialization.
        """

        rid = get_request_id()

        debug_id(
            "[ImageCompletedDocumentAssociationService.resolve_related_entities] "
            f"image_id={image_id}, document_id={document_id}, "
            f"complete_document_id={complete_document_id}",
            rid,
        )

        sess, should_close = self._get_session(session)

        try:
            with log_timed_operation(
                    "ImageCompletedDocumentAssociationService.resolve_related_entities",
                    rid,
            ):

                images, documents, complete_document, associations = (
                    ImageCompletedDocumentAssociation.resolve_related_orm(
                        sess,
                        image_id=image_id,
                        document_id=document_id,
                        complete_document_id=complete_document_id,
                    )
                )

                debug_id(
                    f"[resolve_related_entities] "
                    f"images={len(images)} "
                    f"documents={len(documents)} "
                    f"associations={len(associations)}",
                    rid,
                )

                return {
                    "images": images,
                    "documents": documents,
                    "complete_document": complete_document,
                    "associations": associations,
                }

        except SQLAlchemyError as e:
            error_id(
                f"[resolve_related_entities] Database error: {e}",
                rid,
                exc_info=True,
            )
            raise

        finally:
            if should_close:
                sess.close()


