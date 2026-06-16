from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import and_, text

from modules.configuration.log_config import (
    with_request_id, get_request_id, log_timed_operation,
    debug_id, info_id, error_id, warning_id
)
from .base_repository import BaseRepository

# ---- Robust imports against your tree layout ----
try:
    from modules.emtacdb.emtacdb_fts import (
        Image,
        ImageEmbedding,
        ImagePositionAssociation,
        ImageTaskAssociation,
        ImageProblemAssociation,
        ImageCompletedDocumentAssociation,
        PartsPositionImageAssociation,
        ToolImageAssociation,
        Position,
        CompleteDocument,
        CompletedDocumentPositionAssociation,
    )  # type: ignore
except Exception:
    try:
        from modules.emtacdb.emtacdb_fts import (
            Image,
            ImageEmbedding,
            ImagePositionAssociation,
            ImageTaskAssociation,
            ImageProblemAssociation,
            ImageCompletedDocumentAssociation,
            PartsPositionImageAssociation,
            ToolImageAssociation,
            Position,
            CompleteDocument,
            CompletedDocumentPositionAssociation,
        )  # type: ignore
    except Exception:
        # Minimum fallback: Image must exist for this repo to be useful
        from emtacdb_fts import Image  # type: ignore
        ImageEmbedding = None  # type: ignore
        ImagePositionAssociation = None  # type: ignore
        ImageTaskAssociation = None  # type: ignore
        ImageProblemAssociation = None  # type: ignore
        ImageCompletedDocumentAssociation = None  # type: ignore
        PartsPositionImageAssociation = None  # type: ignore
        ToolImageAssociation = None  # type: ignore
        Position = None  # type: ignore
        CompleteDocument = None  # type: ignore
        CompletedDocumentPositionAssociation = None  # type: ignore


class ImageRepository(BaseRepository):
    """
    Repository for Image domain.
    - Wraps your rich Image classmethods (add_to_db, search_images, serve_file, similarity, etc.)
    - Provides simple/consistent session handling and logging
    """

    # ---------------- Basic getters ----------------

    @with_request_id
    def get_by_id(self, image_id: int, request_id: Optional[str] = None) -> Optional[Image]:
        rid = request_id or get_request_id()
        sess: Session = self._session()
        debug_id(f"[ImageRepository.get_by_id] id={image_id}", rid)
        try:
            with log_timed_operation("ImageRepository.get_by_id", rid):
                return sess.query(Image).filter(Image.id == image_id).first()
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def by_ids(self, image_ids: Sequence[int], request_id: Optional[str] = None) -> List[Image]:
        rid = request_id or get_request_id()
        if not image_ids:
            return []
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.by_ids", rid):
                rows = sess.query(Image).filter(Image.id.in_(list(image_ids))).all()
                debug_id(f"by_ids -> {len(rows)}", rid)
                return rows
        finally:
            if self._owns_session():
                sess.close()

    # ---------------- Create / Add ----------------

    @with_request_id
    def add(
        self,
        *,
        title: str,
        file_path: str,
        description: str,
        position_id: Optional[int] = None,
        complete_document_id: Optional[int] = None,
        metadata: Optional[dict] = None,
        request_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Wrap Image.add_to_db (handles copy, relative path fix, optional associations, commit-with-retry).
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        debug_id(f"[ImageRepository.add] title='{title}', file='{file_path}', pos={position_id}, doc={complete_document_id}", rid)
        try:
            with log_timed_operation("ImageRepository.add", rid):
                new_id = Image.add_to_db(
                    session=sess,
                    title=title,
                    file_path=file_path,
                    description=description,
                    position_id=position_id,
                    complete_document_id=complete_document_id,
                    metadata=metadata or {},
                    request_id=rid,
                )
                info_id(f"Image added id={new_id}", rid)
                return new_id
        finally:
            if self._owns_session():
                # add_to_db may already close/commit via DB manager; but closing our session if we own it is safe.
                try:
                    sess.close()
                except Exception:
                    pass

    # ---------------- Simple search (delegates to your Image.search()) ----------------

    @with_request_id
    def search(
        self,
        *,
        search_text: Optional[str] = None,
        fields: Optional[List[str]] = None,
        image_id: Optional[int] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        position_id: Optional[int] = None,
        tool_id: Optional[int] = None,
        complete_document_id: Optional[int] = None,
        exact_match: bool = False,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "id",
        sort_order: str = "asc",
        request_id: Optional[str] = None,
    ) -> List[Image]:
        """
        Compatibility wrapper that calls Image.search(), which defers to your enhanced search_images() internally.
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        debug_id(f"[ImageRepository.search] title={title}, desc?={bool(description)}, pos={position_id}, tool={tool_id}, cdoc={complete_document_id}", rid)
        try:
            with log_timed_operation("ImageRepository.search", rid):
                return Image.search(
                    search_text=search_text,
                    fields=fields,
                    image_id=image_id,
                    title=title,
                    description=description,
                    file_path=file_path,
                    position_id=position_id,
                    tool_id=tool_id,
                    complete_document_id=complete_document_id,
                    exact_match=exact_match,
                    limit=limit,
                    offset=offset,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    request_id=rid,
                    session=sess,
                )
        finally:
            if self._owns_session():
                sess.close()

    # ---------------- Advanced search (delegates to Image.search_images()) ----------------

    @with_request_id
    def search_images(
        self,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        position_id: Optional[int] = None,
        tool_id: Optional[int] = None,
        task_id: Optional[int] = None,
        problem_id: Optional[int] = None,
        completed_document_id: Optional[int] = None,
        # position hierarchy
        area_id: Optional[int] = None,
        equipment_group_id: Optional[int] = None,
        model_id: Optional[int] = None,
        asset_number_id: Optional[int] = None,
        location_id: Optional[int] = None,
        subassembly_id: Optional[int] = None,
        component_assembly_id: Optional[int] = None,
        assembly_view_id: Optional[int] = None,
        site_location_id: Optional[int] = None,
        # vector search
        similarity_query_embedding: Optional[List[float]] = None,
        similarity_threshold: float = 0.7,
        embedding_model_name: str = "CLIPModelHandler",
        use_hybrid_ranking: bool = True,
        enable_similarity_boost: bool = True,
        limit: int = 50,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Direct wrapper over your Image.search_images() (the big, feature-rich search).
        Returns the rich dicts produced by that method.
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.search_images", rid):
                return Image.search_images(
                    session=sess,
                    title=title,
                    description=description,
                    position_id=position_id,
                    tool_id=tool_id,
                    task_id=task_id,
                    problem_id=problem_id,
                    completed_document_id=completed_document_id,
                    area_id=area_id,
                    equipment_group_id=equipment_group_id,
                    model_id=model_id,
                    asset_number_id=asset_number_id,
                    location_id=location_id,
                    subassembly_id=subassembly_id,
                    component_assembly_id=component_assembly_id,
                    assembly_view_id=assembly_view_id,
                    site_location_id=site_location_id,
                    similarity_query_embedding=similarity_query_embedding,
                    similarity_threshold=similarity_threshold,
                    embedding_model_name=embedding_model_name,
                    use_hybrid_ranking=use_hybrid_ranking,
                    enable_similarity_boost=enable_similarity_boost,
                    limit=limit,
                    request_id=rid,
                )
        finally:
            if self._owns_session():
                sess.close()

    # ---------------- Common association-driven lookups ----------------

    @with_request_id
    def images_by_position_ids(
        self, position_ids: Sequence[int], limit: int = 200, request_id: Optional[str] = None
    ) -> List[Image]:
        """
        Return distinct images directly associated to any of the given positions (ImagePositionAssociation).
        """
        rid = request_id or get_request_id()
        if not position_ids:
            return []
        if ImagePositionAssociation is None:
            raise RuntimeError("ImagePositionAssociation is not importable.")

        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.images_by_position_ids", rid):
                q = (
                    sess.query(Image)
                    .join(ImagePositionAssociation, ImagePositionAssociation.image_id == Image.id)
                    .filter(ImagePositionAssociation.position_id.in_(list(position_ids)))
                    .distinct(Image.id)
                    .limit(limit)
                )
                rows = q.all()
                debug_id(f"images_by_position_ids -> {len(rows)}", rid)
                return rows
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def images_by_complete_document(
        self, complete_document_id: int, limit: int = 200, request_id: Optional[str] = None
    ) -> List[Image]:
        """
        Images associated to a specific CompleteDocument via ImageCompletedDocumentAssociation.
        """
        rid = request_id or get_request_id()
        if ImageCompletedDocumentAssociation is None:
            raise RuntimeError("ImageCompletedDocumentAssociation is not importable.")
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.images_by_complete_document", rid):
                q = (
                    sess.query(Image)
                    .join(ImageCompletedDocumentAssociation, ImageCompletedDocumentAssociation.image_id == Image.id)
                    .filter(ImageCompletedDocumentAssociation.complete_document_id == complete_document_id)
                    .distinct(Image.id)
                    .limit(limit)
                )
                return q.all()
        finally:
            if self._owns_session():
                sess.close()

    # ---------------- Serving files ----------------

    @with_request_id
    def serve_file(self, image_id: int, request_id: Optional[str] = None):
        """
        Wrap Image.serve_file (returns (ok: bool, response_or_message, status_code)).
        """
        rid = request_id or get_request_id()
        debug_id(f"[ImageRepository.serve_file] image_id={image_id}", rid)
        with log_timed_operation("ImageRepository.serve_file", rid):
            return Image.serve_file(image_id=image_id, request_id=rid)

    # ---------------- Vector similarity helpers ----------------

    @with_request_id
    def search_similar_by_embedding(
        self,
        query_embedding: List[float],
        *,
        model_name: str = "CLIPModelHandler",
        limit: int = 10,
        similarity_threshold: float = 0.7,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Wrap Image.search_similar_images_by_embedding (rich dicts).
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.search_similar_by_embedding", rid):
                return Image.search_similar_images_by_embedding(
                    session=sess,
                    query_embedding=query_embedding,
                    model_name=model_name,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                    request_id=rid,
                )
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def find_similar_to_image(
        self,
        image_id: int,
        *,
        model_name: str = "CLIPModelHandler",
        limit: int = 10,
        similarity_threshold: float = 0.7,  # kept for parity even if underlying method ignores it
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Wrap Image.find_similar_images (rich dicts).
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.find_similar_to_image", rid):
                return Image.find_similar_images(
                    session=sess,
                    reference_image_id=image_id,
                    model_name=model_name,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                    request_id=rid,
                )
        finally:
            if self._owns_session():
                sess.close()

    # ---------------- Chunk-context + analytics wrappers ----------------

    @with_request_id
    def get_images_with_chunk_context(
        self, complete_document_id: int, request_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Wrap Image.get_images_with_chunk_context (structure-guided association details).
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.get_images_with_chunk_context", rid):
                return Image.get_images_with_chunk_context(
                    session=sess,
                    complete_document_id=complete_document_id,
                    request_id=rid,
                )
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def search_by_chunk_text(
        self,
        search_text: str,
        *,
        complete_document_id: Optional[int] = None,
        confidence_threshold: float = 0.5,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Wrap Image.search_by_chunk_text (search images via associated chunk text).
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.search_by_chunk_text", rid):
                return Image.search_by_chunk_text(
                    session=sess,
                    search_text=search_text,
                    complete_document_id=complete_document_id,
                    confidence_threshold=confidence_threshold,
                    request_id=rid,
                )
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def get_association_statistics(
        self, *, complete_document_id: Optional[int] = None, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Wrap Image.get_association_statistics (overview of structure-guided associations).
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.get_association_statistics", rid):
                return Image.get_association_statistics(
                    session=sess,
                    complete_document_id=complete_document_id,
                    request_id=rid,
                )
        finally:
            if self._owns_session():
                sess.close()

    # ---------------- Embeddings: migration, indexes, stats ----------------

    @with_request_id
    def migrate_all_embeddings_to_pgvector(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Bulk-migrate all legacy embeddings to pgvector (wraps Image.migrate_all_embeddings_to_pgvector).
        """
        if ImageEmbedding is None:
            raise RuntimeError("ImageEmbedding model not importable.")
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.migrate_all_embeddings_to_pgvector", rid):
                return Image.migrate_all_embeddings_to_pgvector(session=sess, request_id=rid)
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def setup_pgvector_indexes(self, request_id: Optional[str] = None) -> bool:
        """
        Create pgvector indexes for image embeddings (wraps Image.setup_pgvector_indexes).
        """
        if ImageEmbedding is None:
            raise RuntimeError("ImageEmbedding model not importable.")
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.setup_pgvector_indexes", rid):
                return bool(Image.setup_pgvector_indexes(session=sess, request_id=rid))
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def get_embedding_statistics(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Summary stats for embeddings (wraps Image.get_embedding_statistics).
        """
        if ImageEmbedding is None:
            raise RuntimeError("ImageEmbedding model not importable.")
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("ImageRepository.get_embedding_statistics", rid):
                return Image.get_embedding_statistics(session=sess, request_id=rid)
        finally:
            if self._owns_session():
                sess.close()
