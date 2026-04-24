from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime
from typing import Any, Optional

import numpy as np
from PIL import Image as PILImage
from sqlalchemy import and_
from sqlalchemy.orm import Session

from modules.configuration.config import DATABASE_DIR, DATABASE_PATH_IMAGES_FOLDER
from modules.configuration.log_config import logger
from modules.configuration.model_config import ModelsConfig
from modules.emtacdb.emtacdb_fts import (
    CompleteDocument,
    CompletedDocumentPositionAssociation,
    Document,
    Image,
    ImageCompletedDocumentAssociation,
    ImageEmbedding,
    ImagePositionAssociation,
    ImageProblemAssociation,
    ImageTaskAssociation,
    PartsPositionImageAssociation,
    Position,
    ToolImageAssociation,
)
from .image_embedding_service import ImageEmbeddingService


class ImageService:
    """
    Service layer for Image.

    Rules:
    - accepts an active SQLAlchemy Session
    - does not open/close sessions
    - does not commit/rollback
    """

    # -------------------------------------------------------------------------
    # Core create / read / update / delete
    # -------------------------------------------------------------------------
    @staticmethod
    def create(
        session: Session,
        *,
        title: str,
        description: str,
        file_path: str,
        img_metadata: dict | None = None,
    ) -> Image:
        image = Image(
            title=title,
            description=description,
            file_path=file_path,
            img_metadata=img_metadata or {},
        )
        session.add(image)
        session.flush()
        return image

    @staticmethod
    def get_by_id(session: Session, image_id: int) -> Optional[Image]:
        return session.query(Image).filter(Image.id == image_id).first()

    @staticmethod
    def update(
        session: Session,
        *,
        image_id: int,
        title: str | None = None,
        description: str | None = None,
        file_path: str | None = None,
        img_metadata: dict | None = None,
    ) -> Optional[Image]:
        image = ImageService.get_by_id(session, image_id)
        if not image:
            return None

        if title is not None:
            image.title = title
        if description is not None:
            image.description = description
        if file_path is not None:
            image.file_path = file_path
        if img_metadata is not None:
            image.img_metadata = img_metadata

        session.flush()
        return image

    @staticmethod
    def delete(session: Session, *, image_id: int) -> bool:
        image = ImageService.get_by_id(session, image_id)
        if not image:
            return False
        session.delete(image)
        session.flush()
        return True

    # -------------------------------------------------------------------------
    # File staging / add-to-db style workflow
    # -------------------------------------------------------------------------
    @staticmethod
    def stage_image_file(
        *,
        title: str,
        source_file_path: str,
    ) -> tuple[str, str]:
        """
        Copy the file into DB_IMAGES and return:
        - absolute destination path
        - relative database path
        """
        if not title:
            raise ValueError("title is required")

        if not source_file_path:
            raise ValueError("source_file_path is required")

        original_filename = os.path.basename(source_file_path)
        base_name, ext = os.path.splitext(original_filename)

        safe_title = re.sub(r"[^\w\-\.]+", "_", title).strip("_")
        safe_base = re.sub(r"[^\w\-\.]+", "_", base_name).strip("_")

        destination_filename = (
            f"{safe_title}_{safe_base}{ext}"
            if safe_base else f"{safe_title}{ext}"
        )

        destination_absolute_path = os.path.join(
            DATABASE_PATH_IMAGES_FOLDER,
            destination_filename,
        )
        destination_relative_path = os.path.join(
            "DB_IMAGES",
            destination_filename,
        )

        os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)
        shutil.copy(source_file_path, destination_absolute_path)

        return destination_absolute_path, destination_relative_path

    @staticmethod
    def add_to_db(
        session: Session,
        *,
        title: str,
        file_path: str,
        description: str,
        position_id: int | None = None,
        complete_document_id: int | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        Service-layer replacement for Image.add_to_db.

        Important:
        - stages file
        - creates image row
        - optionally creates position association
        - intentionally does NOT commit/rollback
        - intentionally does NOT auto-create complete document association
        """
        if session is None:
            raise RuntimeError("Session required for ImageService.add_to_db")

        if not title:
            raise ValueError("title is required")
        if not file_path:
            raise ValueError("file_path is required")
        if not description:
            raise ValueError("description is required")

        _, destination_relative_path = ImageService.stage_image_file(
            title=title,
            source_file_path=file_path,
        )

        image = ImageService.create(
            session,
            title=title,
            description=description,
            file_path=destination_relative_path,
            img_metadata=metadata or {},
        )

        if position_id is not None:
            ImageService.ensure_position_association(
                session,
                image_id=image.id,
                position_id=position_id,
            )

        # Ownership rule preserved from your model:
        # complete_document_id is accepted, but the association is not created here
        # unless explicitly requested elsewhere.
        _ = complete_document_id

        session.flush()
        return image.id

    # -------------------------------------------------------------------------
    # Position / document / tool associations
    # -------------------------------------------------------------------------
    @staticmethod
    def ensure_position_association(
        session: Session,
        *,
        image_id: int,
        position_id: int,
    ) -> ImagePositionAssociation:
        existing = (
            session.query(ImagePositionAssociation)
            .filter(
                and_(
                    ImagePositionAssociation.image_id == image_id,
                    ImagePositionAssociation.position_id == position_id,
                )
            )
            .first()
        )

        if existing:
            return existing

        association = ImagePositionAssociation(
            image_id=image_id,
            position_id=position_id,
        )
        session.add(association)
        session.flush()
        return association

    @staticmethod
    def create_enhanced_document_association(
        session: Session,
        *,
        image_id: int,
        complete_document_id: int,
        metadata: dict | None = None,
    ) -> ImageCompletedDocumentAssociation:
        existing_association = (
            session.query(ImageCompletedDocumentAssociation)
            .filter(
                and_(
                    ImageCompletedDocumentAssociation.image_id == image_id,
                    ImageCompletedDocumentAssociation.complete_document_id == complete_document_id,
                )
            )
            .first()
        )

        if existing_association is not None:
            return existing_association

        structure_metadata = metadata or {}

        association = ImageCompletedDocumentAssociation(
            image_id=image_id,
            complete_document_id=complete_document_id,
            page_number=structure_metadata.get("page_number"),
            chunk_index=structure_metadata.get("image_index", 0),
            association_method=structure_metadata.get("association_method", "structure_guided"),
            confidence_score=structure_metadata.get("confidence_score", 0.8),
            context_metadata={
                "structure_guided": structure_metadata.get("structure_guided", True),
                "content_type": structure_metadata.get("content_type", "image"),
                "bbox": structure_metadata.get("bbox"),
                "estimated_size": structure_metadata.get("estimated_size"),
                "created_at": datetime.now().isoformat(),
                "processing_method": structure_metadata.get(
                    "processing_method",
                    "enhanced_add_to_db",
                ),
            },
        )
        session.add(association)
        session.flush()
        return association

    @staticmethod
    def create_tool_association(
        session: Session,
        *,
        image_id: int,
        tool_id: int,
        description: str = "Primary uploaded tool image",
    ) -> ToolImageAssociation:
        existing_assoc = (
            session.query(ToolImageAssociation)
            .filter(
                and_(
                    ToolImageAssociation.tool_id == tool_id,
                    ToolImageAssociation.image_id == image_id,
                )
            )
            .first()
        )

        if existing_assoc:
            return existing_assoc

        assoc = ToolImageAssociation(
            tool_id=tool_id,
            image_id=image_id,
            description=description,
        )
        session.add(assoc)
        session.flush()
        return assoc

    @staticmethod
    def create_with_tool_association(
        session: Session,
        *,
        title: str,
        file_path: str,
        tool,
        description: str = "",
        metadata: dict | None = None,
    ) -> tuple[Image, ToolImageAssociation]:
        image_id = ImageService.add_to_db(
            session,
            title=title,
            file_path=file_path,
            description=description,
            metadata=metadata,
        )
        image = ImageService.get_by_id(session, image_id)
        tool_assoc = ImageService.create_tool_association(
            session,
            image_id=image.id,
            tool_id=tool.id,
            description="Primary uploaded tool image",
        )
        return image, tool_assoc

    # -------------------------------------------------------------------------
    # Embedding workflow
    # -------------------------------------------------------------------------
    @staticmethod
    def generate_embedding(
        session: Session,
        *,
        image: Image,
        model_handler,
        prefer_pgvector: bool = True,
    ) -> bool:
        """
        Service-layer replacement for Image.generate_embedding.
        """
        if os.path.isabs(image.file_path):
            absolute_file_path = image.file_path
        else:
            absolute_file_path = os.path.join(DATABASE_DIR, image.file_path)

        pil_image = PILImage.open(absolute_file_path).convert("RGB")

        if not model_handler.is_valid_image(pil_image):
            return False

        model_embedding = model_handler.get_image_embedding(pil_image)
        model_name = model_handler.__class__.__name__

        if model_embedding is None:
            return False

        if hasattr(model_embedding, "tolist"):
            embedding_list = model_embedding.tolist()
        elif isinstance(model_embedding, np.ndarray):
            embedding_list = model_embedding.flatten().tolist()
        else:
            embedding_list = list(model_embedding)

        existing_embedding = ImageEmbeddingService.get_by_image_and_model(
            session,
            image_id=image.id,
            model_name=model_name,
        )

        if existing_embedding is None:
            ImageEmbeddingService.create_or_update_for_image(
                session,
                image_id=image.id,
                model_name=model_name,
                embedding=embedding_list,
                prefer_pgvector=prefer_pgvector,
            )
        else:
            if existing_embedding.get_storage_type() == "legacy" and prefer_pgvector:
                ImageEmbeddingService.migrate_to_pgvector(
                    session,
                    embedding_row=existing_embedding,
                )

        session.flush()
        return True

    # -------------------------------------------------------------------------
    # Search helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def search_by_chunk_text(
        session: Session,
        *,
        search_text: str,
        complete_document_id: int | None = None,
        confidence_threshold: float = 0.5,
    ) -> list[dict]:
        query = (
            session.query(
                Image.id.label("image_id"),
                Image.title.label("image_title"),
                Image.file_path.label("image_path"),
                Image.description.label("image_description"),
                Document.content.label("chunk_content"),
                Document.name.label("chunk_name"),
                CompleteDocument.title.label("document_title"),
                ImageCompletedDocumentAssociation.confidence_score,
                ImageCompletedDocumentAssociation.page_number,
                ImageCompletedDocumentAssociation.association_method,
            )
            .select_from(Image)
            .join(
                ImageCompletedDocumentAssociation,
                Image.id == ImageCompletedDocumentAssociation.image_id,
            )
            .join(
                Document,
                ImageCompletedDocumentAssociation.document_id == Document.id,
            )
            .join(
                CompleteDocument,
                ImageCompletedDocumentAssociation.complete_document_id == CompleteDocument.id,
            )
            .filter(Document.content.ilike(f"%{search_text}%"))
        )

        if complete_document_id:
            query = query.filter(CompleteDocument.id == complete_document_id)

        if confidence_threshold:
            query = query.filter(
                ImageCompletedDocumentAssociation.confidence_score >= confidence_threshold
            )

        query = query.order_by(
            ImageCompletedDocumentAssociation.confidence_score.desc(),
            ImageCompletedDocumentAssociation.page_number,
        )

        results = []
        for row in query.all():
            results.append(
                {
                    "image_id": row.image_id,
                    "image_title": row.image_title,
                    "image_path": row.image_path,
                    "image_description": row.image_description,
                    "chunk_content": row.chunk_content,
                    "chunk_name": row.chunk_name,
                    "document_title": row.document_title,
                    "confidence": row.confidence_score,
                    "page_number": row.page_number,
                    "association_method": row.association_method,
                    "highlighted_content": ImageService.highlight_search_term(
                        row.chunk_content,
                        search_text,
                    ),
                    "view_url": f"/add_document/image/{row.image_id}",
                }
            )

        return results

    @staticmethod
    def highlight_search_term(content: str | None, search_term: str | None) -> str | None:
        if not content or not search_term:
            return content

        import re

        pattern = re.compile(re.escape(search_term), re.IGNORECASE)
        return pattern.sub(f"<mark>{search_term}</mark>", content)

    @staticmethod
    def search(
        session: Session,
        *,
        title: str | None = None,
        description: str | None = None,
        position_id: int | None = None,
        tool_id: int | None = None,
        task_id: int | None = None,
        problem_id: int | None = None,
        completed_document_id: int | None = None,
        area_id: int | None = None,
        equipment_group_id: int | None = None,
        model_id: int | None = None,
        asset_number_id: int | None = None,
        location_id: int | None = None,
        subassembly_id: int | None = None,
        component_assembly_id: int | None = None,
        assembly_view_id: int | None = None,
        site_location_id: int | None = None,
        similarity_query_embedding: list[float] | None = None,
        similarity_threshold: float = 0.7,
        embedding_model_name: str = "CLIPModelHandler",
        use_hybrid_ranking: bool = True,
        limit: int = 50,
    ) -> list[dict]:
        """
        Service-layer wrapper around the existing model search_images implementation.
        This preserves your current hybrid search behavior while moving workflow
        ownership to the service layer.
        """
        return Image.search_images(
            session=session,
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
            limit=limit,
        )

    @staticmethod
    def search_simple(
        session: Session,
        *,
        search_text: str | None = None,
        title: str | None = None,
        description: str | None = None,
        position_id: int | None = None,
        tool_id: int | None = None,
        complete_document_id: int | None = None,
        limit: int = 20,
    ) -> list[Image]:
        results = Image.search(
            search_text=search_text,
            title=title,
            description=description,
            position_id=position_id,
            tool_id=tool_id,
            complete_document_id=complete_document_id,
            limit=limit,
            session=session,
        )
        return results

    @staticmethod
    def search_similar_images_by_embedding(
        session: Session,
        *,
        query_embedding: list[float],
        model_name: str = "CLIPModelHandler",
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        similar_images = ImageEmbeddingService.search_similar_images(
            session,
            query_embedding=query_embedding,
            model_name=model_name,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

        enhanced_results = []
        for result in similar_images:
            image = ImageService.get_by_id(session, result["image_id"])
            if image:
                enhanced_results.append(
                    {
                        **result,
                        "image": {
                            "id": image.id,
                            "title": image.title,
                            "description": image.description,
                            "file_path": image.file_path,
                            "metadata": image.img_metadata,
                            "view_url": f"/add_document/image/{image.id}",
                        },
                    }
                )
        return enhanced_results

    @staticmethod
    def find_similar_images(
        session: Session,
        *,
        reference_image_id: int,
        model_name: str = "CLIPModelHandler",
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        similar_images = ImageEmbeddingService.find_similar_to_image(
            session,
            image_id=reference_image_id,
            model_name=model_name,
            limit=limit,
            exclude_self=True,
        )

        enhanced_results = []
        for result in similar_images:
            image = ImageService.get_by_id(session, result["image_id"])
            if image:
                enhanced_results.append(
                    {
                        **result,
                        "image": {
                            "id": image.id,
                            "title": image.title,
                            "description": image.description,
                            "file_path": image.file_path,
                            "metadata": image.img_metadata,
                            "view_url": f"/add_document/image/{image.id}",
                        },
                    }
                )
        return enhanced_results

    # -------------------------------------------------------------------------
    # Association / metadata helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def get_enhanced_image_associations(
        session: Session,
        *,
        image_id: int,
    ) -> dict:
        associations = {}

        position_assocs = (
            session.query(ImagePositionAssociation)
            .filter(ImagePositionAssociation.image_id == image_id)
            .all()
        )
        associations["positions"] = [
            {"position_id": assoc.position_id}
            for assoc in position_assocs
        ]

        tool_assocs = (
            session.query(ToolImageAssociation)
            .filter(ToolImageAssociation.image_id == image_id)
            .all()
        )
        associations["tools"] = [
            {
                "tool_id": assoc.tool_id,
                "description": assoc.description,
            }
            for assoc in tool_assocs
        ]

        task_assocs = (
            session.query(ImageTaskAssociation)
            .filter(ImageTaskAssociation.image_id == image_id)
            .all()
        )
        associations["tasks"] = [
            {"task_id": assoc.task_id}
            for assoc in task_assocs
        ]

        problem_assocs = (
            session.query(ImageProblemAssociation)
            .filter(ImageProblemAssociation.image_id == image_id)
            .all()
        )
        associations["problems"] = [
            {"problem_id": assoc.problem_id}
            for assoc in problem_assocs
        ]

        doc_assocs = (
            session.query(ImageCompletedDocumentAssociation)
            .filter(ImageCompletedDocumentAssociation.image_id == image_id)
            .all()
        )

        enhanced_doc_assocs = []
        for assoc in doc_assocs:
            context_metadata = {}

            if assoc.context_metadata:
                if isinstance(assoc.context_metadata, dict):
                    context_metadata = assoc.context_metadata
                elif isinstance(assoc.context_metadata, str):
                    try:
                        context_metadata = json.loads(assoc.context_metadata)
                    except Exception:
                        context_metadata = {}
                else:
                    context_metadata = {}

            enhanced_doc_assocs.append(
                {
                    "document_id": assoc.complete_document_id,
                    "page_number": assoc.page_number,
                    "chunk_index": assoc.chunk_index,
                    "association_method": assoc.association_method,
                    "confidence_score": assoc.confidence_score,
                    "structure_guided": context_metadata.get("structure_guided", False),
                    "content_type": context_metadata.get("content_type", "image"),
                    "bbox": context_metadata.get("bbox"),
                    "estimated_size": context_metadata.get("estimated_size"),
                    "processing_method": context_metadata.get("processing_method"),
                }
            )

        associations["completed_documents"] = enhanced_doc_assocs

        parts_assocs = (
            session.query(PartsPositionImageAssociation)
            .filter(PartsPositionImageAssociation.image_id == image_id)
            .all()
        )
        associations["parts_positions"] = [
            {
                "part_id": assoc.part_id,
                "position_id": assoc.position_id,
            }
            for assoc in parts_assocs
        ]

        return associations

    # -------------------------------------------------------------------------
    # Session / commit helpers moved from model, but commit still owned by orchestrator
    # -------------------------------------------------------------------------
    @staticmethod
    def commit_with_retry(
        session: Session,
        *,
        retries: int = 3,
        delay: float = 0.5,
    ) -> bool:
        """
        Kept as a helper because you already use this pattern in image workflows.
        Orchestrators may call this instead of plain commit() when desired.
        """
        for attempt in range(retries):
            try:
                session.commit()
                return True
            except Exception as exc:
                error_msg = str(exc).lower()

                if (
                    "pending rollback" in error_msg
                    or "database is locked" in error_msg
                    or "locked" in error_msg
                    or "deadlock" in error_msg
                    or "timeout" in error_msg
                ):
                    try:
                        session.rollback()
                    except Exception:
                        pass

                    if attempt < retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 1.2, 2)
                    else:
                        raise
                else:
                    try:
                        session.rollback()
                    except Exception:
                        pass
                    raise

        return False