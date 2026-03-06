# modules/services/image_service.py

from __future__ import annotations

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from modules.emtacdb.emtacdb_fts import Image
from modules.configuration.log_config import (
    info_id,
    warning_id,
    debug_id,
    with_request_id,
)


class ImageService:
    """
    Pure domain service for Image.

    HARD RULES:
    - NEVER open sessions
    - NEVER close sessions
    - NEVER commit
    - NEVER rollback
    - Orchestrator owns transactions
    """

    # ---------------------------------------------------------
    # CREATE
    # ---------------------------------------------------------

    @with_request_id
    def create(
        self,
        session: Session,
        *,
        title: str,
        file_path: str,
        description: Optional[str] = None,
        img_metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Image:
        if session is None:
            raise RuntimeError("Session required for ImageService.create")

        if not title:
            raise ValueError("title is required")

        if not file_path:
            raise ValueError("file_path is required")

        image = Image(
            title=title,
            file_path=file_path,
            description=description or "",
            img_metadata=img_metadata or {},
        )

        session.add(image)
        session.flush()

        debug_id(f"[ImageService] Image staged id={image.id}, title='{title}'", request_id)
        return image

    # ---------------------------------------------------------
    # READ
    # ---------------------------------------------------------

    @with_request_id
    def get(
        self,
        session: Session,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> Optional[Image]:
        if session is None:
            raise RuntimeError("Session required for ImageService.get")
        return session.get(Image, image_id)

    @with_request_id
    def find(
        self,
        session: Session,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> List[Image]:
        if session is None:
            raise RuntimeError("Session required for ImageService.find")

        limit = max(1, min(int(limit or 100), 1000))

        query = session.query(Image)

        if title:
            query = query.filter(Image.title.ilike(f"%{title}%"))

        if description:
            query = query.filter(Image.description.ilike(f"%{description}%"))

        if file_path:
            query = query.filter(Image.file_path.ilike(f"%{file_path}%"))

        return query.limit(limit).all()

    # ---------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------

    @with_request_id
    def update(
        self,
        session: Session,
        *,
        image_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        img_metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Image]:
        if session is None:
            raise RuntimeError("Session required for ImageService.update")

        image = session.get(Image, image_id)

        if not image:
            warning_id(f"[ImageService] Image id={image_id} not found", request_id)
            return None

        if title is not None:
            image.title = title

        if description is not None:
            image.description = description

        if img_metadata is not None:
            image.img_metadata = img_metadata

        session.flush()
        debug_id(f"[ImageService] Image updated id={image_id}", request_id)
        return image

    # ---------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------

    @with_request_id
    def remove(
        self,
        session: Session,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> bool:
        if session is None:
            raise RuntimeError("Session required for ImageService.remove")

        image = session.get(Image, image_id)

        if not image:
            warning_id(f"[ImageService] Image id={image_id} not found", request_id)
            return False

        session.delete(image)
        session.flush()

        info_id(f"[ImageService] Image staged for deletion id={image_id}", request_id)
        return True

    # ---------------------------------------------------------
    # SERIALIZATION (Optional Helper)
    # ---------------------------------------------------------

    def serialize(self, image: Image) -> Dict[str, Any]:
        return {
            "id": image.id,
            "title": image.title,
            "description": image.description,
            "file_path": image.file_path,
            "img_metadata": image.img_metadata,
            "url": f"/serve_image/{image.id}",
        }