# modules/services/image_service.py

from __future__ import annotations

from typing import Optional, List, Dict, Any
from copy import deepcopy

from sqlalchemy.orm import Session
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm.exc import DetachedInstanceError

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

    DESIGN NOTES:
    - Methods that mutate data still return ORM instances for compatibility.
    - This service also exposes safe serialization helpers so callers can
      convert ORM rows into plain dicts before leaving the transaction boundary.
    """

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _require_session(session: Session, method_name: str) -> None:
        if session is None:
            raise RuntimeError(f"Session required for ImageService.{method_name}")

    @staticmethod
    def _normalize_text(value: Optional[str], *, field_name: str, required: bool = False) -> str:
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        normalized = value.strip()
        if required and not normalized:
            raise ValueError(f"{field_name} is required")
        return normalized

    @staticmethod
    def _normalize_metadata(img_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if img_metadata is None:
            return {}

        if not isinstance(img_metadata, dict):
            raise TypeError("img_metadata must be a dictionary")

        # Deep copy to prevent accidental caller-side mutation after staging
        return deepcopy(img_metadata)

    @staticmethod
    def _safe_attr(image: Optional[Image], attr_name: str, default: Any = None) -> Any:
        """
        Safely read an ORM attribute without forcing the caller to care whether
        the instance is attached or detached.

        Notes:
        - If the attribute is already loaded, SQLAlchemy will usually return it.
        - If the instance is detached and the attribute is expired, reading it
          may raise DetachedInstanceError. In that case we return default.
        """
        if image is None:
            return default

        try:
            return getattr(image, attr_name)
        except DetachedInstanceError:
            return default
        except Exception:
            return default

    def to_payload(self, image: Optional[Image]) -> Optional[Dict[str, Any]]:
        """
        Convert an Image ORM object to a plain dict safely.

        This is the preferred boundary object for orchestrators/controllers that
        should not depend on a live SQLAlchemy session.
        """
        if image is None:
            return None

        return {
            "id": self._safe_attr(image, "id"),
            "title": self._safe_attr(image, "title", ""),
            "description": self._safe_attr(image, "description", ""),
            "file_path": self._safe_attr(image, "file_path", ""),
            "img_metadata": self._safe_attr(image, "img_metadata", {}) or {},
            "url": f"/serve_image/{self._safe_attr(image, 'id')}" if self._safe_attr(image, "id") is not None else None,
        }

    def snapshot(self, image: Optional[Image]) -> Optional[Dict[str, Any]]:
        """
        Create a plain-data snapshot of current loaded values from an Image row.

        Best used immediately after session.flush() while the object is still
        attached, so callers can safely carry scalar data outside the ORM/session
        boundary.
        """
        if image is None:
            return None

        state = sa_inspect(image)

        data: Dict[str, Any] = {}

        # Pull loaded values when available without triggering lazy refreshes.
        for attr_name in ("id", "title", "description", "file_path", "img_metadata"):
            try:
                attr_state = state.attrs[attr_name]
                if attr_state.loaded_value is not None:
                    try:
                        data[attr_name] = getattr(image, attr_name)
                    except DetachedInstanceError:
                        data[attr_name] = None
                else:
                    data[attr_name] = self._safe_attr(image, attr_name)
            except Exception:
                data[attr_name] = self._safe_attr(image, attr_name)

        image_id = data.get("id")
        data["url"] = f"/serve_image/{image_id}" if image_id is not None else None
        return data

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
        self._require_session(session, "create")

        normalized_title = self._normalize_text(title, field_name="title", required=True)
        normalized_file_path = self._normalize_text(file_path, field_name="file_path", required=True)
        normalized_description = self._normalize_text(description, field_name="description", required=False)
        normalized_metadata = self._normalize_metadata(img_metadata)

        image = Image(
            title=normalized_title,
            file_path=normalized_file_path,
            description=normalized_description,
            img_metadata=normalized_metadata,
        )

        session.add(image)
        session.flush()

        debug_id(
            f"[ImageService] Image staged id={image.id}, title='{normalized_title}'",
            request_id,
        )
        return image

    @with_request_id
    def create_payload(
        self,
        session: Session,
        *,
        title: str,
        file_path: str,
        description: Optional[str] = None,
        img_metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience helper for callers that want plain data instead of an ORM row.
        """
        image = self.create(
            session,
            title=title,
            file_path=file_path,
            description=description,
            img_metadata=img_metadata,
            request_id=request_id,
        )
        return self.snapshot(image) or {}

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
        self._require_session(session, "get")

        if image_id is None:
            raise ValueError("image_id is required")

        return session.get(Image, image_id)

    @with_request_id
    def get_payload(
        self,
        session: Session,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        image = self.get(
            session,
            image_id=image_id,
            request_id=request_id,
        )
        return self.to_payload(image)

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
        self._require_session(session, "find")

        limit = max(1, min(int(limit or 100), 1000))

        query = session.query(Image)

        normalized_title = self._normalize_text(title, field_name="title", required=False) if title is not None else ""
        normalized_description = (
            self._normalize_text(description, field_name="description", required=False)
            if description is not None else ""
        )
        normalized_file_path = (
            self._normalize_text(file_path, field_name="file_path", required=False)
            if file_path is not None else ""
        )

        if normalized_title:
            query = query.filter(Image.title.ilike(f"%{normalized_title}%"))

        if normalized_description:
            query = query.filter(Image.description.ilike(f"%{normalized_description}%"))

        if normalized_file_path:
            query = query.filter(Image.file_path.ilike(f"%{normalized_file_path}%"))

        results = query.limit(limit).all()

        debug_id(
            f"[ImageService] find returned {len(results)} image(s) | "
            f"title='{normalized_title}' | file_path='{normalized_file_path}' | limit={limit}",
            request_id,
        )
        return results

    @with_request_id
    def find_payloads(
        self,
        session: Session,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        images = self.find(
            session,
            title=title,
            description=description,
            file_path=file_path,
            limit=limit,
            request_id=request_id,
        )
        return [self.to_payload(image) for image in images if image is not None]

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
        self._require_session(session, "update")

        if image_id is None:
            raise ValueError("image_id is required")

        image = session.get(Image, image_id)

        if not image:
            warning_id(f"[ImageService] Image id={image_id} not found", request_id)
            return None

        if title is not None:
            image.title = self._normalize_text(title, field_name="title", required=True)

        if description is not None:
            image.description = self._normalize_text(description, field_name="description", required=False)

        if img_metadata is not None:
            image.img_metadata = self._normalize_metadata(img_metadata)

        session.flush()

        debug_id(f"[ImageService] Image updated id={image_id}", request_id)
        return image

    @with_request_id
    def update_payload(
        self,
        session: Session,
        *,
        image_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        img_metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        image = self.update(
            session,
            image_id=image_id,
            title=title,
            description=description,
            img_metadata=img_metadata,
            request_id=request_id,
        )
        return self.snapshot(image)

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
        self._require_session(session, "remove")

        if image_id is None:
            raise ValueError("image_id is required")

        image = session.get(Image, image_id)

        if not image:
            warning_id(f"[ImageService] Image id={image_id} not found", request_id)
            return False

        session.delete(image)
        session.flush()

        info_id(f"[ImageService] Image staged for deletion id={image_id}", request_id)
        return True

    # ---------------------------------------------------------
    # SERIALIZATION
    # ---------------------------------------------------------

    def serialize(self, image: Image) -> Dict[str, Any]:
        """
        Backward-compatible serializer name.
        """
        payload = self.to_payload(image)
        return payload or {
            "id": None,
            "title": "",
            "description": "",
            "file_path": "",
            "img_metadata": {},
            "url": None,
        }