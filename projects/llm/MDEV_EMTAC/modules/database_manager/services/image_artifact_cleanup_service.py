from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from modules.configuration.config import DATABASE_DIR, DATABASE_PATH_IMAGES_FOLDER
from modules.emtacdb.emtacdb_fts import (
    Image,
    ImageCompletedDocumentAssociation,
    ImageEmbedding,
    ImagePositionAssociation,
    ImageProblemAssociation,
    ImageTaskAssociation,
    PartsPositionImageAssociation,
    ToolImageAssociation,
)


try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


@dataclass
class ImageArtifactCleanupPolicy:
    """
    Conservative cleanup policy.

    Defaults are intentionally safe. The cleanup will only select images that are:
      1. too small, missing, or likely irrelevant
      2. not protected by meaningful image associations

    Complete-document association is treated as an origin/reference by default,
    not as a protected semantic label.

    Why:
      Extracted PDF image artifacts often still have complete-document associations.
      If complete-document associations were protected by default, many bad extracted
      artifacts would never be eligible for cleanup.

    Set protect_document_images=True if you want complete-document associations
    to protect images from cleanup.
    """

    min_file_bytes: int = 5_000
    min_width: int = 80
    min_height: int = 80
    min_area: int = 10_000

    include_missing_files: bool = False

    allow_delete_protected: bool = False
    protect_document_images: bool = False

    blank_title_is_irrelevant: bool = True
    blank_description_is_irrelevant: bool = True

    irrelevant_title_patterns: List[str] = field(
        default_factory=lambda: [
            r"^image$",
            r"^img$",
            r"^artifact$",
            r"^page\s*image$",
            r"^pasted\s*image$",
            r"^screenshot$",
            r"^unknown$",
            r"^untitled$",
            r"^blank$",
            r"^logo$",
            r"^icon$",
            r"^spacer$",
            r"^thumbnail$",
        ]
    )

    irrelevant_metadata_keys: List[str] = field(
        default_factory=lambda: [
            "artifact",
            "irrelevant",
            "too_small",
            "blank",
            "decorative",
            "icon",
            "logo",
            "thumbnail",
            "spacer",
        ]
    )


@dataclass
class ImageArtifactCandidate:
    image_id: int
    title: Optional[str]
    description: Optional[str]
    db_file_path: Optional[str]
    resolved_file_path: Optional[str]

    file_exists: bool
    file_size_bytes: Optional[int]
    width: Optional[int]
    height: Optional[int]

    reason_codes: List[str]
    association_counts: Dict[str, int]
    protected_association_count: int

    would_delete: bool

    @property
    def absolute_file_path(self) -> Optional[str]:
        """
        Compatibility alias for orchestrators that use absolute_file_path.
        """
        return self.resolved_file_path


class ImageArtifactCleanupService:
    """
    Service layer for image artifact cleanup.

    Rules:
      - Does NOT open sessions
      - Does NOT commit
      - Does NOT rollback
      - Does NOT own transaction boundaries

    The orchestrator supplies the SQLAlchemy session.
    """

    PROTECTED_ASSOCIATION_MODELS: Tuple[str, ...] = (
        "ImagePositionAssociation",
        "ToolImageAssociation",
        "ImageTaskAssociation",
        "ImageProblemAssociation",
        "PartsPositionImageAssociation",
    )

    DOCUMENT_ASSOCIATION_MODELS: Tuple[str, ...] = (
        "ImageCompletedDocumentAssociation",
    )

    ALL_KNOWN_DEPENDENT_MODELS: Tuple[str, ...] = (
        "ImageEmbedding",
        "ImagePositionAssociation",
        "ImageCompletedDocumentAssociation",
        "ToolImageAssociation",
        "ImageTaskAssociation",
        "ImageProblemAssociation",
        "PartsPositionImageAssociation",
    )

    def __init__(
        self,
        *,
        database_dir: Optional[str] = None,
        image_root: Optional[str] = None,
    ) -> None:
        self.database_dir = Path(database_dir or DATABASE_DIR)
        self.image_root = Path(image_root or DATABASE_PATH_IMAGES_FOLDER)

    # ------------------------------------------------------------
    # Public read/query methods
    # ------------------------------------------------------------

    def get_next_images(
        self,
        session: Session,
        *,
        last_image_id: int,
        batch_size: int,
    ) -> List[Image]:
        return (
            session.query(Image)
            .filter(Image.id > last_image_id)
            .order_by(Image.id.asc())
            .limit(batch_size)
            .all()
        )

    def analyze_image(
        self,
        session: Session,
        *,
        image_row: Image,
        policy: ImageArtifactCleanupPolicy,
    ) -> Optional[ImageArtifactCandidate]:
        image_id = int(image_row.id)

        title = self._safe_str(image_row.title)
        description = self._safe_str(image_row.description)
        db_file_path = self._safe_str(image_row.file_path)

        resolved_file_path = self.resolve_image_path(db_file_path)
        file_exists = bool(resolved_file_path and Path(resolved_file_path).exists())

        file_size_bytes: Optional[int] = None
        width: Optional[int] = None
        height: Optional[int] = None

        if file_exists and resolved_file_path:
            file_size_bytes = self.get_file_size_bytes(resolved_file_path)
            width, height = self.get_image_dimensions(resolved_file_path)

        association_counts = self.get_association_counts(
            session,
            image_id=image_id,
            include_document_as_protected=policy.protect_document_images,
        )

        protected_association_count = self.get_protected_association_count(
            association_counts=association_counts,
            include_document_as_protected=policy.protect_document_images,
        )

        reason_codes: List[str] = []

        if not file_exists:
            reason_codes.append("missing_file")
        else:
            if file_size_bytes is not None and file_size_bytes < policy.min_file_bytes:
                reason_codes.append("file_too_small")

            if width is not None and width < policy.min_width:
                reason_codes.append("width_too_small")

            if height is not None and height < policy.min_height:
                reason_codes.append("height_too_small")

            if width is not None and height is not None:
                area = width * height
                if area < policy.min_area:
                    reason_codes.append("area_too_small")

        if self.is_probably_irrelevant_metadata(
            image_row=image_row,
            title=title,
            description=description,
            policy=policy,
        ):
            reason_codes.append("probably_irrelevant_metadata")

        if not reason_codes:
            return None

        if "missing_file" in reason_codes and not policy.include_missing_files:
            return None

        if protected_association_count > 0 and not policy.allow_delete_protected:
            return None

        return ImageArtifactCandidate(
            image_id=image_id,
            title=title,
            description=description,
            db_file_path=db_file_path,
            resolved_file_path=resolved_file_path,
            file_exists=file_exists,
            file_size_bytes=file_size_bytes,
            width=width,
            height=height,
            reason_codes=reason_codes,
            association_counts=association_counts,
            protected_association_count=protected_association_count,
            would_delete=True,
        )

    # ------------------------------------------------------------
    # Public delete methods
    # ------------------------------------------------------------

    def delete_image_database_rows(
        self,
        session: Session,
        *,
        image_id: int,
    ) -> bool:
        """
        Deletes dependent rows first, then the image row.

        This method name is kept for compatibility with the cleanup orchestrator.
        """
        return self.delete_image_graph_for_cleanup(
            session,
            image_id=image_id,
        )

    def delete_image_graph_for_cleanup(
        self,
        session: Session,
        *,
        image_id: int,
    ) -> bool:
        """
        Deletes cleanup candidate from the database.

        Important:
          - Deletes dependent rows first
          - Deletes Image row last
          - Does NOT delete physical files
          - Does NOT commit
          - Does NOT rollback

        Physical file quarantine/delete should happen in the orchestrator
        after the DB transaction commits successfully.
        """

        image_row = self.get_image_by_id(
            session,
            image_id=image_id,
        )

        if image_row is None:
            return False

        session.query(ImageEmbedding).filter(
            ImageEmbedding.image_id == image_id
        ).delete(synchronize_session=False)

        session.query(ImagePositionAssociation).filter(
            ImagePositionAssociation.image_id == image_id
        ).delete(synchronize_session=False)

        session.query(ImageCompletedDocumentAssociation).filter(
            ImageCompletedDocumentAssociation.image_id == image_id
        ).delete(synchronize_session=False)

        session.query(ToolImageAssociation).filter(
            ToolImageAssociation.image_id == image_id
        ).delete(synchronize_session=False)

        session.query(ImageTaskAssociation).filter(
            ImageTaskAssociation.image_id == image_id
        ).delete(synchronize_session=False)

        session.query(ImageProblemAssociation).filter(
            ImageProblemAssociation.image_id == image_id
        ).delete(synchronize_session=False)

        session.query(PartsPositionImageAssociation).filter(
            PartsPositionImageAssociation.image_id == image_id
        ).delete(synchronize_session=False)

        session.delete(image_row)
        session.flush()

        return True

    def get_image_by_id(
        self,
        session: Session,
        *,
        image_id: int,
    ) -> Optional[Image]:
        if hasattr(session, "get"):
            return session.get(Image, image_id)

        return (
            session.query(Image)
            .filter(Image.id == image_id)
            .first()
        )

    # ------------------------------------------------------------
    # Association helpers
    # ------------------------------------------------------------

    def get_association_counts(
        self,
        session: Session,
        *,
        image_id: int,
        include_document_as_protected: bool = False,
    ) -> Dict[str, int]:
        """
        Returns association counts using class-name keys.

        Class-name keys are used so the cleanup report matches the ORM layer
        and remains clear during troubleshooting.
        """

        counts: Dict[str, int] = {
            "ImageEmbedding": self._count_image_embeddings(
                session,
                image_id=image_id,
            ),
            "ImagePositionAssociation": self._count_image_position_associations(
                session,
                image_id=image_id,
            ),
            "ImageCompletedDocumentAssociation": self._count_image_completed_document_associations(
                session,
                image_id=image_id,
            ),
            "ToolImageAssociation": self._count_tool_image_associations(
                session,
                image_id=image_id,
            ),
            "ImageTaskAssociation": self._count_image_task_associations(
                session,
                image_id=image_id,
            ),
            "ImageProblemAssociation": self._count_image_problem_associations(
                session,
                image_id=image_id,
            ),
            "PartsPositionImageAssociation": self._count_parts_position_image_associations(
                session,
                image_id=image_id,
            ),
        }

        _ = include_document_as_protected
        return counts

    def get_protected_association_count(
        self,
        *,
        association_counts: Dict[str, int],
        include_document_as_protected: bool = False,
    ) -> int:
        protected_names = set(self.PROTECTED_ASSOCIATION_MODELS)

        if include_document_as_protected:
            protected_names.update(self.DOCUMENT_ASSOCIATION_MODELS)

        return sum(
            int(count or 0)
            for model_name, count in association_counts.items()
            if model_name in protected_names
        )

    @staticmethod
    def _count_image_embeddings(
        session: Session,
        *,
        image_id: int,
    ) -> int:
        return (
            session.query(ImageEmbedding)
            .filter(ImageEmbedding.image_id == image_id)
            .count()
        )

    @staticmethod
    def _count_image_position_associations(
        session: Session,
        *,
        image_id: int,
    ) -> int:
        return (
            session.query(ImagePositionAssociation)
            .filter(ImagePositionAssociation.image_id == image_id)
            .count()
        )

    @staticmethod
    def _count_image_completed_document_associations(
        session: Session,
        *,
        image_id: int,
    ) -> int:
        return (
            session.query(ImageCompletedDocumentAssociation)
            .filter(ImageCompletedDocumentAssociation.image_id == image_id)
            .count()
        )

    @staticmethod
    def _count_tool_image_associations(
        session: Session,
        *,
        image_id: int,
    ) -> int:
        return (
            session.query(ToolImageAssociation)
            .filter(ToolImageAssociation.image_id == image_id)
            .count()
        )

    @staticmethod
    def _count_image_task_associations(
        session: Session,
        *,
        image_id: int,
    ) -> int:
        return (
            session.query(ImageTaskAssociation)
            .filter(ImageTaskAssociation.image_id == image_id)
            .count()
        )

    @staticmethod
    def _count_image_problem_associations(
        session: Session,
        *,
        image_id: int,
    ) -> int:
        return (
            session.query(ImageProblemAssociation)
            .filter(ImageProblemAssociation.image_id == image_id)
            .count()
        )

    @staticmethod
    def _count_parts_position_image_associations(
        session: Session,
        *,
        image_id: int,
    ) -> int:
        return (
            session.query(PartsPositionImageAssociation)
            .filter(PartsPositionImageAssociation.image_id == image_id)
            .count()
        )

    # ------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------

    def resolve_image_path(
        self,
        db_file_path: Optional[str],
    ) -> Optional[str]:
        """
        Resolve Image.file_path into an absolute path.

        Handles:
          - absolute paths
          - DB-relative paths like DB_IMAGES/example.png
          - image-root-relative paths
          - filename-only fallback
        """

        if not db_file_path:
            return None

        raw = Path(db_file_path)

        if raw.is_absolute():
            return str(raw)

        candidates = [
            self.database_dir / raw,
            self.image_root / raw,
            self.image_root / raw.name,
        ]

        normalized_candidates = [
            Path(os.path.normpath(str(path)))
            for path in candidates
        ]

        for candidate in normalized_candidates:
            if candidate.exists():
                return str(candidate)

        return str(normalized_candidates[0])

    @staticmethod
    def get_file_size_bytes(
        file_path: str,
    ) -> Optional[int]:
        try:
            return int(Path(file_path).stat().st_size)
        except Exception:
            return None

    @staticmethod
    def get_image_dimensions(
        file_path: str,
    ) -> Tuple[Optional[int], Optional[int]]:
        if PILImage is None:
            return None, None

        try:
            with PILImage.open(file_path) as img:
                width, height = img.size
                return int(width), int(height)
        except Exception:
            return None, None

    # ------------------------------------------------------------
    # Relevance helpers
    # ------------------------------------------------------------

    def is_probably_irrelevant_metadata(
        self,
        *,
        image_row: Image,
        title: Optional[str],
        description: Optional[str],
        policy: ImageArtifactCleanupPolicy,
    ) -> bool:
        title_clean = (title or "").strip()
        desc_clean = (description or "").strip()

        title_blank = not title_clean
        desc_blank = not desc_clean

        if policy.blank_title_is_irrelevant and policy.blank_description_is_irrelevant:
            if title_blank and desc_blank:
                return True

        if title_clean:
            for pattern in policy.irrelevant_title_patterns:
                if re.search(pattern, title_clean, flags=re.IGNORECASE):
                    return True

        metadata = self._extract_metadata(image_row)

        if isinstance(metadata, dict):
            flattened = json.dumps(metadata, default=str).lower()
            for key in policy.irrelevant_metadata_keys:
                if key.lower() in flattened:
                    return True

        elif isinstance(metadata, str):
            lowered = metadata.lower()
            for key in policy.irrelevant_metadata_keys:
                if key.lower() in lowered:
                    return True

        return False

    # ------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------

    @staticmethod
    def candidate_to_dict(
        candidate: ImageArtifactCandidate,
    ) -> Dict[str, Any]:
        return asdict(candidate)

    # ------------------------------------------------------------
    # Compatibility/internal helpers
    # ------------------------------------------------------------

    @staticmethod
    def _extract_metadata(
        image_row: Image,
    ) -> Any:
        value = getattr(image_row, "img_metadata", None)

        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value

        return value

    @staticmethod
    def _safe_str(
        value: Any,
    ) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip()
        return text if text else None