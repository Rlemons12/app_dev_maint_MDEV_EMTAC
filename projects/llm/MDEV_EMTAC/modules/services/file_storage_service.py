from __future__ import annotations

import mimetypes
import os
from typing import Any, Dict, Optional

from werkzeug.utils import secure_filename

from modules.configuration.config import (
    DATABASE_DIR,
    DATABASE_PATH_IMAGES_FOLDER,
    ALLOWED_EXTENSIONS,
    BASE_DIR,
)
from modules.configuration.log_config import logger, with_request_id


class FileStorageService:
    """
    Handles filesystem storage for uploaded documents and images.

    HARD RULES:
    - No DB access
    - No session handling
    - Pure filesystem logic
    """

    def __init__(self):
        self.doc_base_dir = os.path.join(DATABASE_DIR, "DB_DOC")
        self.image_base_dir = DATABASE_PATH_IMAGES_FOLDER

        os.makedirs(self.doc_base_dir, exist_ok=True)
        os.makedirs(self.image_base_dir, exist_ok=True)

    # ==========================================================
    # GENERIC / DOCUMENT STORAGE
    # ==========================================================

    @with_request_id
    def save(self, file: Any) -> str:
        """
        Save uploaded file to DB_DOC directory.
        Handles filename conflicts.
        Returns absolute file path.
        """

        filename = secure_filename(
            getattr(file, "filename", None) or str(file)
        )

        file_path = os.path.join(self.doc_base_dir, filename)

        counter = 1
        while os.path.exists(file_path):
            name, ext = os.path.splitext(filename)
            file_path = os.path.join(
                self.doc_base_dir,
                f"{name}_{counter}{ext}"
            )
            counter += 1

        if hasattr(file, "save"):
            file.save(file_path)
        elif hasattr(file, "read"):
            with open(file_path, "wb") as f:
                f.write(file.read())
        else:
            raise ValueError("Unsupported file type")

        logger.info("Saved document file to %s", file_path)
        return file_path

    # ==========================================================
    # IMAGE STORAGE FOR BOM / EDIT PART
    # ==========================================================

    @with_request_id
    def validate_image_extension(self, filename: str) -> None:
        if not filename or "." not in filename:
            raise ValueError(
                "File type not allowed. Please upload jpg, jpeg, png, or gif files only."
            )

        ext = filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                "File type not allowed. Please upload jpg, jpeg, png, or gif files only."
            )

    @with_request_id
    def save_part_image(
        self,
        *,
        uploaded_file: Any,
        part_number: str,
    ) -> Dict[str, str]:
        """
        Save part image to DB_IMAGES directory.

        Returns:
            {
                "absolute_path": ...,
                "relative_path": ...,
                "filename": ...
            }
        """

        original_filename = getattr(uploaded_file, "filename", "") or ""
        self.validate_image_extension(original_filename)

        safe_filename = secure_filename(original_filename)
        base_name, ext = os.path.splitext(safe_filename)

        safe_part_number = secure_filename(part_number or "part")
        candidate_filename = f"{safe_part_number}_{base_name}{ext}".replace(" ", "_")
        absolute_path = os.path.join(self.image_base_dir, candidate_filename)

        counter = 1
        while os.path.exists(absolute_path):
            candidate_filename = (
                f"{safe_part_number}_{base_name}_{counter}{ext}".replace(" ", "_")
            )
            absolute_path = os.path.join(self.image_base_dir, candidate_filename)
            counter += 1

        uploaded_file.save(absolute_path)

        relative_path = os.path.join("DB_IMAGES", candidate_filename)

        logger.info(
            "Saved part image to absolute_path=%s relative_path=%s",
            absolute_path,
            relative_path,
        )

        return {
            "absolute_path": absolute_path,
            "relative_path": relative_path,
            "filename": candidate_filename,
        }

    @with_request_id
    def resolve_absolute_image_path(self, stored_file_path: str) -> str:
        """
        Supports:
        - new unified path: DB_IMAGES/filename.ext
        - legacy relative path under BASE_DIR
        """
        normalized = (stored_file_path or "").replace("\\", "/")

        if normalized.startswith("DB_IMAGES/") or normalized.startswith("DB_IMAGES"):
            return os.path.join(DATABASE_DIR, normalized.replace("/", os.sep))

        return os.path.join(BASE_DIR, normalized.replace("/", os.sep))

    @with_request_id
    def delete_file_if_exists(self, absolute_path: str) -> None:
        if absolute_path and os.path.exists(absolute_path):
            logger.info("Deleting file: %s", absolute_path)
            os.remove(absolute_path)
        else:
            logger.warning("File not found for deletion: %s", absolute_path)

    @with_request_id
    def file_exists(self, absolute_path: str) -> bool:
        return bool(absolute_path and os.path.exists(absolute_path))

    @with_request_id
    def guess_mimetype(self, absolute_path: str) -> str:
        guessed, _ = mimetypes.guess_type(absolute_path)
        return guessed or "application/octet-stream"


class LocalFileAdapter:
    """
    Adapter to mimic Flask FileStorage for local file ingestion.
    """

    def __init__(self, path: str):
        self.path = path
        self.filename = os.path.basename(path)

    def read(self):
        with open(self.path, "rb") as f:
            return f.read()