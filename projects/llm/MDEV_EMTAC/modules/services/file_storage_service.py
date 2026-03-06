# modules/services/file_storage_service.py

import os
from werkzeug.utils import secure_filename
from typing import Any

from modules.configuration.config import DATABASE_DIR


class FileStorageService:
    """
    Handles filesystem storage for uploaded documents.

    HARD RULES:
    - No DB access
    - No session handling
    - Pure filesystem logic
    """

    def __init__(self):
        self.base_dir = os.path.join(DATABASE_DIR, "DB_DOC")
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, file: Any) -> str:
        """
        Save uploaded file to DB_DOC directory.
        Handles filename conflicts.
        Returns absolute file path.
        """

        filename = secure_filename(
            getattr(file, "filename", None) or str(file)
        )

        file_path = os.path.join(self.base_dir, filename)

        # Resolve name conflicts
        counter = 1
        while os.path.exists(file_path):
            name, ext = os.path.splitext(filename)
            file_path = os.path.join(
                self.base_dir,
                f"{name}_{counter}{ext}"
            )
            counter += 1

        # Save file
        if hasattr(file, "save"):
            file.save(file_path)
        elif hasattr(file, "read"):
            with open(file_path, "wb") as f:
                f.write(file.read())
        else:
            raise ValueError("Unsupported file type")

        return file_path

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