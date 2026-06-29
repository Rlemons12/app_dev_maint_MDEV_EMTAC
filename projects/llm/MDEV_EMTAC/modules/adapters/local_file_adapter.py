# modules/adapters/local_file_adapter.py

import os
import shutil
from typing import Optional


class LocalFileAdapter:
    """
    Adapter that converts a local filesystem path into an object
    compatible with the upload contract expected by FileStorageService.

    This mimics Flask's FileStorage interface minimally by providing:
        - .filename
        - .save(destination_path)

    It is intended for CLI ingestion, background jobs, or testing —
    NOT for use inside core services.
    """

    def __init__(self, path: str):
        if not path:
            raise ValueError("Path must not be empty")

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        if not os.path.isfile(path):
            raise ValueError(f"Path is not a file: {path}")

        self.path: str = path
        self.filename: str = os.path.basename(path)

    def save(self, destination_path: str) -> None:
        """
        Stream-copy the file to the destination path.

        This mirrors Flask FileStorage.save() behavior.
        """
        if not destination_path:
            raise ValueError("Destination path must not be empty")

        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(self.path, destination_path)

    def read(self) -> bytes:
        """
        Optional read method for compatibility.
        Not used by default (save() is preferred).
        """
        with open(self.path, "rb") as f:
            return f.read()

    def __repr__(self) -> str:
        return f"<LocalFileAdapter filename='{self.filename}'>"