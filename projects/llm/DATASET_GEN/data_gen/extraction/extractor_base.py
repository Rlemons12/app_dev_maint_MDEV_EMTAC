# extractor_base.py
# Base interface for all document extractors

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ExtractedPage:
    page_index: int
    text: str
    images: List[str]           # file paths of extracted images
    ocr_used: bool = False
    warnings: Optional[List[str]] = None


@dataclass
class DocumentRawContent:
    source_path: str
    file_type: str
    pages: List[ExtractedPage]
    metadata: Dict[str, Any]
    errors: Optional[List[str]] = None


class BaseExtractor:
    """
    Abstract base class for all extractors.
    Each extractor MUST implement:
        - extract()
        - _extract_text()
        - _extract_images()
    """

    def __init__(self, src_path: str, output_dir: str):
        self.src_path = src_path
        self.output_dir = output_dir

    def extract(self) -> DocumentRawContent:
        """Main dispatcher — must be implemented by subclasses."""
        raise NotImplementedError

    # Utility functions for subclasses

    def _ensure_output_dir(self):
        import os
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_image(self, image_bytes: bytes, img_name: str) -> str:
        """
        Saves raw image bytes to disk and returns full path.
        """
        self._ensure_output_dir()
        out_path = f"{self.output_dir}/{img_name}"
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        return out_path
