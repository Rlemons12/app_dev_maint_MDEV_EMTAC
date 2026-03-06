from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv


class GPUServiceConfig:
    """
    Central configuration for GPU Service.

    Responsibilities
    ----------------
    • Load local .env
    • Artifact storage paths
    • Runtime tuning knobs
    • Directory creation
    """

    def __init__(self):

        # ---------------------------------------------------------
        # Load .env
        # ---------------------------------------------------------
        self._load_local_env()

        # ---------------------------------------------------------
        # Artifact Root
        # ---------------------------------------------------------
        root = os.getenv(
            "GPU_SERVICE_ARTIFACT_ROOT",
            r"E:\emtac\services\gpu\artifacts",
        )

        self.artifact_root = Path(root)

        # ---------------------------------------------------------
        # Artifact Subdirectories
        # ---------------------------------------------------------
        self.pdf_dir = self.artifact_root / "pdf"
        self.image_dir = self.artifact_root / "images"
        self.markdown_dir = self.artifact_root / "markdown"
        self.json_dir = self.artifact_root / "json"

        # ---------------------------------------------------------
        # VLM Runtime Settings
        # ---------------------------------------------------------

        # pages per inference batch
        self.vlm_batch_pages = int(os.getenv("VLM_BATCH_PAGES", "3"))

        # maximum tokens generated per page
        self.vlm_max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "512"))

        # ---------------------------------------------------------
        # PDF PIPELINE PERFORMANCE SETTINGS
        # ---------------------------------------------------------

        # render DPI for PDF pages
        self.pdf_render_dpi = int(os.getenv("PDF_RENDER_DPI", "144"))

        # CPU workers used for page rendering
        self.pdf_cpu_workers = int(os.getenv("PDF_CPU_WORKERS", "4"))

        # number of windows prepared ahead of GPU
        self.pdf_prefetch_windows = int(os.getenv("PDF_PREFETCH_WINDOWS", "2"))

        # pages per GPU window
        self.pdf_window_pages = int(os.getenv("PDF_WINDOW_PAGES", "5"))

        # ---------------------------------------------------------
        # PDF PIPELINE SETTINGS
        # ---------------------------------------------------------

        self.pdf_save_rendered_images = (
            os.getenv("PDF_SAVE_RENDERED_IMAGES", "true").lower() == "true"
        )

        # ---------------------------------------------------------
        # Ensure artifact directories exist
        # ---------------------------------------------------------
        self._ensure_directories()

    # ---------------------------------------------------------
    # ENV Loader
    # ---------------------------------------------------------

    def _load_local_env(self) -> None:
        """
        Loads .env from GPU service root.
        Safe to call multiple times.
        """

        base_dir = Path(__file__).resolve().parents[2]
        env_path = base_dir / ".env"

        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)

    # ---------------------------------------------------------
    # Directory Creation
    # ---------------------------------------------------------

    def _ensure_directories(self) -> None:

        for d in (
            self.artifact_root,
            self.pdf_dir,
            self.image_dir,
            self.markdown_dir,
            self.json_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Path Builders
    # ---------------------------------------------------------

    def build_pdf_path(self, request_id: str) -> Path:
        return self.pdf_dir / f"{request_id}_source.pdf"

    def build_page_render_path(self, request_id: str, page_number: int) -> Path:
        return self.image_dir / f"{request_id}_page_{page_number:04d}_render.png"

    def build_markdown_path(self, request_id: str, page_number: int) -> Path:
        return self.markdown_dir / f"{request_id}_page_{page_number:04d}.md"

    def build_manifest_path(self, request_id: str) -> Path:
        return self.json_dir / f"{request_id}_manifest.json"

    # ---------------------------------------------------------
    # Relative Path Helper
    # ---------------------------------------------------------

    def relative(self, absolute_path: Path) -> str:
        return str(
            absolute_path.relative_to(self.artifact_root)
        ).replace("\\", "/")

    # ---------------------------------------------------------
    # Debug
    # ---------------------------------------------------------

    def describe(self) -> Dict[str, str]:

        return {
            "artifact_root": str(self.artifact_root),
            "pdf_dir": str(self.pdf_dir),
            "image_dir": str(self.image_dir),
            "markdown_dir": str(self.markdown_dir),
            "json_dir": str(self.json_dir),

            "vlm_batch_pages": str(self.vlm_batch_pages),
            "vlm_max_new_tokens": str(self.vlm_max_new_tokens),

            "pdf_render_dpi": str(self.pdf_render_dpi),
            "pdf_cpu_workers": str(self.pdf_cpu_workers),
            "pdf_prefetch_windows": str(self.pdf_prefetch_windows),
            "pdf_window_pages": str(self.pdf_window_pages),

            "pdf_save_rendered_images": str(self.pdf_save_rendered_images),
        }


# ---------------------------------------------------------
# Singleton
# ---------------------------------------------------------

GPU_SERVICE_CONFIG = GPUServiceConfig()