from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from app.config.gpu_logger import (
    gpu_info,
    gpu_debug,
    gpu_warning,
    gpu_phase,
    gpu_snapshot,
)

from app.config.gpu_service_config import GPU_SERVICE_CONFIG
from app.models.model_manager import GPU_MODELS
from app.services.vlm_service import VLM_SERVICE


# ============================================================
# GPU Contracts
# ============================================================

@dataclass(frozen=True)
class GPUPageResult:
    page_number: int
    markdown: str


# ============================================================
# GPU Orchestrator
# ============================================================

class GPUOrchestrator:
    """
    GPU-side orchestration layer.

    Responsibilities:
      - Ensure VLM model is loaded
      - Run inference over page render images
      - Keep VRAM stable
      - Provide observability hooks

    Behavioral compatibility: 100%
    """

    PROMPT: str = "Transcribe this page into Markdown."

    def __init__(self):
        self._loaded_model_name: Optional[str] = None
        self._vlm = None
        self._processor = None

    # ------------------------------------------------------------
    # Ensure Model Loaded
    # ------------------------------------------------------------

    def ensure_model_loaded(self, model_name: str, *, rid: str) -> None:

        if (
            self._loaded_model_name == model_name
            and self._vlm is not None
            and self._processor is not None
        ):
            return

        t0 = time.time()

        vlm, processor = GPU_MODELS.get_vision_model(model_name)

        self._loaded_model_name = model_name
        self._vlm = vlm
        self._processor = processor

        gpu_phase("GPU_ENSURE_MODEL", time.time() - t0, rid)

    # ------------------------------------------------------------
    # Inference Window
    # ------------------------------------------------------------

    def infer_window(
        self,
        *,
        window: "CPUPageWindow",
        model: str,
        max_new_tokens: int,
        rid: str,
    ) -> List[GPUPageResult]:

        t0 = time.time()

        if not window.page_items:
            gpu_warning("GPU infer_window | empty window.page_items", rid)
            return []

        self.ensure_model_loaded(model, rid=rid)

        if self._vlm is None or self._processor is None:
            raise RuntimeError(f"GPU model not loaded correctly: model={model}")

        # ------------------------------------------------------------
        # Token clamping
        # ------------------------------------------------------------

        try:
            cfg_limit = int(getattr(GPU_SERVICE_CONFIG, "vlm_max_new_tokens", 0) or 0)
        except Exception:
            cfg_limit = 0

        effective_max_new_tokens = int(max_new_tokens)

        if cfg_limit > 0:
            effective_max_new_tokens = min(effective_max_new_tokens, cfg_limit)

        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------

        page_items = window.page_items
        first_page = window.page_numbers[0]
        last_page = window.page_numbers[-1]

        gpu_info(
            "GPU infer_window start | "
            f"pages={first_page}..{last_page} "
            f"count={len(page_items)} model={model} "
            f"max_new_tokens={effective_max_new_tokens}",
            rid,
        )

        enable_snapshots = bool(
            getattr(GPU_SERVICE_CONFIG, "enable_gpu_snapshots", True)
        )

        if enable_snapshots:
            gpu_snapshot("before_vlm", rid)

        results: List[GPUPageResult] = []

        # ------------------------------------------------------------
        # Batch configuration
        # ------------------------------------------------------------

        use_batch = bool(getattr(GPU_SERVICE_CONFIG, "vlm_use_batch_infer", False))
        batch_size = int(getattr(GPU_SERVICE_CONFIG, "vlm_batch_pages", 1) or 1)

        vlm = self._vlm
        processor = self._processor

        # ------------------------------------------------------------
        # Batch inference path
        # ------------------------------------------------------------

        if use_batch and batch_size > 1:

            image_paths = [str(item.render_path) for item in page_items]
            page_numbers = [item.page_number for item in page_items]

            t_batch = time.time()

            md_list = VLM_SERVICE.images_to_text(
                model=vlm,
                processor=processor,
                image_paths=image_paths,
                prompt=self.PROMPT,
                max_new_tokens=effective_max_new_tokens,
                request_id=rid,
                max_batch_size=batch_size,
            )

            gpu_phase("GPU_BATCH_INFER", time.time() - t_batch, rid)

            for pn, md in zip(page_numbers, md_list):

                md = md or ""

                gpu_debug(
                    f"GPU page infer complete | page={pn} chars={len(md)}",
                    rid,
                )

                results.append(
                    GPUPageResult(
                        page_number=pn,
                        markdown=md,
                    )
                )

        # ------------------------------------------------------------
        # Sequential inference path
        # ------------------------------------------------------------

        else:

            for item in page_items:

                t_page = time.time()

                markdown = VLM_SERVICE.image_to_text(
                    model=vlm,
                    processor=processor,
                    image_path=str(item.render_path),
                    prompt=self.PROMPT,
                    max_new_tokens=effective_max_new_tokens,
                    request_id=rid,
                )

                markdown = markdown or ""

                gpu_debug(
                    f"GPU page infer complete | page={item.page_number} chars={len(markdown)}",
                    rid,
                )

                results.append(
                    GPUPageResult(
                        page_number=item.page_number,
                        markdown=markdown,
                    )
                )

                gpu_phase("GPU_PAGE_INFER", time.time() - t_page, rid)

        # ------------------------------------------------------------
        # Final logging
        # ------------------------------------------------------------

        if enable_snapshots:
            gpu_snapshot("after_vlm", rid)

        gpu_phase("GPU_INFER_WINDOW_INNER", time.time() - t0, rid)

        gpu_info(f"GPU infer_window complete | pages={len(results)}", rid)

        return results


if TYPE_CHECKING:
    from app.orchestrators.cpu_orchestrator import CPUPageWindow