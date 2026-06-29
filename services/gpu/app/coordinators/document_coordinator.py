# services/gpu/app/orchestrators/document_coordinator.py

from __future__ import annotations

import os
import time
import uuid
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from app.config.gpu_logger import (
    gpu_info,
    gpu_debug,
    gpu_warning,
    gpu_error,
    gpu_phase,
    gpu_snapshot,
)

from app.config.gpu_service_config import GPU_SERVICE_CONFIG
from app.orchestrators.cpu_orchestrator import CPUOrchestrator, CPUPageWindow
from app.orchestrators.gpu_orchestrator import GPUOrchestrator, GPUPageResult


# ============================================================
# Data Contracts
# ============================================================

@dataclass(frozen=True)
class DocumentRequest:
    pdf_path: str
    model: str = "nu_markdown"

    start_page: int = 1
    end_page: Optional[int] = None
    max_pages: int = 50

    dpi: int = GPU_SERVICE_CONFIG.pdf_render_dpi
    max_new_tokens: int = GPU_SERVICE_CONFIG.vlm_max_new_tokens

    min_image_px: int = 80
    min_image_bytes: int = 2048

    cpu_render_workers: int = GPU_SERVICE_CONFIG.pdf_cpu_workers
    gpu_max_inflight: int = GPU_SERVICE_CONFIG.pdf_prefetch_windows
    window_pages: int = GPU_SERVICE_CONFIG.pdf_window_pages

    keep_intermediate: bool = True

    request_id: Optional[str] = None


@dataclass
class DocumentResponse:
    request_id: str
    source_path: str
    doc_type: str
    total_pages: int
    processed_pages: int
    model: str
    timing: Dict[str, Any]
    images: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    manifest_path: str


# ============================================================
# DocumentCoordinator
# ============================================================

class DocumentCoordinator:

    def __init__(
        self,
        cpu_orchestrator: Optional[CPUOrchestrator] = None,
        gpu_orchestrator: Optional[GPUOrchestrator] = None,
    ):
        self.cpu = cpu_orchestrator or CPUOrchestrator()
        self.gpu = gpu_orchestrator or GPUOrchestrator()

    # ------------------------------------------------------------
    # Main Entry
    # ------------------------------------------------------------
    def process_scanned_pdf(self, req: DocumentRequest) -> DocumentResponse:

        rid = req.request_id or str(uuid.uuid4())
        t_total0 = time.time()

        pdf_path = str(req.pdf_path)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)

        gpu_info(
            f"DOC_COORD start | pdf={pdf_path} model={req.model} "
            f"dpi={req.dpi} window_pages={req.window_pages} "
            f"cpu_workers={req.cpu_render_workers} gpu_inflight={req.gpu_max_inflight}",
            rid,
        )

        gpu_snapshot("DOC_COORD startup", rid)

        # ------------------------------------------------------------
        # Step 1 — Plan windows
        # ------------------------------------------------------------

        t0 = time.time()

        total_pages = self.cpu.get_pdf_page_count(pdf_path)

        page_range = self._resolve_page_range(
            total_pages=total_pages,
            start_page=req.start_page,
            end_page=req.end_page,
            max_pages=req.max_pages,
        )

        windows = self.cpu.plan_windows(
            pdf_path=pdf_path,
            page_numbers=page_range,
            window_pages=req.window_pages,
            rid=rid,
        )

        gpu_phase("DOC_PLAN_WINDOWS", time.time() - t0, rid)

        gpu_info(
            f"DOC planned | total_pages={total_pages} "
            f"selected_pages={len(page_range)} windows={len(windows)}",
            rid,
        )

        # ------------------------------------------------------------
        # Step 2 — Ensure GPU model loaded
        # ------------------------------------------------------------

        t0 = time.time()
        self.gpu.ensure_model_loaded(req.model, rid=rid)
        gpu_phase("DOC_MODEL_READY", time.time() - t0, rid)

        # ------------------------------------------------------------
        # Step 3 — Pipeline execution
        # ------------------------------------------------------------

        images: List[Dict[str, Any]] = []
        chunks: List[Dict[str, Any]] = []

        processed_pages = 0

        prepared_queue: deque[CPUPageWindow] = deque()

        window_idx = 0

        executor = ThreadPoolExecutor(max_workers=req.gpu_max_inflight)

        futures = []

        def schedule_prepare():
            nonlocal window_idx

            while window_idx < len(windows) and len(futures) < req.gpu_max_inflight:

                w = windows[window_idx]
                window_idx += 1

                futures.append(
                    executor.submit(
                        self.cpu.prepare_window,
                        w,
                        dpi=req.dpi,
                        min_image_px=req.min_image_px,
                        min_image_bytes=req.min_image_bytes,
                        cpu_workers=req.cpu_render_workers,
                        rid=rid,
                    )
                )

        schedule_prepare()

        while futures:

            future = futures.pop(0)

            try:

                t_cpu = time.time()
                prepared: CPUPageWindow = future.result()
                gpu_phase("CPU_PREP_WINDOW", time.time() - t_cpu, rid)

                prepared_queue.append(prepared)

            except Exception as e:

                gpu_error(f"CPU window preparation failed: {e}", rid)
                continue

            schedule_prepare()

            current = prepared_queue.popleft()

            # --------------------------------------------------------
            # GPU Inference
            # --------------------------------------------------------

            try:

                t_gpu = time.time()

                page_results: List[GPUPageResult] = self.gpu.infer_window(
                    window=current,
                    model=req.model,
                    max_new_tokens=req.max_new_tokens,
                    rid=rid,
                )

                gpu_phase("GPU_INFER_WINDOW", time.time() - t_gpu, rid)

            except Exception as e:

                gpu_error(f"GPU inference failed: {e}", rid)
                continue

            # --------------------------------------------------------
            # Persist results
            # --------------------------------------------------------

            try:

                t_post = time.time()

                w_images, w_chunks = self.cpu.persist_window_results(
                    window=current,
                    page_results=page_results,
                    keep_intermediate=req.keep_intermediate,
                    rid=rid,
                )

                gpu_phase("CPU_PERSIST_WINDOW", time.time() - t_post, rid)

                images.extend(w_images)
                chunks.extend(w_chunks)

                processed_pages += len(current.page_items)

                gpu_info(
                    f"DOC window complete | pages={len(current.page_items)} "
                    f"processed_pages={processed_pages}/{len(page_range)} "
                    f"chunks_total={len(chunks)} images_total={len(images)}",
                    rid,
                )

            except Exception as e:

                gpu_error(f"Result persistence failed: {e}", rid)

        executor.shutdown(wait=True)

        # ------------------------------------------------------------
        # Step 4 — Manifest
        # ------------------------------------------------------------

        t0 = time.time()

        manifest = self._build_manifest(
            rid=rid,
            model=req.model,
            total_pages=total_pages,
            processed_pages=processed_pages,
            pdf_path=pdf_path,
            images=images,
            chunks=chunks,
        )

        manifest_path = GPU_SERVICE_CONFIG.build_manifest_path(rid)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        gpu_phase("DOC_MANIFEST_WRITE", time.time() - t0, rid)

        total_s = time.time() - t_total0

        gpu_phase("DOC_TOTAL", total_s, rid)
        gpu_snapshot("DOC_COORD complete", rid)

        return DocumentResponse(
            request_id=rid,
            source_path=GPU_SERVICE_CONFIG.relative(Path(pdf_path)),
            doc_type="pdf",
            total_pages=total_pages,
            processed_pages=processed_pages,
            model=req.model,
            timing={"total_s": round(total_s, 3)},
            images=images,
            chunks=chunks,
            manifest_path=GPU_SERVICE_CONFIG.relative(manifest_path),
        )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _resolve_page_range(
        self,
        *,
        total_pages: int,
        start_page: int,
        end_page: Optional[int],
        max_pages: int,
    ) -> List[int]:

        s = max(1, start_page)
        e = end_page if end_page else total_pages
        e = min(e, total_pages)

        pages = list(range(s, e + 1))

        if max_pages:
            pages = pages[:max_pages]

        return pages

    def _build_manifest(
        self,
        *,
        rid: str,
        model: str,
        total_pages: int,
        processed_pages: int,
        pdf_path: str,
        images: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        table_chunks = sum(1 for c in chunks if bool(c.get("contains_table")))

        return {
            "request_id": rid,
            "model": model,
            "total_pages": total_pages,
            "processed_pages": processed_pages,
            "pdf": GPU_SERVICE_CONFIG.relative(Path(pdf_path)),
            "images": images,
            "chunks": chunks,
            "stats": {
                "total_chunks": len(chunks),
                "table_chunks": table_chunks,
                "images_saved": len(images),
            },
        }


DOC_COORDINATOR = DocumentCoordinator()