from __future__ import annotations

import os

# Prevent BLAS / OpenMP oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import hashlib
import io
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import fitz
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config.gpu_logger import (
    gpu_info,
    gpu_debug,
    gpu_warning,
    gpu_error,
    gpu_phase,
)
from app.config.gpu_service_config import GPU_SERVICE_CONFIG
from app.utils.text_utils import _normalize_text


# ============================================================
# CPU Contracts
# ============================================================

@dataclass(frozen=True)
class CPUPageItem:
    page_number: int
    render_path: Path
    width: int
    height: int
    images: List[Dict[str, Any]]


@dataclass
class CPUPageWindow:
    pdf_path: str
    page_numbers: List[int]
    page_items: List[CPUPageItem]


# ============================================================
# CPU Orchestrator
# ============================================================

class CPUOrchestrator:
    """
    CPU-side orchestration layer.

    Responsibilities:
      - Open PDFs safely on a per-thread basis
      - Plan page windows
      - Render pages and extract embedded images
      - Persist page/chunk artifacts for downstream GPU + DB workflows

    Important rule:
      - This class does not decide anything about GPU model admission or eviction.
      - It only prepares CPU-side data and persists CPU-side outputs.
    """

    DIAGRAM_MIN_PX = 200
    DIAGRAM_MIN_BYTES = 10_000

    def __init__(self):
        self._tls = threading.local()

    # ------------------------------------------------------------
    # PDF Metadata
    # ------------------------------------------------------------

    def get_pdf_page_count(self, pdf_path: str) -> int:
        doc = self._get_thread_doc(pdf_path)
        return int(doc.page_count)

    def _get_thread_doc(self, pdf_path: str) -> fitz.Document:
        pdf_path = str(Path(pdf_path).resolve())

        doc_map = getattr(self._tls, "doc_map", None)
        if doc_map is None:
            doc_map = {}
            self._tls.doc_map = doc_map

        doc = doc_map.get(pdf_path)
        if doc is None:
            doc = fitz.open(pdf_path)
            doc_map[pdf_path] = doc

        return doc

    def close_thread_docs(self):
        doc_map = getattr(self._tls, "doc_map", None)
        if not doc_map:
            return

        for doc in doc_map.values():
            try:
                doc.close()
            except Exception:
                pass

        doc_map.clear()

    # ------------------------------------------------------------
    # Window Planning
    # ------------------------------------------------------------

    def plan_windows(
        self,
        *,
        pdf_path: str,
        page_numbers: List[int],
        window_pages: int,
        rid: str,
    ) -> List[CPUPageWindow]:

        if window_pages <= 0:
            window_pages = 5

        windows: List[CPUPageWindow] = []
        buf: List[int] = []

        for pn in page_numbers:
            buf.append(pn)

            if len(buf) >= window_pages:
                windows.append(
                    CPUPageWindow(
                        pdf_path=pdf_path,
                        page_numbers=buf,
                        page_items=[],
                    )
                )
                buf = []

        if buf:
            windows.append(
                CPUPageWindow(
                    pdf_path=pdf_path,
                    page_numbers=buf,
                    page_items=[],
                )
            )

        gpu_debug(
            f"CPU plan_windows | windows={len(windows)} window_pages={window_pages}",
            rid,
        )

        return windows

    # ------------------------------------------------------------
    # Window Preparation
    # ------------------------------------------------------------

    def prepare_window(
        self,
        window: CPUPageWindow,
        *,
        dpi: int,
        min_image_px: int,
        min_image_bytes: int,
        cpu_workers: int,
        rid: str,
    ) -> CPUPageWindow:

        if not window.page_numbers:
            gpu_warning("CPU prepare_window | empty page_numbers", rid)
            return CPUPageWindow(
                pdf_path=window.pdf_path,
                page_numbers=[],
                page_items=[],
            )

        t0 = time.time()

        gpu_info(
            f"CPU prepare_window start | pages={window.page_numbers[0]}..{window.page_numbers[-1]} "
            f"count={len(window.page_numbers)} dpi={dpi} workers={cpu_workers}",
            rid,
        )

        page_items: List[CPUPageItem] = []

        cpu_limit = max(1, (os.cpu_count() or 4) // 2)
        max_workers = min(max(1, int(cpu_workers)), cpu_limit, 8)

        batches = self._split_pages_for_workers(window.page_numbers, max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self._render_and_extract_page_batch,
                    window.pdf_path,
                    batch,
                    dpi,
                    min_image_px,
                    min_image_bytes,
                    rid,
                )
                for batch in batches
                if batch
            ]

            for future in as_completed(futures):
                try:
                    page_items.extend(future.result())
                except Exception as exc:
                    gpu_warning(f"CPU page batch failed | err={exc}", rid)

        page_items.sort(key=lambda x: x.page_number)

        gpu_phase("CPU_PREP_WINDOW_INNER", time.time() - t0, rid)

        gpu_info(
            f"CPU prepare_window complete | pages={len(page_items)} "
            f"first={page_items[0].page_number if page_items else None} "
            f"last={page_items[-1].page_number if page_items else None}",
            rid,
        )

        return CPUPageWindow(
            pdf_path=window.pdf_path,
            page_numbers=window.page_numbers,
            page_items=page_items,
        )

    # ------------------------------------------------------------
    # Page batching
    # ------------------------------------------------------------

    def _split_pages_for_workers(
        self,
        page_numbers: List[int],
        workers: int,
    ) -> List[List[int]]:

        workers = max(1, int(workers))
        batches: List[List[int]] = [[] for _ in range(workers)]

        for idx, pn in enumerate(page_numbers):
            batches[idx % workers].append(pn)

        return batches

    # ------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------

    def _render_and_extract_page_batch(
        self,
        pdf_path: str,
        page_numbers: List[int],
        dpi: int,
        min_image_px: int,
        min_image_bytes: int,
        rid: str,
    ) -> List[CPUPageItem]:

        if not page_numbers:
            return []

        t0 = time.time()

        gpu_debug(
            f"CPU batch worker start | pages={page_numbers[0]}..{page_numbers[-1]} "
            f"count={len(page_numbers)} dpi={dpi}",
            rid,
        )

        out: List[CPUPageItem] = []

        scale = float(dpi) / 72.0
        matrix = fitz.Matrix(scale, scale)

        doc = self._get_thread_doc(pdf_path)

        for page_number in page_numbers:
            try:
                out.append(
                    self._render_and_extract_single_page_from_doc(
                        doc=doc,
                        page_number=page_number,
                        matrix=matrix,
                        dpi=dpi,
                        min_image_px=min_image_px,
                        min_image_bytes=min_image_bytes,
                        rid=rid,
                    )
                )
            except Exception as exc:
                gpu_warning(
                    f"CPU page worker error | page={page_number} err={exc}",
                    rid,
                )

        gpu_debug(
            f"CPU batch worker complete | pages={len(out)} time={time.time() - t0:.2f}s",
            rid,
        )

        return out

    # ------------------------------------------------------------
    # Single page
    # ------------------------------------------------------------

    def _render_and_extract_single_page_from_doc(
        self,
        *,
        doc: fitz.Document,
        page_number: int,
        matrix: fitz.Matrix,
        dpi: int,
        min_image_px: int,
        min_image_bytes: int,
        rid: str,
    ) -> CPUPageItem:

        t0 = time.time()

        images: List[Dict[str, Any]] = []
        seen_hashes_local = set()

        page = doc.load_page(page_number - 1)

        # ------------------------------------------------
        # Embedded Images
        # ------------------------------------------------

        img_list = page.get_images(full=True) or []

        for idx, img in enumerate(img_list, start=1):
            try:
                xref = int(img[0])
                extracted = doc.extract_image(xref)
                img_bytes = extracted.get("image")

                if not img_bytes:
                    continue

                if len(img_bytes) < min_image_bytes:
                    continue

                img_hash = hashlib.sha256(img_bytes).hexdigest()

                if img_hash in seen_hashes_local:
                    continue

                seen_hashes_local.add(img_hash)

                with Image.open(io.BytesIO(img_bytes)) as pil:
                    pil.load()
                    w, h = pil.size

                if w < min_image_px or h < min_image_px:
                    continue

                ext = extracted.get("ext", "png")
                img_path = (
                    GPU_SERVICE_CONFIG.image_dir
                    / f"{rid}_page_{page_number:04d}_embedded_{idx:03d}.{ext}"
                )

                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                images.append(
                    {
                        "kind": "embedded",
                        "page_number": page_number,
                        "path": GPU_SERVICE_CONFIG.relative(img_path),
                        "ext": ext,
                        "width": w,
                        "height": h,
                        "byte_size": len(img_bytes),
                        "sha256": img_hash,
                    }
                )

            except Exception as exc:
                gpu_warning(
                    f"CPU embedded image error | page={page_number} idx={idx} err={exc}",
                    rid,
                )

        # ------------------------------------------------
        # Page Render
        # ------------------------------------------------

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
            colorspace=fitz.csRGB,
        )

        render_path = GPU_SERVICE_CONFIG.build_page_render_path(rid, page_number)
        pix.save(str(render_path))

        images.append(
            {
                "kind": "page_render",
                "page_number": page_number,
                "path": GPU_SERVICE_CONFIG.relative(render_path),
                "dpi": dpi,
                "width": pix.width,
                "height": pix.height,
            }
        )

        gpu_debug(
            f"CPU page complete | page={page_number} render={pix.width}x{pix.height} "
            f"images={len(images)} time={time.time() - t0:.2f}s",
            rid,
        )

        return CPUPageItem(
            page_number=page_number,
            render_path=render_path,
            width=pix.width,
            height=pix.height,
            images=images,
        )

    # ------------------------------------------------------------
    # Image filtering
    # ------------------------------------------------------------

    @staticmethod
    def _is_real_diagram(*, width: int, height: int, size_bytes: int) -> bool:
        if width < CPUOrchestrator.DIAGRAM_MIN_PX:
            return False
        if height < CPUOrchestrator.DIAGRAM_MIN_PX:
            return False
        if size_bytes < CPUOrchestrator.DIAGRAM_MIN_BYTES:
            return False
        return True

    def _page_render_path(self, cpu_item: Optional[CPUPageItem]) -> Optional[str]:
        if not cpu_item:
            return None

        for img in cpu_item.images:
            if img.get("kind") == "page_render":
                return str(img.get("path") or "")

        return None

    def _page_diagram_refs(self, cpu_item: Optional[CPUPageItem]) -> List[str]:
        if not cpu_item:
            return []

        refs: List[str] = []

        for img in cpu_item.images:
            if img.get("kind") != "embedded":
                continue

            w = int(img.get("width") or 0)
            h = int(img.get("height") or 0)
            b = int(img.get("byte_size") or 0)

            if not self._is_real_diagram(width=w, height=h, size_bytes=b):
                continue

            p = img.get("path")
            if p:
                refs.append(str(p))

        refs.sort()
        return refs

    # ------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------

    def persist_window_results(
        self,
        *,
        window: CPUPageWindow,
        page_results: List["GPUPageResult"],
        keep_intermediate: bool,
        rid: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

        images: List[Dict[str, Any]] = []
        chunks: List[Dict[str, Any]] = []

        cpu_by_page = {p.page_number: p for p in window.page_items}

        def split_markdown(md: str) -> List[str]:
            parts = [p.strip() for p in (md or "").split("\n\n") if p.strip()]

            merged: List[str] = []
            buf = ""

            for part in parts:
                if len(buf) < 500:
                    buf = (buf + "\n\n" + part).strip() if buf else part
                else:
                    merged.append(buf)
                    buf = part

            if buf:
                merged.append(buf)

            return merged

        def detect_table(md: str) -> bool:
            if "|" in md and "---" in md:
                return True

            for line in md.splitlines():
                if line.count("|") >= 3:
                    return True

            return False

        for pr in page_results:
            pn = pr.page_number
            cpu_item = cpu_by_page.get(pn)

            if cpu_item:
                images.extend(cpu_item.images)

            md = pr.markdown or ""
            page_chunks = split_markdown(md)

            diagram_refs = self._page_diagram_refs(cpu_item)
            page_render = self._page_render_path(cpu_item)

            for cidx, chunk_text in enumerate(page_chunks, start=1):
                chunk_id = f"{rid}_p{pn:04d}_c{cidx:03d}"
                md_path = GPU_SERVICE_CONFIG.markdown_dir / f"{chunk_id}.md"

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(chunk_text)

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page_number": pn,
                        "markdown": chunk_text,
                        "text": _normalize_text(chunk_text),
                        "image_refs": diagram_refs,
                        "images": diagram_refs,
                        "page_render": page_render,
                        "markdown_file": GPU_SERVICE_CONFIG.relative(md_path),
                        "contains_table": detect_table(chunk_text),
                    }
                )

        if not keep_intermediate:
            gpu_warning(
                "CPU persist_window_results | keep_intermediate=False not implemented",
                rid,
            )

        gpu_debug(
            f"CPU persist_window_results complete | images={len(images)} chunks={len(chunks)}",
            rid,
        )

        return images, chunks


if TYPE_CHECKING:
    from app.orchestrators.gpu_orchestrator import GPUPageResult