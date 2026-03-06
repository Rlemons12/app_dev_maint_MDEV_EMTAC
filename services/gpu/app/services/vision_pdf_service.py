# services/gpu/app/services/vision_pdf_service.py
from __future__ import annotations

import time
import json
import hashlib
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import fitz
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from app.models.model_manager import GPU_MODELS
from app.config.gpu_logger import (
    gpu_info,
    gpu_debug,
    gpu_warning,
    gpu_phase,
    gpu_snapshot,
)
from app.config.gpu_service_config import GPU_SERVICE_CONFIG
from app.models.schemas import VisionPDFChunk
from app.utils.text_utils import _normalize_text

# NOTE: Your original snippet calls VLM_SERVICE.images_to_text(...) but did not import it.
# This import is required for the code to run as shown.
from app.services.vlm_service import VLM_SERVICE

# You had this import; leaving it intact for compatibility even if unused here.
from app.utils.vlm_utils import _vlm_image_to_text


class VisionPDFService:
    """
    Scanned-PDF → page images (CPU) → batched VLM inference (GPU) → markdown chunks + manifest.

    True drop-in replacement for your original implementation.

    Key Fix (fitz.open bottleneck):
      - Uses a thread-local PyMuPDF document cache so each worker thread opens the PDF once.
      - Normalizes pdf_path to a resolved string before any fitz.open usage.
      - Submits per-thread cleanup jobs at the end to close cached docs.

    Everything else:
      - Preserves your page-level futures pipeline, embedded scan extraction, dedupe, fallback render,
        GPU batching, chunk writing, and manifest behavior.
    """

    def __init__(self) -> None:
        self._tls = threading.local()

    # =========================================================
    # THREAD-LOCAL DOC CACHE
    # =========================================================

    def _get_thread_doc(self, pdf_path_str: str):
        """
        Return a thread-local fitz.Document for this pdf_path_str.
        Each ThreadPoolExecutor worker thread gets its own doc handle.
        """
        doc_map = getattr(self._tls, "doc_map", None)
        if doc_map is None:
            doc_map = {}
            self._tls.doc_map = doc_map

        doc = doc_map.get(pdf_path_str)
        if doc is None:
            doc = fitz.open(pdf_path_str)
            doc_map[pdf_path_str] = doc

        return doc

    def _close_thread_doc(self, pdf_path_str: str) -> None:
        """
        Close the thread-local doc handle for the given path, if present.
        Intended to be called *inside* pool threads (so each thread closes its own handle).
        """
        doc_map = getattr(self._tls, "doc_map", None)
        if not doc_map:
            return

        doc = doc_map.pop(pdf_path_str, None)
        if doc is not None:
            try:
                doc.close()
            except Exception:
                # Avoid raising during cleanup
                pass

    # =========================================================
    # MAIN ENTRYPOINT
    # =========================================================

    def process_pdf(
        self,
        pdf_path: str,
        model: str,
        max_pages: int,
        dpi: int,
        max_new_tokens: int,
        min_image_px: int,
        min_image_bytes: int,
        request_id: str,
    ) -> Dict[str, Any]:

        rid = request_id
        total_start = time.time()

        # -------------------------------------------------
        # Normalize path (critical for fitz.open stability)
        # -------------------------------------------------
        pdf_path_str = str(Path(pdf_path).resolve())
        pdf_path_obj = Path(pdf_path_str)

        gpu_info(
            f"VISION PDF start | file={pdf_path_obj} model={model}",
            rid,
        )

        gpu_snapshot("startup", rid)

        # -------------------------------------------------
        # MODEL LOAD
        # -------------------------------------------------
        t = time.time()
        vlm, processor = GPU_MODELS.get_vision_model(model)
        gpu_phase("MODEL_LOAD", time.time() - t, rid)

        prompt = (
            "Extract clean, faithful Markdown from this scanned PDF page image. "
            "Preserve headings, tables, lists, and code blocks. "
            "Do not hallucinate."
        )

        chunks: List[VisionPDFChunk] = []
        images: List[Dict[str, Any]] = []

        # -------------------------------------------------
        # Shared dedupe state for scan extraction
        # -------------------------------------------------
        seen_hashes = set()
        seen_lock = threading.Lock()

        # =========================================================
        # OPEN PDF (metadata only)
        # =========================================================
        t0 = time.time()
        with fitz.open(pdf_path_str) as meta_doc:
            total_pages = int(meta_doc.page_count)
        gpu_phase("PDF_OPEN_META", time.time() - t0, rid)

        # max_pages behavior: if <=0 treat as "all pages"
        if max_pages and max_pages > 0:
            use_pages = min(total_pages, int(max_pages))
        else:
            use_pages = total_pages

        gpu_info(
            f"PDF opened | total_pages={total_pages} processing={use_pages}",
            rid,
        )

        # =========================================================
        # WINDOW PLAN (batch pages per GPU call)
        # =========================================================
        batch_pages = int(GPU_SERVICE_CONFIG.vlm_batch_pages)
        if batch_pages <= 0:
            batch_pages = 1

        prefetch_windows = int(GPU_SERVICE_CONFIG.pdf_prefetch_windows)
        if prefetch_windows <= 0:
            prefetch_windows = 1

        gpu_info(
            f"DOC plan | use_pages={use_pages} batch_pages={batch_pages} prefetch_windows={prefetch_windows}",
            rid,
        )

        windows: List[tuple[int, int]] = []
        for start in range(0, use_pages, batch_pages):
            end = min(start + batch_pages, use_pages)
            windows.append((start, end))

        # =========================================================
        # OVERLAPPED PIPELINE:
        # - CPU prepares pages in background threads (page futures)
        # - GPU infers current window
        # =========================================================

        prep_total_start = time.time()

        # Track futures per page index so we can prefetch and then block only when needed
        futures_by_page: Dict[int, Any] = {}

        def _submit_page(idx: int, pool: ThreadPoolExecutor) -> None:
            """
            Submit a page preparation job if not already queued.
            Uses the high-throughput worker architecture.
            """
            if idx in futures_by_page:
                return

            futures_by_page[idx] = pool.submit(
                self._prepare_page_worker,
                pdf_path_str,  # IMPORTANT: pass a resolved string into workers
                idx,
                dpi,
                rid,
                min_image_px,
                min_image_bytes,
                seen_hashes,
                seen_lock,
            )

        def _submit_windows(window_i: int, pool: ThreadPoolExecutor) -> None:
            """
            Ensure page-prep futures are submitted for window_i .. window_i+prefetch_windows-1.
            """
            max_win = min(len(windows), window_i + prefetch_windows)
            for wi in range(window_i, max_win):
                s, e = windows[wi]
                for page_idx in range(s, e):
                    _submit_page(page_idx, pool)

        cpu_workers = int(GPU_SERVICE_CONFIG.pdf_cpu_workers)
        if cpu_workers <= 0:
            cpu_workers = max(1, (threading.active_count() or 1))

        # Cap worker count a bit to reduce thrash on machines with many logical cores
        # (keeps your behavior predictable; adjust if desired)
        cpu_workers = min(cpu_workers, max(1, (os.cpu_count() or 4)))

        with ThreadPoolExecutor(max_workers=cpu_workers) as pool:

            # Prime the pump: submit first few windows
            _submit_windows(0, pool)

            # Process each window
            for win_i, (win_start, win_end) in enumerate(windows):

                # While we work on this window, make sure the next windows are already queued
                _submit_windows(win_i, pool)

                # -------------------------------------------------
                # COLLECT PREP RESULTS FOR THIS WINDOW
                # (blocks only for this window's pages)
                # -------------------------------------------------
                win_prep_start = time.time()

                prepared_pages: List[Dict[str, Any]] = []
                for page_idx in range(win_start, win_end):
                    prepared_pages.append(futures_by_page[page_idx].result())

                # Ensure correct order
                prepared_pages.sort(key=lambda x: x["page_number"])

                gpu_phase("CPU_PREP_WINDOW", time.time() - win_prep_start, rid)

                if prepared_pages:
                    first_page_num = prepared_pages[0]["page_number"]
                    last_page_num = prepared_pages[-1]["page_number"]
                else:
                    first_page_num = None
                    last_page_num = None

                gpu_info(
                    f"CPU prepare_window complete | pages={len(prepared_pages)} first={first_page_num} last={last_page_num}",
                    rid,
                )

                # -------------------------------------------------
                # REGISTER IMAGE METADATA (for manifest/debug)
                # -------------------------------------------------
                for page_data in prepared_pages:
                    image_path: Optional[Path] = page_data.get("image_path")
                    images.append(
                        {
                            "kind": page_data.get("kind"),
                            "page_number": page_data.get("page_number"),
                            "path": GPU_SERVICE_CONFIG.relative(image_path) if image_path else None,
                            "width": page_data.get("width", 0),
                            "height": page_data.get("height", 0),
                        }
                    )

                # Filter out pages that produced no image_path (e.g., duplicates/failed)
                infer_pages = [p for p in prepared_pages if p.get("image_path") is not None]

                if not infer_pages:
                    gpu_warning(
                        f"No inferable pages in window | win={win_i} pages={win_start+1}..{win_end}",
                        rid,
                    )
                    for page_idx in range(win_start, win_end):
                        futures_by_page.pop(page_idx, None)
                    continue

                # -------------------------------------------------
                # GPU BATCH INFERENCE FOR THIS WINDOW (ONE generate)
                # -------------------------------------------------
                batch_page_numbers = [p["page_number"] for p in infer_pages]
                batch_image_paths = [str(p["image_path"]) for p in infer_pages]

                gpu_info(
                    f"GPU infer_window start | pages={batch_page_numbers[0]}..{batch_page_numbers[-1]} "
                    f"count={len(infer_pages)} model={model}",
                    rid,
                )

                gpu_snapshot("before_vlm", rid)

                t = time.time()

                md_list = VLM_SERVICE.images_to_text(
                    model=vlm,
                    processor=processor,
                    image_paths=batch_image_paths,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    request_id=rid,
                )

                gpu_phase("VLM_INFERENCE", time.time() - t, rid)

                gpu_snapshot("after_vlm", rid)

                # -------------------------------------------------
                # PROCESS EACH PAGE RESULT (split + chunk write)
                # -------------------------------------------------
                for page_data, md in zip(infer_pages, md_list):

                    page_number = page_data["page_number"]
                    page_start = time.time()

                    gpu_debug(
                        f"Markdown size | page={page_number} chars={len(md)}",
                        rid,
                    )

                    # Markdown split
                    t = time.time()
                    page_chunks = self._split_markdown(md)
                    gpu_phase("MARKDOWN_SPLIT", time.time() - t, rid)

                    page_images = [
                        img["path"]
                        for img in images
                        if img.get("page_number") == page_number and img.get("path")
                    ]

                    # Chunk write
                    write_start = time.time()

                    for cidx, chunk_text in enumerate(page_chunks, start=1):
                        chunk_id = f"{rid}_p{page_number:04d}_c{cidx:03d}"
                        md_path = self._build_chunk_path(chunk_id)

                        with open(md_path, "w", encoding="utf-8") as f:
                            f.write(chunk_text)

                        chunks.append(
                            VisionPDFChunk(
                                chunk_id=chunk_id,
                                page_number=page_number,
                                markdown=chunk_text,
                                text=_normalize_text(chunk_text),
                                images=page_images,
                            )
                        )

                    gpu_phase("CHUNK_WRITE", time.time() - write_start, rid)

                    gpu_info(
                        f"Page complete | page={page_number} "
                        f"time={time.time() - page_start:.2f}s",
                        rid,
                    )

                # Optional: free completed futures to reduce memory pressure
                for page_idx in range(win_start, win_end):
                    futures_by_page.pop(page_idx, None)

            gpu_phase("PAGE_PREP_TOTAL", time.time() - prep_total_start, rid)

            # -------------------------------------------------
            # IMPORTANT: Close thread-local doc handles
            # -------------------------------------------------
            cleanup_start = time.time()
            cleanup_futures = [
                pool.submit(self._close_thread_doc, pdf_path_str)
                for _ in range(cpu_workers)
            ]
            for f in cleanup_futures:
                try:
                    f.result()
                except Exception:
                    pass
            gpu_phase("PDF_THREAD_DOC_CLEANUP", time.time() - cleanup_start, rid)

        # =========================================================
        # MANIFEST
        # =========================================================
        t = time.time()

        manifest = self._build_manifest(
            chunks,
            images,
            pdf_path_obj,
            model,
            total_pages,
            use_pages,
            seen_hashes,
            rid,
        )

        manifest_path = GPU_SERVICE_CONFIG.build_manifest_path(rid)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        gpu_phase("MANIFEST_BUILD", time.time() - t, rid)

        total_time = time.time() - total_start
        gpu_phase("TOTAL_PROCESS", total_time, rid)

        gpu_info(
            f"VISION PDF complete | pages={use_pages} "
            f"chunks={len(chunks)} images={len(images)}",
            rid,
        )

        gpu_snapshot("complete", rid)

        return {
            "source_path": GPU_SERVICE_CONFIG.relative(pdf_path_obj),
            "doc_type": "pdf",
            "total_pages": total_pages,
            "chunks": chunks,
            "images": images,
            "model": model,
            "request_id": rid,
            "timing": {"total_s": round(total_time, 3)},
        }

    # =========================================================
    # PAGE PREPARATION
    # =========================================================

    def _prepare_page_worker(
        self,
        pdf_path_str: str,   # IMPORTANT: use resolved string for fitz.open / caching
        page_index: int,
        dpi: int,
        rid: str,
        min_image_px: int,
        min_image_bytes: int,
        seen_hashes: set,
        seen_lock,
    ) -> Dict[str, Any]:
        """
        High-throughput PyMuPDF worker

        Preserves your original behavior:
          - embedded scan extraction (first embedded image)
          - thread-safe dedupe via sha256
          - fallback: render page at DPI
          - optional artifact saving
        """

        page_number = page_index + 1

        gpu_debug(f"Preparing page | page={page_number}", rid)

        try:
            # -------------------------------------------------
            # Thread-local PDF handle (fixes repeated fitz.open cost)
            # -------------------------------------------------
            doc = self._get_thread_doc(pdf_path_str)
            page = doc.load_page(page_index)

            # =========================================================
            # FAST PATH: Embedded Scan Extraction
            # =========================================================
            try:
                img_list = page.get_images(full=True) or []
                if img_list:
                    xref = int(img_list[0][0])
                    extracted = doc.extract_image(xref)
                    img_bytes = extracted.get("image")

                    if img_bytes:
                        if len(img_bytes) >= min_image_bytes:
                            img_hash = hashlib.sha256(img_bytes).hexdigest()

                            # Thread-safe dedupe
                            with seen_lock:
                                if img_hash in seen_hashes:
                                    return {
                                        "page_number": page_number,
                                        "image_path": None,
                                        "width": 0,
                                        "height": 0,
                                        "kind": "duplicate_scan_extract",
                                    }
                                seen_hashes.add(img_hash)

                            pil = Image.open(io.BytesIO(img_bytes))
                            width, height = pil.size

                            if width >= min_image_px and height >= min_image_px:
                                ext = extracted.get("ext", "png")

                                img_path = None
                                if GPU_SERVICE_CONFIG.pdf_save_rendered_images:
                                    img_path = (
                                        GPU_SERVICE_CONFIG.image_dir
                                        / f"{rid}_page_{page_number:04d}.{ext}"
                                    )
                                    with open(img_path, "wb") as f:
                                        f.write(img_bytes)

                                return {
                                    "page_number": page_number,
                                    "image_path": img_path,
                                    "width": width,
                                    "height": height,
                                    "kind": "scan_extract",
                                }

            except Exception as e:
                gpu_warning(
                    f"Image extraction failed | page={page_number} err={e}",
                    rid,
                )

            # =========================================================
            # FALLBACK: Render Page
            # =========================================================
            scale = dpi / 72.0
            matrix = fitz.Matrix(scale, scale)

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False,
                colorspace=fitz.csRGB,
            )

            width = pix.width
            height = pix.height

            render_path = None
            if GPU_SERVICE_CONFIG.pdf_save_rendered_images:
                render_path = GPU_SERVICE_CONFIG.build_page_render_path(
                    rid,
                    page_number,
                )
                # IMPORTANT: do NOT pass unsupported kwargs (e.g. deflate)
                pix.save(str(render_path))

            return {
                "page_number": page_number,
                "image_path": render_path,
                "width": width,
                "height": height,
                "kind": "render_fallback",
            }

        except Exception as e:
            gpu_warning(
                f"Page prepare failed | page={page_number} err={e}",
                rid,
            )
            return {
                "page_number": page_number,
                "image_path": None,
                "width": 0,
                "height": 0,
                "kind": "prepare_failed",
            }

    # =========================================================
    # MARKDOWN SPLIT
    # =========================================================

    def _split_markdown(self, md: str) -> List[str]:
        parts = [p.strip() for p in (md or "").split("\n\n") if p.strip()]

        merged: List[str] = []
        buf = ""

        for p in parts:
            if len(buf) < 500:
                buf = (buf + "\n\n" + p).strip() if buf else p
            else:
                merged.append(buf)
                buf = p

        if buf:
            merged.append(buf)

        return merged

    # =========================================================
    # CHUNK PATH BUILDER
    # =========================================================

    def _build_chunk_path(self, chunk_id: str) -> Path:
        return GPU_SERVICE_CONFIG.markdown_dir / f"{chunk_id}.md"

    # =========================================================
    # MANIFEST
    # =========================================================

    def _build_manifest(
        self,
        chunks: List[VisionPDFChunk],
        images: List[Dict[str, Any]],
        pdf_path: Path,
        model: str,
        total_pages: int,
        use_pages: int,
        seen_hashes: set,
        rid: str,
    ) -> Dict[str, Any]:

        return {
            "request_id": rid,
            "model": model,
            "total_pages": total_pages,
            "processed_pages": use_pages,
            "pdf": GPU_SERVICE_CONFIG.relative(pdf_path),
            "images": images,
            "stats": {
                "total_chunks": len(chunks),
                "images_saved": len(images),
                "unique_images": len(seen_hashes),
            },
        }


VISION_PDF_SERVICE = VisionPDFService()