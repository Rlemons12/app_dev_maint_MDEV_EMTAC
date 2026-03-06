from __future__ import annotations

import gc
import os
import tempfile
import shutil
import time
import traceback
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

# SCHEMAS (moved to schemas folder)
from app.schemas.vision import (
    VisionPDFResponse,
    VisionPDFChunk,
    VisionMarkdownResponse,
    VisionDescribeResponse,
)

from app.coordinators.document_coordinator import DOC_COORDINATOR, DocumentRequest
from app.models.model_manager import GPU_MODELS
from app.config.gpu_logger import (
    gpu_info,
    gpu_debug,
    gpu_warning,
    gpu_error,
    get_request_id,
)
from app.config.gpu_service_config import GPU_SERVICE_CONFIG

try:
    from PIL import Image
except Exception:
    Image = None

router = APIRouter(prefix="/vision", tags=["vision"])


# =========================================================
# Utilities
# =========================================================

def _ensure_pil():
    if Image is None:
        raise RuntimeError("Pillow not installed in GPU service environment.")


def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        return tmp.name


def _safe_remove(path: Optional[str], rid: Optional[str] = None):
    if not path:
        return
    try:
        os.remove(path)
        gpu_debug(f"Removed temp file: {path}", rid)
    except Exception:
        pass


def _safe_cuda_cleanup():
    try:
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _normalize_text(md: str) -> str:
    return (md or "").strip()


def _vlm_image_to_text(
    *,
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:

    _ensure_pil()

    img = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        text = prompt

    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
    )

    try:
        if hasattr(model, "device") and str(model.device).startswith("cuda"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception:
        pass

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
        )

    try:
        decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    except Exception:
        decoded = str(out)

    return (decoded or "").strip()


# =========================================================
# ENDPOINTS
# =========================================================


# ---------------------------------------------------------
# MARKDOWN
# ---------------------------------------------------------

@router.post("/markdown", response_model=VisionMarkdownResponse)
def vision_markdown(
    file: UploadFile = File(...),
    model: str = Query("nu_markdown"),
    max_new_tokens: int = Query(768, ge=32, le=4096),
):

    rid = get_request_id()
    t0 = time.time()
    temp_path: Optional[str] = None

    gpu_info(f"Vision markdown request | file={file.filename}", rid)

    try:
        temp_path = _save_upload(file)

        ext = Path(file.filename or "image.png").suffix or ".png"
        image_path = GPU_SERVICE_CONFIG.image_dir / f"{rid}_source{ext}"
        shutil.copy2(temp_path, image_path)

        vlm, processor = GPU_MODELS.get_vision_model(model)

        prompt = (
            "Extract clean, faithful Markdown from this page/image. "
            "Preserve headings, tables (as markdown tables), lists, and code blocks. "
            "Do not hallucinate. Output Markdown only."
        )

        md = _vlm_image_to_text(
            model=vlm,
            processor=processor,
            image_path=str(image_path),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        md_path = GPU_SERVICE_CONFIG.markdown_dir / f"{rid}.md"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)

        manifest = {
            "request_id": rid,
            "model": model,
            "markdown_file": GPU_SERVICE_CONFIG.relative(md_path),
            "source_image": GPU_SERVICE_CONFIG.relative(image_path),
        }

        with open(GPU_SERVICE_CONFIG.build_manifest_path(rid), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        total_s = time.time() - t0

        return VisionMarkdownResponse(
            markdown=md,
            model=model,
            request_id=rid,
            timing={"total_s": round(total_s, 3)},
        )

    except torch.cuda.OutOfMemoryError:
        _safe_cuda_cleanup()
        raise HTTPException(
            status_code=503,
            detail={"error": "CUDA out of memory", "request_id": rid},
        )

    except Exception as e:
        gpu_error(traceback.format_exc(), rid)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "request_id": rid},
        )

    finally:
        _safe_remove(temp_path, rid)


# ---------------------------------------------------------
# DESCRIBE
# ---------------------------------------------------------

@router.post("/describe", response_model=VisionDescribeResponse)
def vision_describe(
    file: UploadFile = File(...),
    model: str = Query("nu_markdown"),
    max_new_tokens: int = Query(256, ge=32, le=2048),
):

    rid = get_request_id()
    t0 = time.time()
    temp_path: Optional[str] = None

    try:
        temp_path = _save_upload(file)

        ext = Path(file.filename or "image.png").suffix or ".png"
        image_path = GPU_SERVICE_CONFIG.image_dir / f"{rid}_source{ext}"
        shutil.copy2(temp_path, image_path)

        vlm, processor = GPU_MODELS.get_vision_model(model)

        prompt = (
            "Describe the image precisely and succinctly. "
            "Focus on visible content only."
        )

        desc = _vlm_image_to_text(
            model=vlm,
            processor=processor,
            image_path=str(image_path),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        manifest = {
            "request_id": rid,
            "model": model,
            "image": GPU_SERVICE_CONFIG.relative(image_path),
            "description": desc,
        }

        with open(GPU_SERVICE_CONFIG.build_manifest_path(rid), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        total_s = time.time() - t0

        return VisionDescribeResponse(
            description=desc,
            model=model,
            request_id=rid,
            timing={"total_s": round(total_s, 3)},
        )

    except torch.cuda.OutOfMemoryError:
        _safe_cuda_cleanup()
        raise HTTPException(
            status_code=503,
            detail={"error": "CUDA out of memory", "request_id": rid},
        )

    except Exception as e:
        gpu_error(traceback.format_exc(), rid)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "request_id": rid},
        )

    finally:
        _safe_remove(temp_path, rid)


# ---------------------------------------------------------
# PDF
# ---------------------------------------------------------

@router.post("/pdf", response_model=VisionPDFResponse)
def vision_pdf(
    file: UploadFile = File(...),
    model: str = Query("nu_markdown"),
    max_pages: int = Query(50, ge=1, le=500),
    dpi: int = Query(200, ge=72, le=400),
    max_new_tokens: int = Query(1024, ge=64, le=4096),
    min_image_px: int = Query(80),
    min_image_bytes: int = Query(2048),

    window_pages: int = Query(5, ge=1, le=50),
    cpu_render_workers: int = Query(4, ge=1, le=32),
    gpu_max_inflight: int = Query(2, ge=1, le=8),

    start_page: int = Query(1, ge=1),
    end_page: Optional[int] = Query(None, ge=1),
):

    rid = get_request_id()
    temp_pdf: Optional[str] = None

    gpu_info(f"VISION /pdf route start | file={file.filename} model={model}", rid)

    try:
        temp_pdf = _save_upload(file)

        pdf_path = GPU_SERVICE_CONFIG.build_pdf_path(rid)
        shutil.copy2(temp_pdf, pdf_path)

        req = DocumentRequest(
            pdf_path=str(pdf_path),
            model=model,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            dpi=dpi,
            max_new_tokens=max_new_tokens,
            min_image_px=min_image_px,
            min_image_bytes=min_image_bytes,
            window_pages=window_pages,
            cpu_render_workers=cpu_render_workers,
            gpu_max_inflight=gpu_max_inflight,
            keep_intermediate=True,
            request_id=rid,
        )

        result = DOC_COORDINATOR.process_scanned_pdf(req)

        return VisionPDFResponse(
            source_path=result.source_path,
            doc_type=result.doc_type,
            total_pages=result.total_pages,
            chunks=result.chunks,
            images=result.images,
            model=result.model,
            request_id=result.request_id,
            timing=result.timing,
        )

    except torch.cuda.OutOfMemoryError:
        _safe_cuda_cleanup()
        raise HTTPException(
            status_code=503,
            detail={"error": "CUDA out of memory", "request_id": rid},
        )

    except Exception as e:
        gpu_error(traceback.format_exc(), rid)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "request_id": rid},
        )

    finally:
        _safe_remove(temp_pdf, rid)