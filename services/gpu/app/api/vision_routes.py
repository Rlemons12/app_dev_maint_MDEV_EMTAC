# services/gpu/app/api/vision_routes.py
# Drop-in route update: use DOC_COORDINATOR instead of doing heavy work in-route.
from __future__ import annotations

import shutil
import traceback
from typing import Optional

import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.config.gpu_logger import get_request_id, gpu_info, gpu_debug, gpu_error
from app.config.gpu_service_config import GPU_SERVICE_CONFIG
from app.api.utils import _save_upload, _safe_remove, _safe_cuda_cleanup

from app.models.schemas import VisionPDFResponse
from app.orchestrators.document_coordinator import DOC_COORDINATOR, DocumentRequest

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/pdf", response_model=VisionPDFResponse)
def vision_pdf(
    file: UploadFile = File(...),
    model: str = Query("nu_markdown"),
    max_pages: int = Query(50, ge=1, le=500),
    dpi: int = Query(200, ge=72, le=400),
    max_new_tokens: int = Query(1024, ge=64, le=4096),
    min_image_px: int = Query(80),
    min_image_bytes: int = Query(2048),

    # Optional knobs (safe defaults)
    window_pages: int = Query(5, ge=1, le=50),
    cpu_render_workers: int = Query(4, ge=1, le=32),
    gpu_max_inflight: int = Query(2, ge=1, le=8),

    # Optional page range
    start_page: int = Query(1, ge=1),
    end_page: Optional[int] = Query(None, ge=1),
):
    rid = get_request_id()
    temp_pdf: Optional[str] = None

    gpu_info(f"VISION /pdf route start | file={file.filename} model={model}", rid)

    try:
        temp_pdf = _save_upload(file)
        gpu_debug(f"Temp upload saved -> {temp_pdf}", rid)

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
            chunks=result.chunks,   # already dicts aligned to schema fields your client expects
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