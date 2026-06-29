from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import io
import base64

from app.runtime.accelerator import ACCELERATOR
from app.config.gpu_logger import gpu_info, gpu_error
from app.models.clip_manager import CLIP_MODELS  # we will define this

router = APIRouter(prefix="/embed-image", tags=["embedding"])


# ---------------------------------------------------------
# Request Model
# ---------------------------------------------------------

class ImageEmbedRequest(BaseModel):
    model: str
    images: List[str]  # base64 encoded images
    normalize: bool = True


class ImageEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    device: str
    dims: int
    normalized: bool


# ---------------------------------------------------------
# Endpoint
# ---------------------------------------------------------

@router.post("", response_model=ImageEmbedResponse)
def embed_image(req: ImageEmbedRequest):

    gpu_info(f"IMAGE EMBED | model={req.model} count={len(req.images)}")

    try:
        model, processor = CLIP_MODELS.get(req.model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        decoded_images = []

        for img_b64 in req.images:
            image_bytes = base64.b64decode(img_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            decoded_images.append(image)

        inputs = processor(images=decoded_images, return_tensors="pt").to(ACCELERATOR.device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        if req.normalize:
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)

        vecs = image_features.cpu().numpy()
        dims = vecs.shape[1]

        return ImageEmbedResponse(
            embeddings=vecs.tolist(),
            model=req.model,
            device=str(ACCELERATOR.device),
            dims=dims,
            normalized=req.normalize,
        )

    except Exception as e:
        gpu_error(f"Image embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))