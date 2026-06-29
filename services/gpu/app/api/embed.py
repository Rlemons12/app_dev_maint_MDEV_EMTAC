#    services\gpu\app\api\embed.py


from __future__ import annotations

from fastapi import APIRouter, HTTPException
import torch

from app.schemas.embed import EmbedRequest, EmbedResponse
from app.models.embeddings import EMBEDDINGS
from app.runtime.accelerator import ACCELERATOR

router = APIRouter(prefix="/embed", tags=["embedding"])


@router.post("", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    try:
        model, model_key = EMBEDDINGS.get(req.model)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding model load failed: {e}")

    try:
        with torch.no_grad():
            vecs = model.encode(
                req.texts,
                batch_size=req.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=req.normalize,
                show_progress_bar=False,
            )

        dims = int(vecs.shape[1]) if hasattr(vecs, "shape") else (len(vecs[0]) if vecs else 0)

        return EmbedResponse(
            embeddings=vecs.tolist(),
            model=model_key,
            device=str(ACCELERATOR.device),
            dims=dims,
            normalized=bool(req.normalize),
            meta={"cuda_available": bool(torch.cuda.is_available())},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding encode failed: {e}")
