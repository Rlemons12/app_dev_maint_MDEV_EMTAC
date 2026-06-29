# app/health.py

from fastapi import APIRouter
from app.runtime.accelerator import ACCELERATOR
from app.models.model_manager import GPU_MODELS

router = APIRouter(tags=["health"])


@router.get("/health")
def gpu_health():
    """
    Primary service health endpoint.

    Guarantees:
    - No model loading
    - No GPU allocation
    - Safe to call frequently
    """

    return {
        "status": "ok",
        "accelerator": {
            "runtime": ACCELERATOR.status(),
            "roles": {
                "generation": str(
                    ACCELERATOR.device_for("generation")
                ) if ACCELERATOR.is_gpu else "cpu",
                "embedding": str(
                    ACCELERATOR.device_for("embedding")
                ) if ACCELERATOR.is_gpu else "cpu",
            },
        },
        "models": GPU_MODELS.status(),
    }
