# app/main.py
from __future__ import annotations

from fastapi import FastAPI

from app.api.generate import router as generate_router
from app.api.embed import router as embed_router
from app.api.train import router as train_router

from app.config.settings import SERVICE_NAME, SERVICE_VERSION
from app.config.gpu_logger import gpu_info


def create_app() -> FastAPI:
    app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)

    app.include_router(generate_router)  # existing :contentReference[oaicite:6]{index=6}
    app.include_router(embed_router)      # existing :contentReference[oaicite:7]{index=7}
    app.include_router(train_router)      # new

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


app = create_app()
gpu_info("GPU training service booted")
