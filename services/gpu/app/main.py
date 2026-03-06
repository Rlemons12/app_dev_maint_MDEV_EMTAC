from fastapi import FastAPI

from app.config.settings import SERVICE_NAME, SERVICE_VERSION
from app.health import router as health_router
from app.api.embed import router as embed_router
from app.api.generate import router as generate_router
from app.api.dashboard import (
    router as dashboard_router,
    install_dashboard_background_tasks,
)
from app.runtime.accelerator import ACCELERATOR
from app.models.model_manager import GPU_MODELS
from app.config.gpu_logger import (gpu_info,
    install_dashboard_log_handler,)
from app.api.vision import router as vision_router
# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
)

# -------------------------------------------------
# Routers
# -------------------------------------------------
app.include_router(health_router)
app.include_router(embed_router)
app.include_router(generate_router)
app.include_router(dashboard_router)
app.include_router(vision_router)
# -------------------------------------------------
# Install dashboard background services
# MUST be called before startup executes
# -------------------------------------------------
from app.api.dashboard import LOG_HUB

def _on_loop_ready(loop):
    install_dashboard_log_handler(LOG_HUB, loop)
    gpu_info("Dashboard log handler installed")

install_dashboard_background_tasks(app, on_loop_ready=_on_loop_ready)


# -------------------------------------------------
# Startup
# -------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """
    Startup initialization for GPU service.
    """

    # -------------------------------------------------
    # 1️⃣ Install dashboard log streaming handler
    # -------------------------------------------------
    # Import here to ensure dashboard startup has already
    # captured the real uvicorn event loop.
    from app.api.dashboard import LOG_HUB, EVENT_LOOP

    gpu_info("Dashboard log handler installed")
    gpu_info(f"EVENT_LOOP is: {EVENT_LOOP}")
    # -------------------------------------------------
    # 2️⃣ Initialize accelerator
    # -------------------------------------------------
    accelerator_status = ACCELERATOR.status()

    gpu_info(
        "Accelerator initialized | "
        f"device={accelerator_status.get('device')} | "
        f"gpus={accelerator_status.get('gpu_count')} | "
        f"cuda_available={accelerator_status.get('cuda_available')}"
    )

    # -------------------------------------------------
    # 3️⃣ Confirm model configuration
    # -------------------------------------------------
    gpu_info("Models configured | lazy_load=True")

    gpu_info("GPU Service startup complete")


# -------------------------------------------------
# Shutdown
# -------------------------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    gpu_info("GPU Service shutting down")


# -------------------------------------------------
# Optional health endpoint
# -------------------------------------------------
@app.get("/health/models")
def model_health():
    return GPU_MODELS.status()