from fastapi import FastAPI

from app.config.settings import SERVICE_NAME, SERVICE_VERSION
from app.health import router as health_router
from app.api.embed import router as embed_router
from app.api.generate import router as generate_router
from app.runtime.accelerator import ACCELERATOR
from app.models.model_manager import GPU_MODELS
from app.config.gpu_logger import gpu_info

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

# -------------------------------------------------
# Startup
# -------------------------------------------------
@app.on_event("startup")
async def startup_event():
    accelerator_status = ACCELERATOR.status()

    gpu_info(
        "Accelerator initialized | "
        f"device={accelerator_status.get('device')} | "
        f"gpus={accelerator_status.get('gpu_count')} | "
        f"cuda_available={accelerator_status.get('cuda_available')}"
    )

    gpu_info("Models configured | lazy_load=True")

# -------------------------------------------------
# Optional inline health endpoints
# -------------------------------------------------
@app.get("/health/models")
def model_health():
    return GPU_MODELS.status()
