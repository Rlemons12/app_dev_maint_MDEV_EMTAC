from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

# --------------------------------------------------
# Load shared EMTAC environment
# --------------------------------------------------
DEFAULT_ENV_PATH = Path(r"E:\emtac\dev_env\.env")
ENV_PATH = Path(os.getenv("EMTAC_ENV_PATH", str(DEFAULT_ENV_PATH)))

if not ENV_PATH.exists():
    raise FileNotFoundError(f"Environment file not found: {ENV_PATH}")

load_dotenv(ENV_PATH, override=False)

# --------------------------------------------------
# GPU service bind config
# --------------------------------------------------
GPU_SERVICE_HOST = os.getenv("GPU_SERVICE_HOST", "0.0.0.0")
GPU_SERVICE_PORT = int(os.getenv("GPU_SERVICE_PORT", "5051"))

# --------------------------------------------------
# GPU service logging
# --------------------------------------------------
from app.config.gpu_log_config import LOGGING_CONFIG

# --------------------------------------------------
# Run GPU service
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=GPU_SERVICE_HOST,
        port=GPU_SERVICE_PORT,
        log_config=LOGGING_CONFIG,
        access_log=True,
    )