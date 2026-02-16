from pathlib import Path
from dotenv import load_dotenv
import uvicorn
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

# --------------------------------------------------
# Load shared EMTAC environment
# --------------------------------------------------
ENV_PATH = Path(r"E:\emtac\dev_env\.env")
load_dotenv(ENV_PATH, override=False)

# --------------------------------------------------
# GPU service logging (NEW)
# --------------------------------------------------
from app.config.gpu_log_config import LOGGING_CONFIG

# --------------------------------------------------
# Run GPU service
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5050,
        log_config=LOGGING_CONFIG,  # ✅ GPU-safe logging
        access_log=True,            # ✅ uvicorn.access preserved
    )
