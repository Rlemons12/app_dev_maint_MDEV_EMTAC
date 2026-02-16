import os
from pathlib import Path

# -------------------------------------------------
# Base paths (resolved safely)
# -------------------------------------------------
BASE_DIR = Path(os.getenv("MODEL_TRAINING_BASE", r"E:\emtac\projects\MODEL_TRAINING"))

CONFIG_DIR = BASE_DIR / "configuration"
MLFLOW_DIR = BASE_DIR / "mlflow"
LOGS_DIR = BASE_DIR / "logs"

# -------------------------------------------------
# MLflow logging
# -------------------------------------------------
MLFLOW_LOGS_DIR = LOGS_DIR / "mlflow"
MLFLOW_ARTIFACT_DIR = MLFLOW_DIR / "artifacts"
MLFLOW_RUNS_DIR = MLFLOW_DIR / "runs"

# Ensure directories exist (safe on import)
for _p in (
    MLFLOW_DIR,
    LOGS_DIR,
    MLFLOW_LOGS_DIR,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_RUNS_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Optional helpers (legacy-safe)
# -------------------------------------------------
BASE_DIR_STR = str(BASE_DIR)
MLFLOW_DIR_STR = str(MLFLOW_DIR)
MLFLOW_LOGS_DIR_STR = str(MLFLOW_LOGS_DIR)
GPU_ADAPTER=r"E:\emtac\projects\MODEL_TRAINING\modules\gpu\gpu_training_adapter.py"