import subprocess
import os
from pathlib import Path

# ------------------------------------------------------------
# Resolve project paths dynamically
# ------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]  # .../MODEL_TRAINING

VENV_PYTHON = PROJECT_ROOT / ".venv_model_training" / "Scripts" / "python.exe"
ARTIFACT_ROOT = PROJECT_ROOT / "logs" / "mlflow"

BACKEND_STORE_URI = (
    "postgresql+psycopg2://emtac_trainer:emtac_trainer123@localhost:5432/mlflow_training"
)

HOST = "0.0.0.0"   # change if needed
PORT = "5000"

# ------------------------------------------------------------


def start_mlflow_server():
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Venv python not found: {VENV_PYTHON}")

    os.chdir(PROJECT_ROOT)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    # ✅ IMPORTANT CHANGE: use `-m mlflow`, NOT `-m mlflow.server`
    cmd = [
        str(VENV_PYTHON),
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri", BACKEND_STORE_URI,
        "--default-artifact-root", str(ARTIFACT_ROOT),
        "--host", HOST,
        "--port", PORT,
    ]

    print("=" * 60)
    print("Starting MLflow server")
    print(f"Script location  : {SCRIPT_PATH}")
    print(f"Project root    : {PROJECT_ROOT}")
    print(f"Python executable: {VENV_PYTHON}")
    print(f"Artifact root   : {ARTIFACT_ROOT}")
    print(f"Bind address    : {HOST}:{PORT}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 60)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    start_mlflow_server()
