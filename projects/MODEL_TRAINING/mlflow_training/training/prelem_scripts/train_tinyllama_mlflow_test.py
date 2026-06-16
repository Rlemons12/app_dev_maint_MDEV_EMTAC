"""
TinyLlama MLflow Smoke Test (DB-backed, Offline-safe, GPU-aware)

Verifies:
- MLflow tracking via PostgreSQL (MLflowDatabaseConfig)
- MLflowLogManager integration
- Local TinyLlama model loading
- Tokenization + HF Trainer loop
- Optional GPU service delegation
"""

import os
import sys
from pathlib import Path

import torch
import mlflow
from sqlalchemy import text
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# 0. Environment bootstrap
# ---------------------------------------------------------------------
GLOBAL_ENV = Path(r"E:\emtac\dev_env\.env")
load_dotenv(GLOBAL_ENV)

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Prevent accidental EMTAC app DB usage
os.environ.setdefault("DATABASE_URL", "sqlite:///dummy.db")

# ---------------------------------------------------------------------
# 1. Ensure project imports work (no cwd dependence)
# ---------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------
# 2. Imports (MLflow + logging + GPU adapter)
# ---------------------------------------------------------------------
from configuration.config_env import MLflowDatabaseConfig
from configuration.log_config import MLflowLogManager
from modules.gpu.gpu_training_adapter import GPUTrainingAdapter

# ---------------------------------------------------------------------
# 3. Resolve REQUIRED paths from .env (FAIL FAST)
# ---------------------------------------------------------------------
def require_env_path(name: str, mkdir: bool = False) -> Path:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    path = Path(value).resolve()
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise FileNotFoundError(f"{name} path does not exist: {path}")
    return path


MODEL_PATH = require_env_path("MODELS_TINY_LLAMA_DIR")
LOGS_DIR = require_env_path("SMOKE_TESTS", mkdir=True)

# ---------------------------------------------------------------------
# 4. Configure MLflow (STRICT DB-backed)
# ---------------------------------------------------------------------
mlflow_db = MLflowDatabaseConfig()
mlflow_uri = mlflow_db.get_database_url()

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("TinyLlama_SmokeTest")

# Health check (logged later)
with mlflow_db.get_session() as session:
    exp_count = session.execute(
        text("SELECT COUNT(*) FROM experiments")
    ).scalar()

# ---------------------------------------------------------------------
# 5. GPU adapter (auto-detect)
# ---------------------------------------------------------------------
gpu_adapter = GPUTrainingAdapter()
USE_GPU_SERVICE = gpu_adapter.is_available()

# ---------------------------------------------------------------------
# 6. MLflow logger (isolated)
# ---------------------------------------------------------------------
passed = False

with MLflowLogManager(run_name="tinyllama_smoke", to_console=True) as mllog:
    mllog.logger.info("Starting TinyLlama MLflow smoke test")
    mllog.logger.info("Model path: %s", MODEL_PATH)
    mllog.logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    mllog.logger.info("MLflow experiments (count): %s", exp_count)
    mllog.logger.info(
        "GPU service available: %s",
        "YES" if USE_GPU_SERVICE else "NO (local execution)",
    )

    # -----------------------------------------------------------------
    # 7. Load model + tokenizer (local only)
    # -----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    mllog.logger.info("Using device: %s", device)

    # -----------------------------------------------------------------
    # 8. Dummy offline dataset
    # -----------------------------------------------------------------
    dataset = Dataset.from_dict(
        {
            "text": [
                "The quick brown fox jumps over the lazy dog.",
                "TinyLlama is a small but capable model.",
                "MLflow tracks experiments.",
            ]
        }
    )

    split = dataset.train_test_split(test_size=0.33, seed=42)

    def tokenize_fn(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=64,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_dataset = split["train"].map(
        tokenize_fn, batched=True, remove_columns=["text"]
    )
    eval_dataset = split["test"].map(
        tokenize_fn, batched=True, remove_columns=["text"]
    )

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # -----------------------------------------------------------------
    # 9. Training args
    # -----------------------------------------------------------------
    output_dir = LOGS_DIR / "results_tinyllama" / "run_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=2,
        num_train_epochs=0.01,
        learning_rate=5e-5,
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=1,
        save_steps=2,
        report_to=[],  # disable W&B / TensorBoard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # -----------------------------------------------------------------
    # 10. MLflow run
    # -----------------------------------------------------------------
    try:
        with mlflow.start_run(run_name="TinyLlama_SmokeTest_Run"):
            mlflow.log_param("model_path", str(MODEL_PATH))
            mlflow.log_param("device", device)
            mlflow.log_param("max_steps", training_args.max_steps)
            mlflow.log_param("tracking_uri", mlflow.get_tracking_uri())
            mlflow.log_param("gpu_service_used", USE_GPU_SERVICE)

            if hasattr(mllog, "log_run_start"):
                mllog.log_run_start(
                    "TinyLlama_SmokeTest",
                    "TinyLlama_SmokeTest_Run",
                )

            # -------------------------------------------------------------
            # GPU vs local execution
            # -------------------------------------------------------------
            if USE_GPU_SERVICE:
                mllog.logger.info("Delegating training to GPU service")

                payload = {
                    "model_path": str(MODEL_PATH),
                    "output_dir": str(output_dir),
                    "max_steps": training_args.max_steps,
                    "learning_rate": training_args.learning_rate,
                    "batch_size": training_args.per_device_train_batch_size,
                    "experiment": "TinyLlama_SmokeTest",
                    "run_name": "TinyLlama_SmokeTest_Run",
                }

                result = gpu_adapter.submit_training_job(payload)
                metrics = result.get("metrics", {})

            else:
                mllog.logger.info("Running local training loop")
                trainer.train()
                metrics = trainer.evaluate()

            # -------------------------------------------------------------
            # Log metrics + artifacts
            # -------------------------------------------------------------
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    mlflow.log_param(f"metric_{k}", str(v))

                if hasattr(mllog, "log_metric"):
                    mllog.log_metric(k, v)

            artifact_dir = output_dir / "tinyllama_smoke_artifact"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(artifact_dir)
            tokenizer.save_pretrained(artifact_dir)

            mlflow.log_artifacts(str(artifact_dir))
            if hasattr(mllog, "log_artifact"):
                mllog.log_artifact(str(artifact_dir))

            passed = True
            mllog.logger.info("Smoke test completed successfully.")

    except Exception as exc:
        passed = False
        if hasattr(mllog, "log_exception"):
            mllog.log_exception(exc)
        mllog.logger.exception("Smoke test failed")

# ---------------------------------------------------------------------
# 11. Exit status
# ---------------------------------------------------------------------
sys.exit(0 if passed else 1)
