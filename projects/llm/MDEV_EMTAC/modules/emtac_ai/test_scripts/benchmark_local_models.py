#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark locally-available, MLflow-registered LLMs.
- Loads ONLY models found in the MLflow Model Registry.
- Assumes each model's local files live at: {MODELS_LLM_DIR}/{registered_model_name}
- Forces offline operation (no Hub calls).
- Logs per-model results to CSV and MLflow.
"""

import os
import sys
import time
import csv
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# --- Environment / MLflow ---
from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# --- Inference stack ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================================================
# USER-CONFIGURABLE INPUTS
# ==================================================
import builtins

def ask_user(prompt_text: str, default=None, cast_func=str):
    if default is not None:
        user_input = input(f"{prompt_text} [{default}]: ").strip()
    else:
        user_input = input(f"{prompt_text}: ").strip()
    if user_input == "":
        return default
    try:
        return cast_func(user_input)
    except Exception:
        print(f"Invalid input, using default: {default}")
        return default


print("\n=== MODEL BENCHMARK CONFIGURATION ===")

TEST_PROMPT = ask_user(
    "Enter benchmark prompt",
    default=("Explain how a pneumatic cylinder works in an industrial machine. "
             "Focus on ports, seals, double-acting operation, and failure modes."),
    cast_func=str
)

MAX_TOKENS = ask_user("Maximum new tokens per model", default=256, cast_func=int)

print("\nChoose generation mode:")
print("  1 = Greedy only (deterministic)")
print("  2 = Sampling only (stochastic)")
print("  3 = Both (run each model twice)")
mode_choice = ask_user("Select mode (1/2/3)", default="1", cast_func=str)

RUN_GREEDY = mode_choice in ["1", "3"]
RUN_SAMPLING = mode_choice in ["2", "3"]

TEMPERATURE = ask_user("Sampling temperature", default=0.7, cast_func=float)
TOP_P = ask_user("Top-p sampling cutoff", default=0.95, cast_func=float)
SAVE_OUTPUT_ARTIFACT = ask_user(
    "Save model outputs as artifact files? (True/False)",
    default=True,
    cast_func=lambda x: x.lower() in ["true", "t", "1"]
)

print("\n[INFO] Configuration Summary:")
print(f"  PROMPT: {TEST_PROMPT[:60]}{'...' if len(TEST_PROMPT) > 60 else ''}")
print(f"  MAX_TOKENS: {MAX_TOKENS}")
print(f"  RUN_GREEDY: {RUN_GREEDY}")
print(f"  RUN_SAMPLING: {RUN_SAMPLING}")
if RUN_SAMPLING:
    print(f"  TEMPERATURE: {TEMPERATURE}")
    print(f"  TOP_P: {TOP_P}")
print(f"  SAVE_OUTPUT_ARTIFACT: {SAVE_OUTPUT_ARTIFACT}")
print("======================================\n")


# ==================================================
# ENVIRONMENT & DEFAULTS
# ==================================================
dotenv_path = os.getenv("DOTENV_PATH", os.path.join("E:", "emtac", ".env"))
load_dotenv(dotenv_path)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "postgresql://postgres:emtac123@localhost:5432/mlflowdb")
MLFLOW_ARTIFACT_DIR = os.getenv("MLFLOW_ARTIFACT_DIR", r"E:\emtac\logs\smoke_tests\mlartifacts")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_BENCH_EXPERIMENT", "model_benchmarks")

MODELS_LLM_DIR = os.getenv("MODELS_LLM_DIR", r"E:\emtac\models\llm")
CSV_PATH = os.getenv("MLFLOW_BENCH_CSV", r"E:\emtac\logs\smoke_tests\mlflow_model_benchmarks.csv")
LOG_FILE = os.getenv("MLFLOW_BENCH_LOG", r"E:\emtac\logs\smoke_tests\benchmark_runs.log")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

Path(MLFLOW_ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(CSV_PATH)).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(LOG_FILE)).mkdir(parents=True, exist_ok=True)


# ==================================================
# LOGGING SETUP
# ==================================================
logger = logging.getLogger("bench_models")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)


# ==================================================
# UTILITIES
# ==================================================
def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def find_local_model_dir(models_root: Path, model_name: str) -> Optional[Path]:
    candidate = models_root / model_name
    if not candidate.exists() or not candidate.is_dir():
        return None
    has_files = any(
        p.name in ("config.json", "tokenizer.json") or p.suffix == ".safetensors"
        for p in candidate.glob("**/*")
    )
    return candidate if has_files else None

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def ensure_csv_header(csv_path: str, header: List[str]) -> None:
    if not Path(csv_path).exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def append_csv(csv_path: str, row: List[Any]) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ==================================================
# MLflow SETUP
# ==================================================
def setup_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        exp_id = mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=Path(MLFLOW_ARTIFACT_DIR).as_uri()
        )
        logger.info(f"Created MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' (id={exp_id})")
    else:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient()
    logger.info(f"Connected to MLflow: {MLFLOW_TRACKING_URI}")
    return client

def get_registered_model_names(client: MlflowClient) -> List[str]:
    try:
        rms = client.search_registered_models()
        names = [rm.name for rm in rms]
        logger.info(f"Found {len(names)} registered models.")
        return names
    except MlflowException as e:
        logger.error(f"Failed to query registry: {e}")
        return []


# ==================================================
# INFERENCE
# ==================================================
def benchmark_model(model_name: str, model_dir: Path, device: str, dtype: torch.dtype):
    results = []

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir.as_posix(),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir.as_posix(),
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    ).to(device)

    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(device)
    input_tokens = count_tokens(tokenizer, TEST_PROMPT)

    def run_generation(do_sample: bool):
        gen_cfg = {
            "max_new_tokens": MAX_TOKENS,
            "do_sample": do_sample,
            "temperature": TEMPERATURE if do_sample else None,
            "top_p": TOP_P if do_sample else None,
            "use_cache": True,
        }
        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_cfg)
        duration = time.time() - start
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_tokens = max(len(output_ids[0]) - input_tokens, 0)
        throughput = (new_tokens / duration) if duration > 0 else 0.0
        return text, duration, new_tokens, throughput

    if RUN_GREEDY:
        text, duration, new_tokens, thr = run_generation(do_sample=False)
        results.append(("greedy", text, duration, new_tokens, thr))
    if RUN_SAMPLING:
        text, duration, new_tokens, thr = run_generation(do_sample=True)
        results.append(("sampling", text, duration, new_tokens, thr))
    return results


# ==================================================
# MAIN
# ==================================================
def main() -> int:
    logger.info(f"Artifact dir: {Path(MLFLOW_ARTIFACT_DIR).as_uri()}")
    logger.info(f"Models root: {MODELS_LLM_DIR}")
    logger.info(f"CSV path: {CSV_PATH}")

    client = setup_mlflow()
    registered = get_registered_model_names(client)
    if not registered:
        logger.error("No registered models found. Exiting.")
        return 2

    models_root = Path(MODELS_LLM_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    logger.info(f"Using device={device}, dtype={dtype}")

    ensure_csv_header(CSV_PATH, [
        "timestamp", "model_name", "mode", "device", "dtype",
        "new_tokens", "throughput_toks_per_s", "generate_s"
    ])

    for model_name in registered:
        model_dir = find_local_model_dir(models_root, model_name)
        if model_dir is None:
            logger.warning(f"Skipping '{model_name}': local folder not found or missing HF files.")
            continue

        logger.info(f"Benchmarking '{model_name}' from {model_dir} ...")
        try:
            mode_results = benchmark_model(model_name, model_dir, device, dtype)
        except Exception as e:
            logger.exception(f"Failed while benchmarking '{model_name}': {e}")
            continue

        for mode, text, gen_time, new_tokens, thr in mode_results:
            timestamp = now_iso()
            append_csv(CSV_PATH, [
                timestamp, model_name, mode, device, str(dtype),
                new_tokens, round(thr, 2), round(gen_time, 4)
            ])

            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"bench_{model_name}_{mode}"):
                mlflow.log_params({
                    "model_name": model_name,
                    "mode": mode,
                    "device": device,
                    "dtype": str(dtype),
                    "max_new_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE if mode == "sampling" else 0.0,
                    "top_p": TOP_P if mode == "sampling" else 0.0,
                })
                mlflow.log_metrics({
                    "generate_s": round(gen_time, 4),
                    "new_tokens": new_tokens,
                    "throughput_toks_per_s": round(thr, 2),
                })

                if SAVE_OUTPUT_ARTIFACT:
                    out_path = Path(f"{model_name}_{mode}_output.txt")
                    with out_path.open("w", encoding="utf-8") as f:
                        f.write("=== PROMPT ===\n")
                        f.write(TEST_PROMPT + "\n\n=== OUTPUT ===\n" + text)
                    mlflow.log_artifact(out_path.as_posix())

            logger.info(f"Done: {model_name} [{mode}] | gen={gen_time:.2f}s, new_tok={new_tokens}, thr={thr:.2f} tok/s")

    logger.info("Benchmarking complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
