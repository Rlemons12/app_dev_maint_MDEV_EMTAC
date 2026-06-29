"""
config.py
Global pipeline configuration for the EMTAC Q&A Dataset Generator

This file stores:
- Paths (datasets, cache, model directories)
- Chunking settings
- Structure analyzer settings
- Q&A generation parameters
- Logging and debug flags

Database settings are handled in pg_db_config.py
"""

import os
from pathlib import Path

# ---------------------------------------------------------
# Resolve project root and dev_env
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # points to: E:\emtac\projects\llm
DEV_ENV_DIR = PROJECT_ROOT / "dev_env"


# ---------------------------------------------------------
# === PATH CONFIGURATION ===
# ---------------------------------------------------------

RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_documention"
OUTPUT_DATASET_DIR = PROJECT_ROOT / "training_data" / "datasets"
TEMP_DIR = PROJECT_ROOT / "temp_extraction"
MODEL_CACHE_DIR = PROJECT_ROOT / "models"

for d in [RAW_DOCS_DIR, OUTPUT_DATASET_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# === STRUCTURE ANALYZER SETTINGS ===
# ---------------------------------------------------------

STRUCTURE_CONFIG = {
    "max_chunk_words": 180,
    "min_chunk_words": 25,

    "include_images": True,
    "image_output_dir": PROJECT_ROOT / "training_data" / "extracted_images",

    "ocr_min_text_chars": 40,
    "force_ocr": False,

    "page_merge_threshold": 0.30,
    "heading_detection_enabled": True,

    "debug_save_markdown": True,
}

# Ensure image directory exists
STRUCTURE_CONFIG["image_output_dir"].mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# === Q&A GENERATION SETTINGS ===
# ---------------------------------------------------------

QNA_CONFIG = {
    "num_questions": 3,
    "temperature": 0.2,
    "max_retries": 5,

    "require_factual": True,
    "avoid_how_why": True,
    "short_questions": True,

    "ranked_mode": False,
    "passes_mode": False,
    "test_mode": False,
}


# ---------------------------------------------------------
# === MODEL CONFIGURATION ===
# (all local, no internet required)
# ---------------------------------------------------------

MODEL_CONFIG = {
    "qg_model_path": str(MODEL_CACHE_DIR / "t5-base-question-generator"),
    "embeddings_model_path": str(MODEL_CACHE_DIR / "MiniLM-L6-v2"),
}


# ---------------------------------------------------------
# === LOGGING SETTINGS ===
# ---------------------------------------------------------

LOGGING_CONFIG = {
    "save_logs": True,
    "verbose": True,
    "log_path": PROJECT_ROOT / "logs" / "dataset_gen.log",
}

LOGGING_CONFIG["log_path"].parent.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------
# === GENERAL PIPELINE FLAGS ===
# ---------------------------------------------------------

PIPELINE_FLAGS = {
    "offline_mode": True,
    "save_intermediate": True,
    "fail_gracefully": True,
}
