from __future__ import annotations

import os
import json
from typing import Dict, List

from modules.configuration.log_config import logger
from modules.configuration import config
from modules.ai.config.models_config import ModelsConfig


# ============================================================
# CANONICAL LLM REGISTRY (APP-LEVEL)
#
# - name        → what ModelsConfig stores & Admin selects
# - gpu_key     → what GPU service expects
# - folder key  → must exist under MODEL_PATH_LLM
# ============================================================

LLM_REGISTRY: Dict[str, Dict[str, int | str | list]] = {
    "tinyllama": {
        "name": "TinyLlamaModel",
        "label": "TinyLlama 1.1B (Fast)",
        "gpu_key": "tinyllama",
        "context_window": 2048,
        "folder_aliases": ["tinyllama_1_1b"],
    },
    "mistral": {
        "name": "MistralModel",
        "label": "Mistral 7B (High Quality)",
        "gpu_key": "mistral",
        "context_window": 8192,
        "folder_aliases": ["mistral_7b_v03"],
    },
    "qwen": {
        "name": "QwenModel",
        "label": "Qwen 2.5 (Instruction)",
        "gpu_key": "qwen",
        "context_window": 8192,
        "folder_aliases": ["qwen2.5-3b-instruct"],
    },
    "gemma": {
        "name": "GemmaModel",
        "label": "Gemma 2 (Google)",
        "gpu_key": "gemma",
        "context_window": 8192,
        "folder_aliases": ["google_gemma-2-2b-it"],
    },
    "openelm": {
        "name": "OpenELMModel",
        "label": "Apple OpenELM 1.1B",
        "gpu_key": "openelm",
        "context_window": 2048,
        "folder_aliases": ["apple_openelm-1_1b-instruct"],
    },
    "flan": {
        "name": "FlanT5Model",
        "label": "FLAN-T5 Large",
        "gpu_key": "flan",
        "context_window": 2048,
        "folder_aliases": ["flan_t5_large"],
    },
}



# ============================================================
# REGISTRATION LOGIC
# ============================================================

def sync_ai_models_from_disk() -> None:
    """
    Scan MODEL_PATH_LLM and register discovered LLM models
    into ModelsConfig.available_models (model_type='ai').

    - Safe to call multiple times
    - Never removes DB entries
    - Adds new models only
    """

    base_path = os.getenv("MODEL_PATH_LLM")

    if not base_path:
        logger.warning("[MODEL REGISTRY] MODEL_PATH_LLM not defined in environment")
        return

    if not os.path.isdir(base_path):
        logger.warning(
            "[MODEL REGISTRY] MODEL_PATH_LLM is not a directory: %s",
            base_path,
        )
        return

    logger.info("[MODEL REGISTRY] Scanning LLM directory: %s", base_path)

    discovered: List[dict] = []

    # --------------------------------------------------
    # Normalize registry for case-insensitive matching
    # --------------------------------------------------
    normalized_registry = {
        key.lower(): {
            **meta,
            "aliases": [
                key.lower(),
                *(a.lower() for a in meta.get("folder_aliases", [])),
            ],
        }
        for key, meta in LLM_REGISTRY.items()
    }

    for folder in os.listdir(base_path):
        model_dir = os.path.join(base_path, folder)

        if not os.path.isdir(model_dir):
            continue

        folder_key = folder.lower().strip()
        matched_key = None

        for reg_key, meta in normalized_registry.items():
            if folder_key in meta["aliases"]:
                matched_key = reg_key
                break

        if not matched_key:
            logger.debug(
                "[MODEL REGISTRY] Skipping unknown model folder: %s",
                folder,
            )
            continue

        meta = normalized_registry[matched_key]

        discovered.append({
            "name": meta["name"],                 # App-level model identity
            "label": meta["label"],               # Admin UI label
            "backend": "gpu_service",
            "gpu_key": meta["gpu_key"],            # GPU routing key
            "enabled": True,
            "context_window": meta["context_window"],
            "path": model_dir,
        })

    if not discovered:
        logger.warning("[MODEL REGISTRY] No known LLM models discovered")
        return

    # --------------------------------------------------
    # Merge with existing DB configuration
    # --------------------------------------------------
    existing = ModelsConfig.get_available_models("ai") or []
    existing_by_name = {
        m.get("name"): m for m in existing if isinstance(m, dict)
    }

    updated = list(existing)
    added = 0

    for model in discovered:
        if model["name"] in existing_by_name:
            # Update metadata in-place (path changes, etc.)
            existing_by_name[model["name"]].update({
                "label": model["label"],
                "gpu_key": model["gpu_key"],
                "context_window": model["context_window"],
                "path": model["path"],
            })
            continue

        updated.append(model)
        added += 1
        logger.info(
            "[MODEL REGISTRY] Registered AI model: %s",
            model["name"],
        )

    ModelsConfig.set_config_value(
        model_type="ai",
        key="available_models",
        value=json.dumps(updated),
    )

    logger.info(
        "[MODEL REGISTRY] AI model registration complete "
        "(added=%d total=%d)",
        added,
        len(updated),
    )

