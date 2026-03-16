import os
import json
from typing import Dict

from modules.configuration.log_config import logger
from modules.ai.config.models_config import ModelsConfig


# ============================================================
# CANONICAL EMBEDDING REGISTRY
#
# - key     → folder name (lowercase)
# - name    → ModelsConfig identity
# - label   → Admin UI label
# - dim     → embedding dimension (pgvector safety)
# - backend → REQUIRED (local only here)
# ============================================================

EMBEDDING_REGISTRY = {
    "all-minilm-l6-v2": {
        "name": "all-MiniLM-L6-v2",          # DB identity
        "label": "MiniLM L6 v2 (Fast)",
        "dim": 384,
        "gpu_key": "all-minilm-l6-v2",       # GPU SERVICE NAME ✅
    },
    "nomic-embed-text-v1.5": {
        "name": "nomic-embed-text-v1.5",
        "label": "Nomic Embed v1.5",
        "dim": 768,
        "gpu_key": "nomic-embed-text-v1.5",
    },
}



def sync_embedding_models_from_disk() -> None:
    """
    Discover embedding MODEL IDENTITIES from disk and register them
    for GPU execution in ModelsConfig.available_models (model_type='embedding').

    SAFETY GUARANTEES
    - Disk is used ONLY for discovery (identity + metadata)
    - Embeddings are ALWAYS executed via GPU service
    - Never registers local execution paths
    - Never overrides EXECUTION_BACKEND
    - Safe to call repeatedly
    """

    base_path = os.getenv("MODEL_PATH_EMBEDDING")

    if not base_path:
        logger.warning("[EMBEDDING REGISTRY] MODEL_PATH_EMBEDDING not set")
        return

    if not os.path.isdir(base_path):
        logger.warning(
            "[EMBEDDING REGISTRY] MODEL_PATH_EMBEDDING is not a directory: %s",
            base_path,
        )
        return

    logger.info("[EMBEDDING REGISTRY] Scanning embedding directory: %s", base_path)

    discovered = []

    # --------------------------------------------------
    # Normalize registry for folder matching
    # --------------------------------------------------
    normalized_registry = {
        key.lower(): meta
        for key, meta in EMBEDDING_REGISTRY.items()
    }

    for folder in os.listdir(base_path):
        model_dir = os.path.join(base_path, folder)

        if not os.path.isdir(model_dir):
            continue

        folder_key = folder.lower().strip()

        if folder_key not in normalized_registry:
            logger.debug(
                "[EMBEDDING REGISTRY] Skipping unknown embedding folder: %s",
                folder,
            )
            continue

        meta = normalized_registry[folder_key]

        # --------------------------------------------------
        # Structural validation (identity only)
        # --------------------------------------------------
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            logger.warning(
                "[EMBEDDING REGISTRY] Missing config.json for embedding model: %s",
                folder,
            )
            continue

        discovered.append({
            "name": meta["name"],  # all-MiniLM-L6-v2 (UI / DB identity)
            "label": meta["label"],
            "backend": "gpu_service",
            "gpu_key": "minilm",  # ✅ MUST match GPU registry
            "enabled": True,
            "dim": meta["dim"],
        })

    if not discovered:
        logger.warning("[EMBEDDING REGISTRY] No embedding models discovered")
        return

    # --------------------------------------------------
    # Merge with existing DB configuration (NON-DESTRUCTIVE)
    # --------------------------------------------------
    existing = ModelsConfig.get_available_models("embedding") or []

    existing_by_name = {
        m.get("name"): m for m in existing if isinstance(m, dict)
    }

    updated = list(existing)
    added = 0

    for model in discovered:
        name = model["name"]

        if name in existing_by_name:
            # Update ONLY GPU-safe fields
            existing_by_name[name].update({
                "label": model["label"],
                "backend": "gpu_service",
                "gpu_key": model["gpu_key"],
                "dim": model["dim"],
                "enabled": True,
            })
            continue

        updated.append(model)
        added += 1

        logger.info(
            "[EMBEDDING REGISTRY] Registered embedding model: %s",
            name,
        )

    ModelsConfig.set_config_value(
        model_type="embedding",
        key="available_models",
        value=json.dumps(updated),
    )

    logger.info(
        "[EMBEDDING REGISTRY] Embedding registry sync complete "
        "(added=%d total=%d)",
        added,
        len(updated),
    )
