from __future__ import annotations

import os
import json
from typing import Dict, List

from modules.configuration.log_config import logger
from modules.ai.config.models_config import ModelsConfig


# ============================================================
# CANONICAL IMAGE EMBEDDING REGISTRY
#
# - key          → canonical folder identity
# - name         → ModelsConfig identity (Admin selects this)
# - label        → Admin UI display label
# - dim          → embedding dimension (pgvector safety)
# - gpu_key      → GPU service registry key
#
# IMPORTANT:
# - Images ALWAYS execute via GPU service
# - Main app NEVER loads CLIP locally
# ============================================================

IMAGE_REGISTRY: Dict[str, Dict[str, object]] = {
    "openai_clip-vit-base-patch32": {
        "name": "CLIPModelHandler",
        "label": "CLIP ViT-B/32",
        "gpu_key": "clip",      # MUST match GPU service registry key
        "dim": 512,
        "folder_aliases": [
            "openai_clip-vit-base-patch32",
            "clip-vit-base-patch32",
        ],
    },
}


# ============================================================
# REGISTRATION LOGIC
# ============================================================

def sync_image_models_from_disk() -> None:
    """
    Discover image models under MODEL_PATH_IMAGE and register them
    into ModelsConfig.available_models (model_type='image').

    SAFETY GUARANTEES:
    - Discovery only (no model loading)
    - Never overrides EXECUTION_BACKEND
    - Non-destructive DB merge
    - Always forces backend='gpu_service'
    """

    base_path = os.getenv("MODEL_PATH_IMAGE")

    if not base_path:
        logger.warning("[IMAGE REGISTRY] MODEL_PATH_IMAGE not defined")
        return

    if not os.path.isdir(base_path):
        logger.warning(
            "[IMAGE REGISTRY] MODEL_PATH_IMAGE is not a directory: %s",
            base_path,
        )
        return

    logger.info("[IMAGE REGISTRY] Scanning image directory: %s", base_path)

    discovered: List[dict] = []

    # Normalize canonical registry for case-insensitive matching
    normalized_registry = {
        key.lower(): {
            **meta,
            "aliases": [
                key.lower(),
                *(a.lower() for a in meta.get("folder_aliases", [])),
            ],
        }
        for key, meta in IMAGE_REGISTRY.items()
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
                "[IMAGE REGISTRY] Unknown image folder skipped: %s",
                folder,
            )
            continue

        meta = normalized_registry[matched_key]

        # Structural validation (identity only)
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            logger.warning(
                "[IMAGE REGISTRY] Missing config.json for image model: %s",
                folder,
            )
            continue

        discovered.append({
            "name": meta["name"],
            "label": meta["label"],
            "backend": "gpu_service",   # Enforced
            "gpu_key": meta["gpu_key"],
            "enabled": True,
            "dim": meta["dim"],
        })

    if not discovered:
        logger.warning("[IMAGE REGISTRY] No image models discovered")
        return

    # --------------------------------------------------
    # Merge with existing DB configuration
    # --------------------------------------------------
    existing = ModelsConfig.get_available_models("image") or []
    existing_by_name = {
        m.get("name"): m for m in existing if isinstance(m, dict)
    }

    updated = list(existing)
    added = 0

    for model in discovered:
        name = model["name"]

        if name in existing_by_name:
            # Update GPU-safe metadata only
            existing_by_name[name].update({
                "label": model["label"],
                "gpu_key": model["gpu_key"],
                "dim": model["dim"],
                "enabled": True,
                "backend": "gpu_service",
            })
            continue

        updated.append(model)
        added += 1

        logger.info(
            "[IMAGE REGISTRY] Registered image model: %s",
            name,
        )

    ModelsConfig.set_config_value(
        model_type="image",
        key="available_models",
        value=json.dumps(updated),
    )

    logger.info(
        "[IMAGE REGISTRY] Sync complete (added=%d total=%d)",
        added,
        len(updated),
    )