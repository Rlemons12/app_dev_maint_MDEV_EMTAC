"""
model_resolver_config.py

Centralized configuration utility for resolving
the active trained model version automatically.

Behavior:
- Reads MODEL_TRAINED from .env
- Scans for folders ending in _vXX
- Selects highest version
- Exposes:
    • model_name
    • model_path
    • version
    • root
- Lazy singleton (safe import anywhere)
- Supports cache refresh (used during promotion)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any

VERSION_PATTERN = re.compile(r"_v(\d{1,4})$", re.IGNORECASE)


# ==========================================================
# Core Resolver
# ==========================================================

class ModelResolverConfig:
    """
    Resolves latest trained model from MODEL_TRAINED directory.
    """

    def __init__(self) -> None:
        self.model_root = os.getenv("MODEL_TRAINED", "").strip()
        self.auto_select = (
            os.getenv("AUTO_SELECT_LATEST_MODEL", "true").strip().lower() == "true"
        )

        if not self.model_root:
            raise RuntimeError("MODEL_TRAINED is not defined in .env")

        root_path = Path(self.model_root)

        if not root_path.exists():
            raise FileNotFoundError(
                f"MODEL_TRAINED path does not exist: {self.model_root}"
            )

        if not root_path.is_dir():
            raise RuntimeError(
                f"MODEL_TRAINED is not a directory: {self.model_root}"
            )

        if not self.auto_select:
            raise RuntimeError(
                "AUTO_SELECT_LATEST_MODEL is disabled. "
                "Static model selection not implemented."
            )

        self._resolve()

    # ------------------------------------------------------
    # Resolution
    # ------------------------------------------------------

    def _resolve(self) -> None:
        """
        Resolve latest model and populate public fields.
        """
        path, version = self._find_latest()

        self.active_model_path: str = path
        self.active_version: int = version
        self.active_model_name: str = Path(path).name

    def refresh(self) -> None:
        """
        Force re-scan of MODEL_TRAINED directory.
        Used after new training or promotion.
        """
        self._resolve()

    # ------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------

    def _extract_version(self, name: str) -> Optional[int]:
        """
        Extract version number from folder name ending in _vXX

        Example:
            emtac_mistral_sft_v21 -> 21
        """
        match = VERSION_PATTERN.search(name)
        if match:
            return int(match.group(1))
        return None

    def _find_latest(self) -> tuple[str, int]:
        """
        Scan MODEL_TRAINED directory and return:
            (full_path, version_number)
        """

        root = Path(self.model_root)

        highest_version = -1
        latest_path: Optional[Path] = None

        for item in root.iterdir():
            if not item.is_dir():
                continue

            version = self._extract_version(item.name)
            if version is None:
                continue

            if version > highest_version:
                highest_version = version
                latest_path = item.resolve()

        if latest_path is None:
            raise RuntimeError(
                f"No versioned model folders found in {self.model_root}. "
                f"Expected folders ending in _vXX."
            )

        return str(latest_path), highest_version


# ==========================================================
# Lazy Singleton
# ==========================================================

_model_resolver_singleton: Optional[ModelResolverConfig] = None


def get_model_resolver_config(refresh: bool = False) -> ModelResolverConfig:
    """
    Returns singleton instance.

    If refresh=True → forces re-scan.
    """
    global _model_resolver_singleton

    if _model_resolver_singleton is None:
        _model_resolver_singleton = ModelResolverConfig()
    elif refresh:
        _model_resolver_singleton.refresh()

    return _model_resolver_singleton


# ==========================================================
# Public API
# ==========================================================

def resolve_latest_trained_model(refresh: bool = False) -> Dict[str, Any]:
    """
    Returns:

    {
        "model_name": "emtac_mistral_sft_v21",
        "model_path": "E:\\emtac\\models\\llm\\trained_models\\emtac_mistral_sft_v21",
        "version": 21,
        "root": "E:\\emtac\\models\\llm\\trained_models"
    }
    """

    cfg = get_model_resolver_config(refresh=refresh)

    return {
        "model_name": cfg.active_model_name,
        "model_path": cfg.active_model_path,
        "version": cfg.active_version,
        "root": cfg.model_root,
    }


# ==========================================================
# Backwards Compatibility Export
# ==========================================================

# NOTE:
# We DO NOT instantiate eagerly.
# This prevents import-time crashes if .env isn't loaded yet.
model_resolver_config: Optional[ModelResolverConfig] = None
