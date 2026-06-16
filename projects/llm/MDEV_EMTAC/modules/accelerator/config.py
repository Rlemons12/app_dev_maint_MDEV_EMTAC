"""
Configuration for accelerator selection and behavior.

Environment variables:
- EMTAC_ACCELERATOR:
    "auto" (default), "cuda", "mps", "cpu", "rocm", "xpu"
- EMTAC_CUDA_VISIBLE_DEVICES:
    If set, applied to CUDA device visibility (e.g. "0,1")
- EMTAC_ACCELERATOR_LOG_LEVEL:
    Optional override for accelerator package logger level (e.g. "INFO", "DEBUG")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AcceleratorConfig:
    backend_priority: List[str]
    force_backend: str
    cuda_visible_devices: str | None
    log_level: str | None


DEFAULT_PRIORITY = ["cuda", "mps", "rocm", "xpu", "cpu"]


def load_accelerator_config() -> AcceleratorConfig:
    force_backend = os.getenv("EMTAC_ACCELERATOR", "auto").strip().lower()
    cuda_visible_devices = os.getenv("EMTAC_CUDA_VISIBLE_DEVICES")
    log_level = os.getenv("EMTAC_ACCELERATOR_LOG_LEVEL")

    return AcceleratorConfig(
        backend_priority=list(DEFAULT_PRIORITY),
        force_backend=force_backend,
        cuda_visible_devices=cuda_visible_devices.strip() if cuda_visible_devices else None,
        log_level=log_level.strip().upper() if log_level else None,
    )
