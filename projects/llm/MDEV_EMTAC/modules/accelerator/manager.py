"""
Accelerator manager: selects backend, exposes stable API.

Callers should use:
    from modules.emtac_ai.accelerator import ACCELERATOR

Key design goals:
- Hardware-agnostic caller code
- Safe CPU fallback
- Extensible backends
- Good diagnostics/logging
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from .config import load_accelerator_config
from .detection import detect_all_devices, DeviceInfo
from .diagnostics import build_summary, format_summary_text
from .policies import decide_precision_policy, PrecisionPolicy

from .backends.cpu import CPUBackend
from .backends.cuda import CUDABackend
from .backends.mps import MPSBackend
from .backends.rocm import ROCmBackend
from .backends.xpu import XPUBackend

logger = logging.getLogger("ematac_logger")


class AcceleratorManager:
    def __init__(self):
        self.cfg = load_accelerator_config()
        self._apply_process_env()

        self.detected_devices = detect_all_devices()
        self.backend = self._select_backend()

        self.precision_policy: PrecisionPolicy = decide_precision_policy(
            selected_backend=self.backend.name,
            detected_devices=self.detected_devices,
        )

        self._summary_cache: Dict[str, Any] = {}
        self._log_startup_summary()

    def _apply_process_env(self) -> None:
        # If you want to constrain CUDA devices per process, this must be set
        # before torch first enumerates GPUs. Setting here is usually early enough
        # if ACCELERATOR is imported near process start.
        if self.cfg.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.cuda_visible_devices

        if self.cfg.log_level:
            try:
                level = getattr(logging, self.cfg.log_level, None)
                if isinstance(level, int):
                    logger.setLevel(level)
            except Exception:
                # Keep existing logger configuration
                pass

    def _backend_registry(self):
        return {
            "cpu": CPUBackend(),
            "cuda": CUDABackend(),
            "mps": MPSBackend(),
            "rocm": ROCmBackend(),
            "xpu": XPUBackend(),
        }

    def _select_backend(self):
        registry = self._backend_registry()

        force = (self.cfg.force_backend or "auto").lower()
        if force != "auto":
            backend = registry.get(force)
            if backend and backend.is_available():
                return backend
            logger.warning(
                "[ACCELERATOR] Forced backend '%s' unavailable; falling back to auto selection",
                force,
            )

        for name in self.cfg.backend_priority:
            backend = registry.get(name)
            if backend and backend.is_available():
                return backend

        return registry["cpu"]

    def _log_startup_summary(self) -> None:
        summary = self.summary()
        # Summary is safe to log at INFO; details are valuable on servers.
        logger.info("[ACCELERATOR]\n%s", format_summary_text(summary))

    # ---------------------------
    # Public API
    # ---------------------------

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def enabled(self) -> bool:
        return self.backend.name != "cpu"

    @property
    def device(self):
        return self.backend.device()

    @property
    def count(self) -> int:
        try:
            return int(self.backend.device_count())
        except Exception:
            return 0

    @property
    def preferred_precision(self) -> str:
        return self.precision_policy.precision

    @property
    def enable_autocast(self) -> bool:
        return bool(self.precision_policy.enable_autocast)

    def prepare_model(self, model: Any) -> Any:
        return self.backend.prepare_model(model)

    def prepare_batch(self, batch: Any) -> Any:
        return self.backend.prepare_batch(batch)

    def autocast_context(self):
        if self.enable_autocast:
            return self.backend.autocast_context()
        # If policy disables autocast, return a no-op context from CPU backend.
        return CPUBackend().autocast_context()

    def summary(self) -> Dict[str, Any]:
        # Cache once; hardware doesn’t change at runtime per process.
        if self._summary_cache:
            return self._summary_cache

        device_str = str(self.device)
        summary = build_summary(
            selected_backend=self.name,
            device_str=device_str,
            device_count=self.count,
            detected_devices=self.detected_devices,
            precision_policy=self.precision_policy,
        )
        self._summary_cache = summary
        return summary

    def detected(self) -> list[DeviceInfo]:
        return list(self.detected_devices)


# Singleton instance
ACCELERATOR = AcceleratorManager()
