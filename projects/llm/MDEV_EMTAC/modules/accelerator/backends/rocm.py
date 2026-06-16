from __future__ import annotations

from contextlib import nullcontext
from typing import Any
import torch

from ..base import AcceleratorBackend


class ROCmBackend(AcceleratorBackend):
    """
    Stub backend for AMD ROCm.

    Many ROCm installs are exposed via torch.cuda.* (with ROCm builds of PyTorch),
    but proper vendor detection / policy may differ.

    Implement when you need it. For now, it reports unavailable.
    """

    name = "rocm"

    def is_available(self) -> bool:
        return False

    def device(self) -> torch.device:
        return torch.device("cuda")  # Placeholder, ROCm builds often still use "cuda" device type.

    def device_count(self) -> int:
        return 0

    def prepare_model(self, model: Any) -> Any:
        return model

    def prepare_batch(self, batch: Any) -> Any:
        return batch

    def autocast_context(self):
        return nullcontext()
