from __future__ import annotations

from contextlib import nullcontext
from typing import Any
import torch

from ..base import AcceleratorBackend


class MPSBackend(AcceleratorBackend):
    name = "mps"

    def is_available(self) -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def device(self) -> torch.device:
        return torch.device("mps")

    def device_count(self) -> int:
        return 1 if self.is_available() else 0

    def prepare_model(self, model: Any) -> Any:
        return model.to(self.device())

    def prepare_batch(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device())

        if isinstance(batch, dict):
            return {k: self.prepare_batch(v) for k, v in batch.items()}

        if isinstance(batch, (list, tuple)):
            return type(batch)(self.prepare_batch(v) for v in batch)

        return batch

    def autocast_context(self):
        # MPS autocast is not universally safe across workflows; keep no-op by default.
        return nullcontext()
