from __future__ import annotations

from contextlib import nullcontext
from typing import Any
import torch

from ..base import AcceleratorBackend


class CPUBackend(AcceleratorBackend):
    name = "cpu"

    def is_available(self) -> bool:
        return True

    def device(self) -> torch.device:
        return torch.device("cpu")

    def device_count(self) -> int:
        return 0

    def prepare_model(self, model: Any) -> Any:
        try:
            return model.to(self.device())
        except Exception:
            return model

    def prepare_batch(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device())

        if isinstance(batch, dict):
            return {k: self.prepare_batch(v) for k, v in batch.items()}

        if isinstance(batch, (list, tuple)):
            return type(batch)(self.prepare_batch(v) for v in batch)

        return batch

    def autocast_context(self):
        return nullcontext()
