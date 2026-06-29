from __future__ import annotations

from typing import Any
import torch

from ..base import AcceleratorBackend


class CUDABackend(AcceleratorBackend):
    name = "cuda"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def device(self) -> torch.device:
        return torch.device("cuda")

    def device_count(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

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
        return torch.cuda.amp.autocast()
