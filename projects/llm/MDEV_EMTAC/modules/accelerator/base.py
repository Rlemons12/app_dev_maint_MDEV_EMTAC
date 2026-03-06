"""
Abstract backend interface.

Each backend must provide:
- availability check
- device handle
- model/batch preparation
- autocast context (safe no-op if not supported)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import torch


class AcceleratorBackend(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def device_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def prepare_model(self, model: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def prepare_batch(self, batch: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def autocast_context(self):
        raise NotImplementedError
