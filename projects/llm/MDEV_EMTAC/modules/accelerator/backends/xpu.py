from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from ..base import AcceleratorBackend


class XPUBackend(AcceleratorBackend):
    """
    Stub backend for Intel XPU.

    Intel GPU acceleration typically requires additional packages/extensions.
    Implement availability and device transfer when you adopt that stack.
    """

    name = "xpu"

    def is_available(self) -> bool:
        return False

    def device(self):
        # Placeholder; actual device type depends on Intel stack.
        # Keeping as string avoids importing optional libs here.
        return "xpu"

    def device_count(self) -> int:
        return 0

    def prepare_model(self, model: Any) -> Any:
        return model

    def prepare_batch(self, batch: Any) -> Any:
        return batch

    def autocast_context(self):
        return nullcontext()
