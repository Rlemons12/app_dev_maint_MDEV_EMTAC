"""
Policy decisions based on detected hardware.

This is where you decide:
- preferred precision (fp16/bf16/fp32)
- whether mixed precision should be enabled
- future: batch size hints, memory safety margins

Callers should not encode these rules directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from .detection import DeviceInfo


@dataclass(frozen=True)
class PrecisionPolicy:
    precision: str            # "fp32" | "fp16" | "bf16"
    enable_autocast: bool
    reason: str


def _cuda_policy(devices: List[DeviceInfo]) -> PrecisionPolicy:
    """
    Heuristic:
    - Default to fp16 on CUDA for speed, because Tesla T4 supports fp16 well.
    - bf16 is not consistently supported on older GPUs. Keep it conservative.
    """
    # If you want to get fancier later, you ca
