"""
Hardware detection utilities.

This module does read-only detection of available accelerators and returns
structured device info.

Supported now:
- CUDA via torch.cuda
- MPS via torch.backends.mps (Apple Silicon)
- CPU fallback

ROCm / XPU:
- Included as stubs (availability checks can be expanded later)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import platform

import torch


@dataclass(frozen=True)
class DeviceInfo:
    backend: str               # "cuda" | "mps" | "rocm" | "xpu" | "cpu"
    vendor: str                # "NVIDIA" | "Apple" | "AMD" | "Intel" | "CPU"
    name: str                  # Device name
    index: Optional[int] = None
    total_memory_gb: Optional[float] = None
    capability: Optional[str] = None  # e.g. "7.5" for Tesla T4


def detect_cuda_devices() -> List[DeviceInfo]:
    devices: List[DeviceInfo] = []
    if not torch.cuda.is_available():
        return devices

    count = torch.cuda.device_count()
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        total_gb = round(float(props.total_memory) / (1024 ** 3), 2)
        cap = f"{props.major}.{props.minor}"
        devices.append(
            DeviceInfo(
                backend="cuda",
                vendor="NVIDIA",
                name=str(props.name),
                index=i,
                total_memory_gb=total_gb,
                capability=cap,
            )
        )
    return devices


def detect_mps_devices() -> List[DeviceInfo]:
    # MPS is generally a single logical device from PyTorch’s perspective.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return [
            DeviceInfo(
                backend="mps",
                vendor="Apple",
                name="Apple Silicon GPU (MPS)",
                index=0,
            )
        ]
    return []


def detect_rocm_devices() -> List[DeviceInfo]:
    # Placeholder: PyTorch ROCm is typically exposed through torch.cuda as well,
    # but vendor/arch detection is different. We keep a stub for future.
    return []


def detect_xpu_devices() -> List[DeviceInfo]:
    # Placeholder: Intel XPU support depends on extensions (e.g., intel-extension-for-pytorch).
    return []


def detect_cpu_device() -> List[DeviceInfo]:
    cpu_name = platform.processor() or platform.machine_
