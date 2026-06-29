"""
Diagnostics helpers for accelerator state.

Intended for logging, debug endpoints, or CLI prints.
"""

from __future__ import annotations

from typing import List, Dict, Any

from .detection import DeviceInfo
from .policies import PrecisionPolicy


def devices_to_dict(devices: List[DeviceInfo]) -> List[Dict[str, Any]]:
    return [
        {
            "backend": d.backend,
            "vendor": d.vendor,
            "name": d.name,
            "index": d.index,
            "total_memory_gb": d.total_memory_gb,
            "capability": d.capability,
        }
        for d in devices
    ]


def build_summary(
    selected_backend: str,
    device_str: str,
    device_count: int,
    detected_devices: List[DeviceInfo],
    precision_policy: PrecisionPolicy,
) -> Dict[str, Any]:
    return {
        "selected_backend": selected_backend,
        "device": device_str,
        "device_count": device_count,
        "detected_devices": devices_to_dict(detected_devices),
        "precision_policy": {
            "precision": precision_policy.precision,
            "enable_autocast": precision_policy.enable_autocast,
            "reason": precision_policy.reason,
        },
    }


def format_summary_text(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"selected_backend={summary.get('selected_backend')}")
    lines.append(f"device={summary.get('device')} count={summary.get('device_count')}")
    lines.append("detected_devices:")
    for d in summary.get("detected_devices", []):
        lines.append(
            f"  - backend={d.get('backend')} vendor={d.get('vendor')} "
            f"name={d.get('name')} index={d.get('index')} "
            f"mem_gb={d.get('total_memory_gb')} cap={d.get('capability')}"
        )
    pp = summary.get("precision_policy", {})
    lines.append(
        f"precision_policy: precision={pp.get('precision')} "
        f"autocast={pp.get('enable_autocast')} reason={pp.get('reason')}"
    )
    return "\n".join(lines)
