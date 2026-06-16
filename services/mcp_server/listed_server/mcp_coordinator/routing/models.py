from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    target_capability: str
    target_tool: str | None
    confidence: float
    reason: str
    safe_to_execute: bool
    needs_confirmation: bool
    suggested_arguments: dict[str, Any] = field(default_factory=dict)

