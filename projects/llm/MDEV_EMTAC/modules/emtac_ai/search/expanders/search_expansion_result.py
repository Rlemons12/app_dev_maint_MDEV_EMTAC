from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class SearchExpansionResult:
    """
    Canonical return type for ALL search expanders.

    Rules:
      - `intent` MUST be set
      - `primary` MUST contain exactly ONE primary entity type
      - `context` may contain zero or more secondary entity types
      - expanders NEVER generate answers
    """

    intent: str

    # Exactly one primary entity group
    primary: Dict[str, List[Any]] = field(default_factory=dict)

    # Optional supporting context
    context: Dict[str, List[Any]] = field(default_factory=dict)

    # Optional metadata for debugging / UI
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------
    # Convenience helpers
    # ----------------------------

    def add_primary(self, key: str, values: List[Any]):
        self.primary[key] = values

    def add_context(self, key: str, values: List[Any]):
        if values:
            self.context[key] = values

    def is_empty(self) -> bool:
        return not any(self.primary.values())
