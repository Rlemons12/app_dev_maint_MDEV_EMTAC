"""
Default fallback router for EMTAC Orchestrator.
This is used when:
    - no specialized router matches the intent
    - intent confidence is low
    - entities extracted but irrelevant to any domain router
"""

from typing import Dict, Any, List


def default_router(*, text: str, intent: str, confidence: float, entities: Dict[str, Any]):
    """
    BaseRouter-compatible fallback return.

    Shape returned must be:
        (models: list, matched_on: str, related_dict: dict)
    """

    models: List[Any] = []  # No data for default fallback
    matched_on: str = "fallback_default"

    related = {
        "message": "Default fallback router used. No domain-specific matches.",
        "original_text": text,
        "entities": entities,
    }

    # BaseRouter interprets this as:
    # models = []
    # matched_on = "fallback_default"
    # serializer = default serializer from BaseRouter (ignored because no models)
    # related_override = related dict
    return models, matched_on, related
