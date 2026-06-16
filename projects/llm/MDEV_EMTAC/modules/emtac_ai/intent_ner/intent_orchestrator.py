from __future__ import annotations

from typing import Dict, Any, Optional, List
from pathlib import Path
from modules.configuration.log_config import (
    info_id,
    debug_id,
    warning_id,
    with_request_id,
)

from modules.configuration.config import (
    ORC_INTENT_MODEL_DIR,
    ORC_PARTS_MODEL_DIR,
    ORC_IMAGES_MODEL_DIR,
    ORC_DOCUMENTS_MODEL_DIR,
    ORC_DRAWINGS_MODEL_DIR,
)

from modules.emtac_ai.emtac_intent_entity import IntentEntityPlugin


# ------------------------------------------------------------------
# INTENT → NER + RESOLUTION / EXPANSION CONFIG
# ------------------------------------------------------------------
INTENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "documents": {
        "primary_category": "documents",
        "ner_model_dir": ORC_DOCUMENTS_MODEL_DIR,
        "expansion_strategy": "by_association",
        "expand_categories": ["images", "drawings", "parts", "positions"],
    },
    "parts": {
        "primary_category": "parts",
        "ner_model_dir": ORC_PARTS_MODEL_DIR,
        "expansion_strategy": "by_association",
        "expand_categories": ["drawings", "images", "documents", "positions"],
    },
    "drawings": {
        "primary_category": "drawings",
        "ner_model_dir": ORC_DRAWINGS_MODEL_DIR,
        "expansion_strategy": "by_association",
        "expand_categories": ["parts", "documents", "images", "positions"],
    },
    "images": {
        "primary_category": "images",
        "ner_model_dir": ORC_IMAGES_MODEL_DIR,
        "expansion_strategy": "by_association",
        "expand_categories": ["documents", "drawings", "parts", "positions"],
    },
    "general": {
        "primary_category": "documents",
        "ner_model_dir": ORC_DOCUMENTS_MODEL_DIR,
        "expansion_strategy": "mixed",
        "expand_categories": ["documents", "images", "drawings", "parts", "positions"],
    },
}


def resolve_latest_model_dir(base_dir: str) -> str:
    """
    Resolve models/<intent>/LATEST.txt → models/<intent>/<run>/best
    Fallbacks safely to base_dir if no versioning is present.
    """
    base = Path(base_dir)
    latest_file = base / "LATEST.txt"

    if latest_file.exists():
        run_name = latest_file.read_text().strip()
        candidate = base / run_name / "best"
        if candidate.exists():
            return str(candidate)

    return base_dir


# ------------------------------------------------------------------
# CANONICAL INTENT + NER ORCHESTRATOR
# ------------------------------------------------------------------
class IntentNEROrchestrator:
    """
    Canonical Intent + NER Orchestrator.

    Responsibilities:
      1. Classify intent
      2. Load intent-specific NER
      3. Extract + normalize entities
      4. Emit resolver-ready instructions

    DOES NOT:
      - Resolve DB IDs
      - Call services
      - Expand graphs
      - Perform ranking
      - Call RAG
    """

    def __init__(self, intent_model_dir: Optional[str] = None):
        self.intent_model_dir = intent_model_dir or ORC_INTENT_MODEL_DIR

        self.intent_plugin = IntentEntityPlugin(
            intent_model_dir=self.intent_model_dir,
            ner_model_dir=None,
        )

        debug_id(
            f"[IntentNER] Initialized with intent_model_dir={self.intent_model_dir}"
        )

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    @with_request_id
    def process(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return self._empty_response(query)

        # --------------------------------------------------------------
        # 1. INTENT CLASSIFICATION
        # --------------------------------------------------------------
        intent, confidence = self.intent_plugin.classify_intent(query)
        intent = (intent or "general").lower()

        if intent not in INTENT_CONFIG:
            debug_id(f"[IntentNER] Unknown intent '{intent}', falling back to 'general'")
            intent = "general"

        debug_id(
            f"[IntentNER] intent='{intent}' confidence={confidence:.3f}"
        )

        intent_cfg = INTENT_CONFIG[intent]

        # --------------------------------------------------------------
        # 2. LOAD INTENT-SPECIFIC NER MODEL
        # --------------------------------------------------------------
        base_ner_dir = intent_cfg["ner_model_dir"]
        resolved_ner_dir = resolve_latest_model_dir(base_ner_dir)

        self.intent_plugin.set_ner_model(resolved_ner_dir)

        debug_id(
            f"[IntentNER] Using NER model at {resolved_ner_dir}"
        )

        # --------------------------------------------------------------
        # 3. ENTITY EXTRACTION
        # --------------------------------------------------------------
        try:
            raw_entities = self.intent_plugin.extract_entities(query)
        except Exception as e:
            warning_id(
                f"[IntentNER] NER extraction failed: {e}"
            )
            raw_entities = []

        entities = self._normalize_entities(raw_entities)

        info_id(
            f"[IntentNER] Extracted {sum(len(v) for v in entities.values())} entities"
        )

        # --------------------------------------------------------------
        # 4. BUILD RESOLVER-READY PAYLOAD
        # --------------------------------------------------------------
        return {
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "primary_category": intent_cfg["primary_category"],
            "expansion_strategy": intent_cfg["expansion_strategy"],
            "expand_categories": intent_cfg["expand_categories"],
            "entities": entities,  # ← Resolver contract starts here
        }

    # ------------------------------------------------------------------
    # ENTITY NORMALIZATION
    # ------------------------------------------------------------------
    def _normalize_entities(self, entities: Any) -> Dict[str, List[str]]:
        """
        Normalize raw NER output into resolver-ready labels.

        Canonical labels (examples):
          DOCUMENT_ID
          DOCUMENT_TITLE
          FILE_NAME
          PART_ID
          ITEMNUM
          MANUFACTURER
          MODEL
          DESCRIPTION
          DRAWING_NUMBER
        """
        if not entities:
            return {}

        normalized: Dict[str, List[str]] = {}

        # Case 1: already a dict-like structure
        if isinstance(entities, dict):
            for label, values in entities.items():
                key = label.upper()
                normalized.setdefault(key, [])
                normalized[key].extend(str(v) for v in values if v)
            return normalized

        # Case 2: list of NER spans
        for ent in entities:
            label = ent.get("label")
            value = ent.get("entity")
            if not label or not value:
                continue

            key = label.upper()
            normalized.setdefault(key, []).append(str(value))

        return normalized

    # ------------------------------------------------------------------
    # EMPTY / FALLBACK RESPONSE
    # ------------------------------------------------------------------
    def _empty_response(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "intent": "general",
            "confidence": 0.0,
            "primary_category": "documents",
            "expansion_strategy": "mixed",
            "expand_categories": ["documents", "images", "drawings", "parts", "positions"],
            "entities": {},
        }


# ------------------------------------------------------------------
# BACKWARD COMPATIBILITY
# ------------------------------------------------------------------
IntentOrchestrator = IntentNEROrchestrator
