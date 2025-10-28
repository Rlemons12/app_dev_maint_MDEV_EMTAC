"""
spacy_search.py
---------------
SpaCy-enhanced NLP search pipeline.
This module wraps tokenization, entity extraction, and intent classification.
"""

from typing import Dict, Any, List, Optional
import logging
import re
import spacy

from .ml_models import IntentClassifierML
from modules.configuration.log_config import get_request_id, debug_id, info_id

logger = logging.getLogger(__name__)


class SpaCyEnhancedAggregateSearch:
    """
    NLP layer that:
      - Uses SpaCy for tokenization + entity extraction
      - Uses ML intent classifier for intent detection
      - Provides enriched output with confidence + fallback
    """

    def __init__(self, nlp_model: str = "en_core_web_sm"):
        self.request_id = get_request_id()
        self.intent_classifier = IntentClassifierML()
        try:
            self.nlp = spacy.load(nlp_model)
            info_id(f"[SpaCyEnhancedAggregateSearch] Loaded spaCy model {nlp_model}", self.request_id)
        except Exception as e:
            logger.warning(f"Failed to load spaCy model {nlp_model}, falling back: {e}")
            self.nlp = None

        # simple in-memory cache
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------ Public API ------------------

    def analyze_user_input(self, text: str) -> Dict[str, Any]:
        """
        Analyze a user query: detect intent, extract entities, assign confidence.
        Mirrors old nlp_search.analyze_user_input structure.
        """
        if not text:
            return {
                "query": text,
                "intent": "unknown",
                "entities": [],
                "confidence_score": 0.0,
                "processing_method": "empty",
            }

        if text in self._analysis_cache:
            return self._analysis_cache[text]

        rid = get_request_id()
        debug_id(f"[SpaCyEnhancedAggregateSearch] analyzing: {text}", rid)

        # run intent classifier
        intent, confidence = self.intent_classifier.predict_intent(text)

        # normalize confidence to float
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        # extract entities
        entities = self._extract_entities(text)

        result = {
            "query": text,
            "intent": intent,
            "entities": entities,
            "confidence_score": confidence,  # ✅ guaranteed float
            "processing_method": "spacy" if self.nlp else "fallback",
        }

        self._analysis_cache[text] = result
        return result

    # ------------------ Helpers ------------------

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities with spaCy, fallback to regex if spaCy not available.
        """
        entities: List[Dict[str, Any]] = []

        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({"text": ent.text, "label": ent.label_})
        else:
            # fallback: detect numbers + simple keywords
            for match in re.finditer(r"\b\d+\b", text):
                entities.append({"text": match.group(0), "label": "NUMBER"})
            if "area" in text.lower():
                entities.append({"text": "area", "label": "LOCATION"})

        return entities

    def clear_cache(self) -> None:
        """Clear cached analyses."""
        self._analysis_cache.clear()
