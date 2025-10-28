"""
ml_models.py
------------
Machine Learning model stubs used in NLP search.
These are runtime helpers, separate from the MLModel ORM table in models.py.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class IntentClassifierML:
    """
    Stub for an intent classification ML model.
    In production, this could load a HuggingFace/Spacy model.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "default_intent_model"
        self._is_loaded = False
        self._load_model()

    def _load_model(self):
        """
        Simulate loading a model (replace with actual ML code later).
        """
        logger.info(f"[IntentClassifierML] Loading model from {self.model_path}")
        self._is_loaded = True

    def predict_intent(self, text: str) -> Dict[str, Any]:
        """
        Predict intent for a given query text.
        Returns a dict with 'intent' and 'confidence'.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        # Stub prediction
        logger.debug(f"[IntentClassifierML] Predicting intent for: {text}")
        return {
            "intent": "search.part",   # hardcoded for now
            "confidence": 0.95
        }


class FeedbackLearner:
    """
    Stub for a feedback-driven learning component.
    In production, this could retrain or fine-tune models.
    """

    def __init__(self):
        self._records = []

    def record_feedback(self, query_id: int, rating: int, comment: Optional[str] = None):
        """
        Store user feedback for later learning.
        """
        record = {"query_id": query_id, "rating": rating, "comment": comment}
        self._records.append(record)
        logger.debug(f"[FeedbackLearner] Recorded feedback: {record}")
        return True

    def retrain_if_needed(self):
        """
        Stub retrain trigger.
        """
        logger.info(f"[FeedbackLearner] Retrain check: {len(self._records)} feedback records available")
        return False
