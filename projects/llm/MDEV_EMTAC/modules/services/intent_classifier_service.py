"""
IntentClassifierService
-----------------------

Loads DistilBERT intent classifier model ONCE and exposes:

    IntentClassifierService.predict(text)

Model path comes from .env via:
    MODELS_DISTILBERT_INTENT=E:\emtac\models\...\distilbert_intent
"""

import os
import torch
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from modules.emtac_ai.intent_ner.configuration.ner_config import get_intent_model


class IntentClassifierService:
    _model = None
    _tokenizer = None
    _label_map = None

    @classmethod
    def load_model(cls):
        """
        Load DistilBERT model exactly once.
        """
        if cls._model is not None:
            return

        model_path = get_intent_model()

        cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
        cls._model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # default label map (update if you fine-tune)
        cls._label_map = {
            0: "Parts",
            1: "Images",
            2: "Documents",
            3: "Drawings",
            4: "Tools",
            5: "Troubleshooting",
        }

        print(f"[INTENT CLASSIFIER] Loaded DistilBERT from {model_path}")

    @classmethod
    def predict(cls, text: str) -> Dict[str, object]:
        """
        Predict the intent for the given text.
        Returns: { intent: "Parts", confidence: 0.92 }
        """
        cls.load_model()

        tokens = cls._tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = cls._model(**tokens)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        intent_id = int(torch.argmax(probs))
        confidence = float(probs[intent_id])
        label = cls._label_map.get(intent_id, "Unknown")

        return {
            "intent": label,
            "confidence": confidence
        }
