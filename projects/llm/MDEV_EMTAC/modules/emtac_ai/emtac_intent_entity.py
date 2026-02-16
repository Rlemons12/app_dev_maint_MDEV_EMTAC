import os
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

# Canonical directories come from config (single source of truth)
from modules.configuration.config import ORC_INTENT_MODEL_DIR, ORC_PARTS_MODEL_DIR  # change NER default if needed
from modules.configuration.log_config import logger


class IntentEntityPlugin:
    def __init__(
        self,
        intent_model_dir=None,
        ner_model_dir=None,
        intent_labels=None,
        ner_labels=None,
    ):
        """
        intent_model_dir: Path to intent classifier model directory
        ner_model_dir: Path to NER model directory
        """

        import inspect
        import os

        # ---- DIAGNOSTIC: caller + raw args + config defaults ----
        try:
            caller = inspect.stack()[1]
            logger.debug(f"[IntentEntityPlugin] LOADED FROM: {inspect.getfile(IntentEntityPlugin)}")
            logger.debug(f"[IntentEntityPlugin] CALLED BY: {caller.filename}:{caller.lineno}")
        except Exception:
            pass

        logger.debug(f"[IntentEntityPlugin] ARG intent_model_dir={intent_model_dir!r}, ner_model_dir={ner_model_dir!r}")
        logger.debug(f"[IntentEntityPlugin] CFG ORC_INTENT_MODEL_DIR={ORC_INTENT_MODEL_DIR}")
        logger.debug(f"[IntentEntityPlugin] CFG ORC_PARTS_MODEL_DIR={ORC_PARTS_MODEL_DIR}")

        # ---- Resolve to config defaults & coerce legacy literals ----
        LEGACY_INTENT = {"models/intent", "intent", "./models/intent", ".\\models\\intent"}
        LEGACY_NER = {"models/ner", "ner", "./models/ner", ".\\models\\ner"}

        if not intent_model_dir or intent_model_dir in LEGACY_INTENT:
            intent_model_dir = ORC_INTENT_MODEL_DIR

        if not ner_model_dir or ner_model_dir in LEGACY_NER:
            ner_model_dir = ORC_PARTS_MODEL_DIR

        self.intent_model_dir = intent_model_dir
        self.ner_model_dir = ner_model_dir

        logger.debug(f"[IntentEntityPlugin] RESOLVED intent_model_dir={self.intent_model_dir}")
        logger.debug(f"[IntentEntityPlugin] RESOLVED ner_model_dir={self.ner_model_dir}")

        # ---- Back-compat redirect ----
        if os.path.basename(self.intent_model_dir.rstrip("/\\")) == "intent":
            candidate = os.path.join(os.path.dirname(self.intent_model_dir), "intent_classifier")
            if os.path.isdir(candidate):
                logger.info(f"[IntentEntityPlugin] Redirecting 'intent' -> '{candidate}'")
                self.intent_model_dir = candidate

        # ---- Ensure folders exist ----
        os.makedirs(self.intent_model_dir, exist_ok=True)
        if self.ner_model_dir:
            os.makedirs(self.ner_model_dir, exist_ok=True)

        # ---- Auto-detect checkpoints ----
        self.intent_model_dir = self._auto_detect_checkpoint(self.intent_model_dir, "intent")
        self.ner_model_dir = self._auto_detect_checkpoint(self.ner_model_dir, "ner")

        # ---- Labels ----
        self.intent_labels = intent_labels or [
            "parts", "images", "documents", "prints", "tools", "troubleshooting"
        ]
        self.intent_id2label = dict(enumerate(self.intent_labels))
        self.intent_label2id = {v: k for k, v in self.intent_id2label.items()}

        self.ner_labels = ner_labels or ["O", "B-PARTDESC", "B-PARTNUM"]
        self.ner_id2label = dict(enumerate(self.ner_labels))
        self.ner_label2id = {v: k for k, v in self.ner_id2label.items()}

        # ---- Pipelines ----
        self.intent_classifier = None
        self.ner = None

        self._load_intent_pipeline()
        self._load_ner_pipeline()

    # ------------------------------------------------------------------
    # NEW: dynamic NER model switching (REQUIRED by orchestrator)
    # ------------------------------------------------------------------
    def set_ner_model(self, ner_model_dir: str):
        """
        Dynamically switch the active NER model.
        Safe to call repeatedly.
        """
        if not ner_model_dir:
            return

        if self.ner_model_dir == ner_model_dir and self.ner is not None:
            return  # already loaded

        logger.info(f"[IntentEntityPlugin] Switching NER model -> {ner_model_dir}")
        self.ner_model_dir = ner_model_dir
        self.ner = None
        self._load_ner_pipeline()

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------
    def _load_intent_pipeline(self):
        try:
            cfg = os.path.join(self.intent_model_dir, "config.json")
            weights = [f for f in os.listdir(self.intent_model_dir) if f.endswith((".bin", ".safetensors"))]

            if not (os.path.exists(cfg) and weights):
                logger.warning(f"[IntentEntityPlugin] Intent model files not found in {self.intent_model_dir}")
                return

            tokenizer = AutoTokenizer.from_pretrained(
                self.intent_model_dir, local_files_only=True, trust_remote_code=False
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.intent_model_dir, local_files_only=True, trust_remote_code=False
            )

            self.intent_classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                trust_remote_code=False,
            )

            logger.info("[IntentEntityPlugin] Intent classifier loaded.")

        except Exception as e:
            logger.warning(f"[IntentEntityPlugin] Could not load intent classifier: {e}")

    def _load_ner_pipeline(self):
        if not self.ner_model_dir:
            return

        try:
            cfg = os.path.join(self.ner_model_dir, "config.json")
            weights = [f for f in os.listdir(self.ner_model_dir) if f.endswith((".bin", ".safetensors"))]

            if not (os.path.exists(cfg) and weights):
                logger.warning(f"[IntentEntityPlugin] NER model files not found in {self.ner_model_dir}")
                return

            tokenizer = AutoTokenizer.from_pretrained(
                self.ner_model_dir, local_files_only=True, trust_remote_code=False
            )
            model = AutoModelForTokenClassification.from_pretrained(
                self.ner_model_dir, local_files_only=True, trust_remote_code=False
            )

            self.ner = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                trust_remote_code=False,
            )

            logger.info("[IntentEntityPlugin] NER pipeline loaded.")

        except Exception as e:
            logger.warning(f"[IntentEntityPlugin] Could not load NER model: {e}")

    def _auto_detect_checkpoint(self, base_dir, label):
        if not base_dir or not os.path.isdir(base_dir):
            return base_dir

        if os.path.exists(os.path.join(base_dir, "config.json")):
            return base_dir

        try:
            for f in os.scandir(base_dir):
                if f.is_dir() and os.path.exists(os.path.join(f.path, "config.json")):
                    logger.info(f"[IntentEntityPlugin] Auto-detected {label} checkpoint: {f.path}")
                    return f.path
        except Exception as e:
            logger.warning(f"[IntentEntityPlugin] Unable to scan {label} dir '{base_dir}': {e}")

        return base_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify_intent(self, text: str):
        if not self.intent_classifier:
            logger.warning("[IntentEntityPlugin] Intent classifier not initialized.")
            return None, 0.0

        try:
            result = self.intent_classifier(text)
            if result:
                return result[0]["label"], float(result[0]["score"])
        except Exception as e:
            logger.warning(f"[IntentEntityPlugin] Intent classification failed: {e}")

        return None, 0.0

    def extract_entities(self, text: str):
        if not self.ner:
            logger.warning("[IntentEntityPlugin] NER model not initialized.")
            return []

        try:
            return [
                {
                    "entity": ent["word"],
                    "label": ent["entity_group"],
                    "score": float(ent["score"]),
                }
                for ent in self.ner(text)
            ]
        except Exception as e:
            logger.warning(f"[IntentEntityPlugin] NER extraction failed: {e}")
            return []


