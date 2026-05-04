# qpl_test.py
# Complete Query Pipeline Script for Intent Classification, NER, Query Expansion, and Retrieval
# Updated to use shared DatabaseConfig singleton via get_db_config()

from __future__ import annotations

import os
import json
import logging
from typing import List, Optional

import numpy as np
from datasets import load_dataset
from transformers import pipeline

# Shared DB config from app configuration
from modules.configuration.config_env import get_db_config

# Import real AI models and functions from ai_models.py
from plugins.ai_modules.ai_models import (
    ModelsConfig,
    NoAIModel,
    NoEmbeddingModel,
    TinyLlamaEmbeddingModel,
    generate_and_store_embedding,
    search_similar_embeddings,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def info_id(msg, req_id=None):
    logger.info(msg)


def error_id(msg, req_id=None):
    logger.error(msg)


def get_request_id():
    return "default"


def log_timed_operation(name, req_id=None):
    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return Context()


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
ORC_BASE_DIR = os.path.join(BASE_DIR, "modules", "emtac_ai")
ORC_MODELS_DIR = os.path.join(ORC_BASE_DIR, "models")
ORC_INTENT_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "intent_classifier")
ORC_PARTS_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "parts")
ORC_TRAINING_DATA_DIR = os.path.join(ORC_BASE_DIR, "training_data", "datasets")
ORC_INTENT_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "intent_classifier")
ORC_PARTS_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "parts")

os.makedirs(ORC_INTENT_MODEL_DIR, exist_ok=True)
os.makedirs(ORC_PARTS_MODEL_DIR, exist_ok=True)
os.makedirs(ORC_INTENT_TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(ORC_PARTS_TRAIN_DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Intent and NER plugin
# ---------------------------------------------------------------------------
class IntentEntityPlugin:
    def __init__(
        self,
        intent_model_dir: Optional[str] = None,
        ner_model_dir: Optional[str] = None,
        intent_labels: Optional[List[str]] = None,
        ner_labels: Optional[List[str]] = None,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.intent_model_dir = (
            self.to_abs_path(intent_model_dir, base_dir) if intent_model_dir else None
        )
        self.ner_model_dir = (
            self.to_abs_path(ner_model_dir, base_dir) if ner_model_dir else None
        )

        self.intent_labels = intent_labels or [
            "parts",
            "images",
            "documents",
            "prints",
            "tools",
            "troubleshooting",
        ]
        self.intent_id2label = {
            i: label for i, label in enumerate(self.intent_labels)
        }
        self.intent_label2id = {
            label: i for i, label in enumerate(self.intent_labels)
        }

        self.ner_labels = ner_labels or ["O", "PART_NAME", "MODEL"]
        self.ner_id2label = {i: label for i, label in enumerate(self.ner_labels)}
        self.ner_label2id = {label: i for i, label in enumerate(self.ner_labels)}

        self.intent_classifier = None
        self.ner = None

        if self.intent_model_dir and os.path.exists(self.intent_model_dir):
            try:
                config_path = os.path.join(self.intent_model_dir, "config.json")
                model_files = [
                    f
                    for f in os.listdir(self.intent_model_dir)
                    if f.endswith((".bin", ".safetensors"))
                ]
                if os.path.exists(config_path) and model_files:
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model=self.intent_model_dir,
                    )
                    logger.info(
                        "Loaded intent classifier from %s",
                        self.intent_model_dir,
                    )
                else:
                    logger.warning(
                        "No valid model files in %s, using mock",
                        self.intent_model_dir,
                    )
            except Exception as exc:
                logger.warning(
                    "Could not load intent classifier from %s: %s",
                    self.intent_model_dir,
                    exc,
                )

        if self.ner_model_dir and os.path.exists(self.ner_model_dir):
            try:
                config_path = os.path.join(self.ner_model_dir, "config.json")
                model_files = [
                    f
                    for f in os.listdir(self.ner_model_dir)
                    if f.endswith((".bin", ".safetensors"))
                ]
                if os.path.exists(config_path) and model_files:
                    self.ner = pipeline(
                        "ner",
                        model=self.ner_model_dir,
                        aggregation_strategy="simple",
                    )
                    logger.info("Loaded NER model from %s", self.ner_model_dir)
                else:
                    logger.warning(
                        "No valid model files in %s, using mock",
                        self.ner_model_dir,
                    )
            except Exception as exc:
                logger.warning(
                    "Could not load NER model from %s: %s",
                    self.ner_model_dir,
                    exc,
                )

    def to_abs_path(self, path: Optional[str], base_dir: str) -> Optional[str]:
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(base_dir, path))

    def classify_intent(self, text: str):
        if not self.intent_classifier:
            logger.warning("Intent classifier not loaded, using mock")
            return "parts", 0.9

        try:
            results = self.intent_classifier(text)
            if results:
                return results[0]["label"], float(results[0]["score"])
        except Exception as exc:
            logger.error("Error during intent classification: %s", exc)

        return None, 0.0

    def extract_entities(self, text: str):
        if not self.ner:
            logger.warning("NER model not loaded, using mock")
            parts = text.split()
            fallback = parts[-1] if parts else text
            return [{"word": fallback, "entity_group": "MODEL"}]

        try:
            raw_entities = self.ner(text)
            for ent in raw_entities:
                entity_group = ent.get("entity_group")
                if entity_group and entity_group.startswith("LABEL_"):
                    label_id = int(entity_group.split("_")[1])
                    ent["entity_group"] = self.ner_id2label.get(label_id, entity_group)
                if "score" in ent:
                    ent["score"] = float(ent["score"])
            return raw_entities
        except Exception as exc:
            logger.error("Error during entity extraction: %s", exc)
            return []

    def train_intent(self, train_data, output_dir="models/intent-custom", epochs=3):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        if not self.intent_model_dir or not os.path.exists(self.intent_model_dir):
            logger.error(
                "Base intent model directory not found: %s",
                self.intent_model_dir,
            )
            return

        tokenizer = AutoTokenizer.from_pretrained(self.intent_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.intent_model_dir,
            num_labels=len(self.intent_labels),
            id2label=self.intent_id2label,
            label2id=self.intent_label2id,
            ignore_mismatched_sizes=True,
        )

        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, padding=True)

        dataset = (
            load_dataset("json", data_files=train_data)["train"]
            if isinstance(train_data, str)
            else train_data
        )

        def map_intent(example):
            example["label"] = self.intent_label2id[example["intent"]]
            return example

        dataset = dataset.map(map_intent)
        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            save_steps=10,
            save_total_limit=2,
            logging_steps=5,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Fine-tuned Intent model saved to: %s", output_dir)

    def train_ner(self, train_data, output_dir="models/ner-custom", epochs=3):
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        if not self.ner_model_dir or not os.path.exists(self.ner_model_dir):
            logger.error("Base NER model directory not found: %s", self.ner_model_dir)
            return

        tokenizer = AutoTokenizer.from_pretrained(self.ner_model_dir)
        model = AutoModelForTokenClassification.from_pretrained(
            self.ner_model_dir,
            num_labels=len(self.ner_labels),
            id2label=self.ner_id2label,
            label2id=self.ner_label2id,
            ignore_mismatched_sizes=True,
        )

        def tokenize_and_align_labels(example):
            tokenized_inputs = tokenizer(
                example["tokens"],
                truncation=True,
                padding="max_length",
                max_length=128,
                is_split_into_words=True,
                return_offsets_mapping=True,
            )

            labels = []
            word_ids = tokenized_inputs.word_ids()
            prev_word_id = None
            label_ids = example["ner_tags"]

            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id != prev_word_id:
                    labels.append(label_ids[word_id])
                else:
                    labels.append(label_ids[word_id])
                prev_word_id = word_id

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        dataset = (
            load_dataset("json", data_files=train_data)["train"]
            if isinstance(train_data, str)
            else train_data
        )

        tokenized_dataset = dataset.map(
            tokenize_and_align_labels,
            batched=False,
            remove_columns=dataset.column_names,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=10,
            save_total_limit=2,
            logging_steps=5,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Fine-tuned NER model saved to: %s", output_dir)


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
class QueryExpansionRAG:
    def __init__(
        self,
        ai_model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        use_spacy: bool = True,
    ):
        request_id = get_request_id()
        info_id("Initializing QueryExpansionRAG system", request_id)

        try:
            self.ai_model = ModelsConfig.load_ai_model(ai_model_name) or NoAIModel()
            self.llm_available = not isinstance(self.ai_model, NoAIModel)
            info_id(
                f"Loaded AI model: {ai_model_name or 'NoAIModel'}",
                request_id,
            )
        except Exception as exc:
            logger.error("Failed to load AI model: %s", exc)
            self.ai_model = NoAIModel()
            self.llm_available = False

        try:
            self.embedding_model = (
                ModelsConfig.load_embedding_model(embedding_model_name)
                or TinyLlamaEmbeddingModel()
            )
            self.embeddings_available = not isinstance(
                self.embedding_model,
                NoEmbeddingModel,
            )
            info_id(
                f"Loaded embedding model: {embedding_model_name or 'TinyLlamaEmbeddingModel'}",
                request_id,
            )
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            self.embedding_model = NoEmbeddingModel()
            self.embeddings_available = False

    def multi_query_expansion_ai(self, query: str, num_variants: int = 4) -> List[str]:
        if not self.llm_available:
            return [f"{query} variant {i}" for i in range(num_variants)]

        try:
            prompt = f"Generate {num_variants} alternative queries for: '{query}'"
            _response = self.ai_model.get_response(prompt)
            return [f"{query} variant {i}" for i in range(num_variants)]
        except Exception as exc:
            logger.error("Error in multi_query_expansion_ai: %s", exc)
            return [f"{query} variant {i}" for i in range(num_variants)]

    def comprehensive_expansion(self, query: str, top_docs=None):
        return {"multi_ai": self.multi_query_expansion_ai(query)}

    def expand_query(
        self,
        query: str,
        method: str = "multi_query_ai",
        num_variants: int = 4,
        top_docs=None,
    ):
        if method == "comprehensive":
            return self.comprehensive_expansion(query, top_docs)
        return self.multi_query_expansion_ai(query, num_variants)


# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------
class QueryPipeline:
    def __init__(
        self,
        ai_model_name: str = "TinyLlamaModel",
        embedding_model_name: str = "TinyLlamaEmbeddingModel",
        intent_plugin: Optional[IntentEntityPlugin] = None,
        db_config=None,
    ):
        self.expander = QueryExpansionRAG(ai_model_name, embedding_model_name)
        self.intent_plugin = intent_plugin or IntentEntityPlugin(
            intent_model_dir=ORC_INTENT_MODEL_DIR,
            ner_model_dir=ORC_PARTS_MODEL_DIR,
        )

        # Use shared singleton DB config instead of local DatabaseConfig()
        self.db_config = db_config or get_db_config()

        info_id("QueryPipeline initialized", get_request_id())

    def process_query(
        self,
        raw_query: str,
        top_docs=None,
        num_variants: int = 4,
        method: str = "comprehensive",
    ):
        request_id = get_request_id()
        info_id(f"Processing query: '{raw_query}'", request_id)

        intent, intent_conf = self.intent_plugin.classify_intent(raw_query)
        if not intent:
            intent = "general"
            intent_conf = 0.0

        entities = self.intent_plugin.extract_entities(raw_query)
        entity_terms = [
            ent.get("word", "")
            for ent in entities
            if str(ent.get("entity_group", "")).startswith(("PART_NAME", "MODEL"))
        ]

        enhanced_query = (
            f"{raw_query} {' '.join(entity_terms)}".strip()
            if entity_terms
            else raw_query
        )

        if method == "comprehensive":
            expansions = self.expander.comprehensive_expansion(
                enhanced_query,
                top_docs=top_docs,
            )
        else:
            expansions = self.expander.expand_query(
                enhanced_query,
                method=method,
                num_variants=num_variants,
                top_docs=top_docs,
            )

        if isinstance(expansions, dict):
            all_expanded = [
                query_text
                for queries in expansions.values()
                for query_text in queries
            ]
        else:
            all_expanded = expansions

        retrieved_docs = []

        # Shared session factory from singleton DB config
        with self.db_config.get_main_session() as session:
            for expanded in all_expanded[:3]:
                query_emb = self.expander.embedding_model.get_embeddings(expanded)
                similar = search_similar_embeddings(
                    session,
                    query_emb,
                    threshold=0.8,
                    limit=5,
                )
                retrieved_docs.extend(similar)

        unique_docs = {}
        for doc in retrieved_docs:
            document_id = doc.get("document_id")
            if document_id is not None and document_id not in unique_docs:
                unique_docs[document_id] = doc

        return {
            "intent": intent,
            "intent_confidence": intent_conf,
            "entities": entities,
            "expanded_queries": all_expanded,
            "retrieved_docs": list(unique_docs.values()),
        }


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    intent_data_path = os.path.join(ORC_INTENT_TRAIN_DATA_DIR, "intent_data.jsonl")
    if not os.path.exists(intent_data_path):
        with open(intent_data_path, "w", encoding="utf-8") as file:
            file.write('{"text": "Show me the pump schematic", "intent": "documents"}\n')
            file.write('{"text": "Troubleshoot HVAC system failure", "intent": "troubleshooting"}\n')
            file.write('{"text": "Find wiring diagram for motor", "intent": "documents"}\n')

    ner_data_path = os.path.join(ORC_PARTS_TRAIN_DATA_DIR, "ner_data.jsonl")
    if not os.path.exists(ner_data_path):
        with open(ner_data_path, "w", encoding="utf-8") as file:
            file.write(
                '{"tokens": ["Show", "me", "the", "pump", "schematic", "for", "part", "number", "VFD-123"], "ner_tags": [0, 0, 0, 1, 1, 0, 0, 0, 2]}\n'
            )
            file.write(
                '{"tokens": ["Troubleshoot", "HVAC", "system", "failure"], "ner_tags": [0, 1, 1, 0]}\n'
            )
            file.write(
                '{"tokens": ["Find", "wiring", "diagram", "for", "motor"], "ner_tags": [0, 1, 1, 0, 1]}\n'
            )

    # Optional training block
    """
    plugin = IntentEntityPlugin(
        intent_model_dir=ORC_INTENT_MODEL_DIR,
        ner_model_dir=ORC_PARTS_MODEL_DIR
    )
    if os.path.exists(intent_data_path):
        plugin.train_intent(
            train_data=intent_data_path,
            output_dir=ORC_INTENT_MODEL_DIR,
            epochs=3
        )
    if os.path.exists(ner_data_path):
        plugin.train_ner(
            train_data=ner_data_path,
            output_dir=ORC_PARTS_MODEL_DIR,
            epochs=3
        )
    """

    query_pipeline = QueryPipeline(
        ai_model_name="TinyLlamaModel",
        embedding_model_name="TinyLlamaEmbeddingModel",
    )

    test_queries = [
        "Show me the pump schematic for part number VFD-123",
        "Troubleshoot HVAC system failure",
        "Find wiring diagram for motor",
    ]

    for query in test_queries:
        result = query_pipeline.process_query(query)
        print(f"\nQuery: {query}")
        print(json.dumps(result, indent=4, cls=NumpyEncoder))