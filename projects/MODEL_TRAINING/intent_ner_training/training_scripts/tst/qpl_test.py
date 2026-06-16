# qpl_test.py
# Complete Query Pipeline Script for Intent Classification, NER, Query Expansion, and Retrieval
# Integrated with real AI models from ai_models.py and fixed NameError for List

import os
import json
import logging
import numpy as np
from typing import List, Optional  # Added List import
from transformers import pipeline
from datasets import load_dataset
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Import real AI models and functions from ai_models.py
from plugins.ai_modules.ai_models import (
    ModelsConfig, NoAIModel, NoEmbeddingModel, TinyLlamaEmbeddingModel,
    generate_and_store_embedding, search_similar_embeddings
)


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Project paths from config.py
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
DATABASE_URL = "postgresql://postgres:emtac123@localhost:5432/emtac"
ORC_BASE_DIR = os.path.join(BASE_DIR, "modules", "emtac_ai")
ORC_MODELS_DIR = os.path.join(ORC_BASE_DIR, "models")
ORC_INTENT_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "intent_classifier")
ORC_PARTS_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "parts")
ORC_TRAINING_DATA_DIR = os.path.join(ORC_BASE_DIR, "training_data", "datasets")
ORC_INTENT_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "intent_classifier")
ORC_PARTS_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "parts")

# Create directories
os.makedirs(ORC_INTENT_MODEL_DIR, exist_ok=True)
os.makedirs(ORC_PARTS_MODEL_DIR, exist_ok=True)
os.makedirs(ORC_INTENT_TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(ORC_PARTS_TRAIN_DATA_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def info_id(msg, req_id=None): logger.info(msg)


def error_id(msg, req_id=None): logger.error(msg)


def get_request_id(): return "default"


def log_timed_operation(name, req_id=None):
    class Context:
        def __enter__(self): return self

        def __exit__(self, *args): pass

    return Context()


# Database Config
class DatabaseConfig:
    def __init__(self):
        self.main_engine = create_engine(DATABASE_URL)
        self.MainSessionMaker = sessionmaker(bind=self.main_engine)

    @contextmanager
    def main_session(self):
        session = self.MainSessionMaker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()


# Intent and NER Plugin
class IntentEntityPlugin:
    def __init__(self, intent_model_dir=None, ner_model_dir=None, intent_labels=None, ner_labels=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.intent_model_dir = self.to_abs_path(intent_model_dir, base_dir) if intent_model_dir else None
        self.ner_model_dir = self.to_abs_path(ner_model_dir, base_dir) if ner_model_dir else None

        self.intent_labels = intent_labels or ["parts", "images", "documents", "prints", "tools", "troubleshooting"]
        self.intent_id2label = {i: label for i, label in enumerate(self.intent_labels)}
        self.intent_label2id = {label: i for i, label in enumerate(self.intent_labels)}
        self.ner_labels = ner_labels or ["O", "PART_NAME", "MODEL"]
        self.ner_id2label = {i: label for i, label in enumerate(self.ner_labels)}
        self.ner_label2id = {label: i for i, label in enumerate(self.ner_labels)}

        self.intent_classifier = None
        self.ner = None

        # Load intent classifier
        if self.intent_model_dir and os.path.exists(self.intent_model_dir):
            try:
                config_path = os.path.join(self.intent_model_dir, "config.json")
                model_files = [f for f in os.listdir(self.intent_model_dir) if f.endswith(('.bin', '.safetensors'))]
                if os.path.exists(config_path) and model_files:
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model=self.intent_model_dir
                    )
                    logger.info(f"Loaded intent classifier from {self.intent_model_dir}")
                else:
                    logger.warning(f"No valid model files in {self.intent_model_dir}, using mock")
            except Exception as e:
                logger.warning(f"Could not load intent classifier from {self.intent_model_dir}: {e}")

        # Load NER model
        if self.ner_model_dir and os.path.exists(self.ner_model_dir):
            try:
                config_path = os.path.join(self.ner_model_dir, "config.json")
                model_files = [f for f in os.listdir(self.ner_model_dir) if f.endswith(('.bin', '.safetensors'))]
                if os.path.exists(config_path) and model_files:
                    self.ner = pipeline(
                        "ner",
                        model=self.ner_model_dir,
                        aggregation_strategy="simple"
                    )
                    logger.info(f"Loaded NER model from {self.ner_model_dir}")
                else:
                    logger.warning(f"No valid model files in {self.ner_model_dir}, using mock")
            except Exception as e:
                logger.warning(f"Could not load NER model from {self.ner_model_dir}: {e}")

    def to_abs_path(self, path, base_dir):
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(base_dir, path))

    def classify_intent(self, text):
        if not self.intent_classifier:
            logger.warning("Intent classifier not loaded, using mock")
            return "parts", 0.9
        try:
            results = self.intent_classifier(text)
            if results:
                return results[0]['label'], float(results[0]['score'])
        except Exception as e:
            logger.error(f"Error during intent classification: {e}")
            return None, 0.0

    def extract_entities(self, text):
        if not self.ner:
            logger.warning("NER model not loaded, using mock")
            return [{'word': text.split()[-1], 'entity_group': 'MODEL'}]
        try:
            raw_entities = self.ner(text)
            for ent in raw_entities:
                entity_group = ent.get('entity_group', None)
                if entity_group and entity_group.startswith("LABEL_"):
                    label_id = int(entity_group.split("_")[1])
                    ent['entity_group'] = self.ner_id2label.get(label_id, entity_group)
                if 'score' in ent:
                    ent['score'] = float(ent['score'])
            return raw_entities
        except Exception as e:
            logger.error(f"Error during entity extraction: {e}")
            return []

    def train_intent(self, train_data, output_dir="models/intent-custom", epochs=3):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
        if not os.path.exists(self.intent_model_dir):
            logger.error(f"Base intent model directory not found: {self.intent_model_dir}")
            return

        tokenizer = AutoTokenizer.from_pretrained(self.intent_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.intent_model_dir,
            num_labels=len(self.intent_labels),
            id2label=self.intent_id2label,
            label2id=self.intent_label2id,
            ignore_mismatched_sizes=True
        )

        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, padding=True)

        if isinstance(train_data, str):
            dataset = load_dataset('json', data_files=train_data)['train']
        else:
            dataset = train_data

        def map_intent(example):
            example["label"] = self.intent_label2id[example["intent"]]
            return example

        dataset = dataset.map(map_intent)
        tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            save_steps=10,
            save_total_limit=2,
            logging_steps=5,
            report_to="none"
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
        logger.info(f"Fine-tuned Intent model saved to: {output_dir}")

    def train_ner(self, train_data, output_dir="models/ner-custom", epochs=3):
        from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
        if not os.path.exists(self.ner_model_dir):
            logger.error(f"Base NER model directory not found: {self.ner_model_dir}")
            return

        tokenizer = AutoTokenizer.from_pretrained(self.ner_model_dir)
        model = AutoModelForTokenClassification.from_pretrained(
            self.ner_model_dir,
            num_labels=len(self.ner_labels),
            id2label=self.ner_id2label,
            label2id=self.ner_label2id,
            ignore_mismatched_sizes=True
        )

        def tokenize_and_align_labels(example):
            tokenized_inputs = tokenizer(
                example["tokens"],
                truncation=True,
                padding="max_length",
                max_length=128,
                is_split_into_words=True,
                return_offsets_mapping=True
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

        if isinstance(train_data, str):
            dataset = load_dataset('json', data_files=train_data)['train']
        else:
            dataset = train_data

        tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False, remove_columns=dataset.column_names)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=10,
            save_total_limit=2,
            logging_steps=5,
            report_to="none"
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
        logger.info(f"Fine-tuned NER model saved to: {output_dir}")


# Query Expansion
class QueryExpansionRAG:
    def __init__(self, ai_model_name=None, embedding_model_name=None, use_spacy=True):
        request_id = get_request_id()
        info_id("Initializing QueryExpansionRAG system", request_id)
        try:
            self.ai_model = ModelsConfig.load_ai_model(ai_model_name) or NoAIModel()
            self.llm_available = not isinstance(self.ai_model, NoAIModel)
            info_id(f"Loaded AI model: {ai_model_name or 'NoAIModel'}", request_id)
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            self.ai_model = NoAIModel()
            self.llm_available = False

        try:
            self.embedding_model = ModelsConfig.load_embedding_model(embedding_model_name) or TinyLlamaEmbeddingModel()
            self.embeddings_available = not isinstance(self.embedding_model, NoEmbeddingModel)
            info_id(f"Loaded embedding model: {embedding_model_name or 'TinyLlamaEmbeddingModel'}", request_id)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = NoEmbeddingModel()
            self.embeddings_available = False

    def multi_query_expansion_ai(self, query: str, num_variants: int = 4) -> List[str]:
        # Placeholder: Replace with full implementation from query_expansion_techniques.py
        if not self.llm_available:
            return [query + f" variant {i}" for i in range(num_variants)]
        try:
            prompt = f"Generate {num_variants} alternative queries for: '{query}'"
            response = self.ai_model.get_response(prompt)
            # Assume response is a list of queries; adjust parsing as needed
            return [f"{query} variant {i}" for i in range(num_variants)]  # Mock parsing
        except Exception as e:
            logger.error(f"Error in multi_query_expansion_ai: {e}")
            return [query + f" variant {i}" for i in range(num_variants)]

    def comprehensive_expansion(self, query: str, top_docs=None):
        # Placeholder: Replace with full implementation
        return {"multi_ai": self.multi_query_expansion_ai(query)}

    def expand_query(self, query: str, method="multi_query_ai", num_variants=4, top_docs=None):
        if method == "comprehensive":
            return self.comprehensive_expansion(query, top_docs)
        return self.multi_query_expansion_ai(query, num_variants)


# Query Pipeline
class QueryPipeline:
    def __init__(self, ai_model_name="TinyLlamaModel", embedding_model_name="TinyLlamaEmbeddingModel",
                 intent_plugin=None):
        self.expander = QueryExpansionRAG(ai_model_name, embedding_model_name)
        self.intent_plugin = intent_plugin or IntentEntityPlugin(
            intent_model_dir=ORC_INTENT_MODEL_DIR,
            ner_model_dir=ORC_PARTS_MODEL_DIR
        )
        self.db_config = DatabaseConfig()
        info_id("QueryPipeline initialized", get_request_id())

    def process_query(self, raw_query: str, top_docs=None, num_variants=4, method="comprehensive"):
        request_id = get_request_id()
        info_id(f"Processing query: '{raw_query}'", request_id)

        intent, intent_conf = self.intent_plugin.classify_intent(raw_query)
        if not intent:
            intent = "general"

        entities = self.intent_plugin.extract_entities(raw_query)
        entity_terms = [ent['word'] for ent in entities if ent['entity_group'].startswith(('PART_NAME', 'MODEL'))]

        enhanced_query = raw_query + " " + " ".join(entity_terms) if entity_terms else raw_query

        if method == "comprehensive":
            expansions = self.expander.comprehensive_expansion(enhanced_query, top_docs=top_docs)
        else:
            expansions = self.expander.expand_query(enhanced_query, method=method, num_variants=num_variants,
                                                    top_docs=top_docs)

        if isinstance(expansions, dict):
            all_expanded = [q for queries in expansions.values() for q in queries]
        else:
            all_expanded = expansions

        retrieved_docs = []
        with self.db_config.main_session() as session:
            for expanded in all_expanded[:3]:
                query_emb = self.expander.embedding_model.get_embeddings(expanded)
                similar = search_similar_embeddings(session, query_emb, threshold=0.8, limit=5)
                retrieved_docs.extend(similar)

        unique_docs = {doc['document_id']: doc for doc in retrieved_docs}.values()

        return {
            'intent': intent,
            'intent_confidence': intent_conf,
            'entities': entities,
            'expanded_queries': all_expanded,
            'retrieved_docs': list(unique_docs)
        }


# Main execution
if __name__ == "__main__":
    # Create sample training data if none exists
    intent_data_path = os.path.join(ORC_INTENT_TRAIN_DATA_DIR, "intent_data.jsonl")
    if not os.path.exists(intent_data_path):
        with open(intent_data_path, 'w') as f:
            f.write('{"text": "Show me the pump schematic", "intent": "documents"}\n')
            f.write('{"text": "Troubleshoot HVAC system failure", "intent": "troubleshooting"}\n')
            f.write('{"text": "Find wiring diagram for motor", "intent": "documents"}\n')

    ner_data_path = os.path.join(ORC_PARTS_TRAIN_DATA_DIR, "ner_data.jsonl")
    if not os.path.exists(ner_data_path):
        with open(ner_data_path, 'w') as f:
            f.write(
                '{"tokens": ["Show", "me", "the", "pump", "schematic", "for", "part", "number", "VFD-123"], "ner_tags": [0, 0, 0, 1, 1, 0, 0, 0, 2]}\n')
            f.write('{"tokens": ["Troubleshoot", "HVAC", "system", "failure"], "ner_tags": [0, 1, 1, 0]}\n')
            f.write('{"tokens": ["Find", "wiring", "diagram", "for", "motor"], "ner_tags": [0, 1, 1, 0, 1]}\n')

    # Train models (uncomment to fine-tune)
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

    # Initialize and test pipeline with real AI models
    pipeline = QueryPipeline(ai_model_name="TinyLlamaModel", embedding_model_name="TinyLlamaEmbeddingModel")
    test_queries = [
        "Show me the pump schematic for part number VFD-123",
        "Troubleshoot HVAC system failure",
        "Find wiring diagram for motor"
    ]
    for query in test_queries:
        result = pipeline.process_query(query)
        print(f"\nQuery: {query}")
        print(json.dumps(result, indent=4, cls=NumpyEncoder))