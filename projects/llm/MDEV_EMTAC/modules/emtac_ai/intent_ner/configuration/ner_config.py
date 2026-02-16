"""
modules/emtac_ai/intent_ner/configuration/ner_config.py

DB-driven NER/Intent Config
---------------------------

This version replaces:
    - env variable model paths
    - hardcoded MODELS dict
    - manual intent/entity/fallback selection

with:

    AIModelsEmbeddingService  → chooses the model NAME from DB
    ModelsConfig              → loads the embedding MODEL INSTANCE
                                and exposes `.model_path`
"""

# ----------------------------------------------------------------------
# Correct imports for your project structure
# ----------------------------------------------------------------------
from modules.services.ai_models_embedding_service import (
    AIModelsEmbeddingService,
)
from modules.services.ai_models_service import ModelsConfig


# ----------------------------------------------------------------------
# Helper: convert model name → HF directory path
# ----------------------------------------------------------------------
def _resolve_hf_path(model_name: str) -> str:
    """
    Resolve a DB-stored embedding model name into a real
    HuggingFace directory path using the model instance.

    Embedding model classes are expected to define:
        self.model_path : str
    """
    model_instance = ModelsConfig.load_embedding_model(model_name)

    if model_instance is None:
        raise RuntimeError(
            f"[NER CONFIG] Could not load embedding model instance for '{model_name}'"
        )

    hf_path = getattr(model_instance, "model_path", None)

    if not hf_path:
        raise RuntimeError(
            f"[NER CONFIG] Embedding model '{model_name}' has no `.model_path`"
        )

    return hf_path


# ----------------------------------------------------------------------
# INTENT MODEL
# ----------------------------------------------------------------------
def get_intent_model() -> str:
    name = AIModelsEmbeddingService.get_current_model_name()
    return _resolve_hf_path(name)


# ----------------------------------------------------------------------
# ENTITY MODEL
# ----------------------------------------------------------------------
def get_entity_model() -> str:
    name = AIModelsEmbeddingService.get_current_model_name()
    return _resolve_hf_path(name)


# ----------------------------------------------------------------------
# FALLBACK MODEL
# ----------------------------------------------------------------------
def get_fallback_model() -> str:
    name = AIModelsEmbeddingService.get_current_model_name()
    return _resolve_hf_path(name)


# ----------------------------------------------------------------------
# Debug print helper
# ----------------------------------------------------------------------
def debug_print():
    name = AIModelsEmbeddingService.get_current_model_name()
    path = _resolve_hf_path(name)

    print("NER CONFIGURATION (DB-driven):")
    print(f"  Active embedding model name: {name}")
    print(f"  HuggingFace directory path: {path}")
