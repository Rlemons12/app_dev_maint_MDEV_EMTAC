"""
test_ner_config.py

FULL DROP-IN TEST FOR ner_config.py

This script:
    - Uses project_tests/demos/bootstrap.py to fix sys.path
    - Ensures the MDEV_EMTAC project root is discoverable
    - Imports ner_config safely
    - Verifies DB model selection
    - Loads actual embedding model instance via ModelsConfig.load_embedding_model()
    - Verifies resolved HF model directory from model_instance.model_path
    - Performs HuggingFace tokenizer smoke test

Run:
    python test_ner_config.py
or:
    pytest project_tests/intent_ner_routers/test_ner_config.py
"""

import os
import sys
import traceback


# =============================================================================
# 1. LOAD BOOTSTRAP.PY TO FIX PYTHONPATH
# =============================================================================

def _bootstrap():
    """
    Load project_tests/demos/bootstrap.py and execute bootstrap_paths().
    Ensures project root is added to sys.path before imports.
    """

    test_dir = os.path.dirname(os.path.abspath(__file__))

    bootstrap_path = os.path.abspath(
        os.path.join(test_dir, "../demos/bootstrap.py")
    )

    if not os.path.exists(bootstrap_path):
        raise RuntimeError(
            f"bootstrap.py not found. Expected at:\n  {bootstrap_path}"
        )

    # Dynamically import bootstrap.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("bootstrap", bootstrap_path)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)

    # Call bootstrap_paths()
    project_root = bootstrap.bootstrap_paths()
    print(f"[BOOTSTRAP] Project root added: {project_root}")

    assert project_root in sys.path, "Bootstrap path insertion failed."


_bootstrap()


# =============================================================================
# 2. IMPORT AFTER BOOTSTRAP
# =============================================================================

from modules.emtac_ai.intent_ner.configuration import ner_config
from modules.services.ai_models_embedding_service import AIModelsEmbeddingService
from plugins.ai_modules.ai_models import ModelsConfig   # correct location


# =============================================================================
# 3. TEST LOGIC
# =============================================================================

def _print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _safe(label, value):
    print(f"{label:<40}: {value}")


def test_ner_config_main():
    """
    Main test function executed by pytest or standalone run.
    """
    _print_header("NER CONFIG TEST (DB-driven)")

    try:
        # ------------------------------------------------------------
        # 1. Read current embedding model from DB
        # ------------------------------------------------------------
        model_name = AIModelsEmbeddingService.get_current_model_name()
        _safe("Active model NAME (DB)", model_name)

        # ------------------------------------------------------------
        # 2. Load embedding model instance (REAL loader)
        # ------------------------------------------------------------
        embed_instance = ModelsConfig.load_embedding_model(model_name)
        if embed_instance is None:
            raise RuntimeError(
                f"Embedding model instance for '{model_name}' could not be loaded."
            )

        hf_path = getattr(embed_instance, "model_path", None)
        _safe("HF directory PATH (from embedding model)", hf_path)

        if not hf_path:
            raise RuntimeError(
                f"Embedding model '{model_name}' does not define a .model_path attribute."
            )

        if not os.path.exists(hf_path):
            raise FileNotFoundError(
                f"HF directory does NOT exist on disk:\n{hf_path}"
            )

        # ------------------------------------------------------------
        # 3. Validate ner_config mappings
        # ------------------------------------------------------------
        intent_path = ner_config.get_intent_model()
        entity_path = ner_config.get_entity_model()
        fallback_path = ner_config.get_fallback_model()

        _safe("ner_config.get_intent_model()", intent_path)
        _safe("ner_config.get_entity_model()", entity_path)
        _safe("ner_config.get_fallback_model()", fallback_path)

        # ------------------------------------------------------------
        # 4. Try loading a tokenizer from resolved HF directory
        # ------------------------------------------------------------
        _print_header("HUGGINGFACE TOKENIZER LOAD TEST")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(intent_path)

        _safe("Tokenizer loaded OK from", intent_path)

        # ------------------------------------------------------------
        # SUCCESS
        # ------------------------------------------------------------
        _print_header("NER CONFIG TEST PASSED ✓")

    except Exception:
        _print_header("NER CONFIG TEST FAILED ✗")
        traceback.print_exc()
        raise


# Allow execution without pytest
if __name__ == "__main__":
    test_ner_config_main()
