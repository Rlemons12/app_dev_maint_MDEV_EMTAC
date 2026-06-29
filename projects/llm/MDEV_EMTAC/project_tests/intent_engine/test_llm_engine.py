"""
test_llm_engine.py

FULL TEST FOR LLMENTityEngine
"""

import os
import sys
import traceback
import pytest


# ============================================================================
# 1. BOOTSTRAP PROJECT ROOT
# ============================================================================
def _bootstrap():
    """Load bootstrap.py to fix sys.path so imports succeed."""

    test_dir = os.path.dirname(os.path.abspath(__file__))
    bootstrap_path = os.path.abspath(
        os.path.join(test_dir, "../demos/bootstrap.py")
    )

    if not os.path.exists(bootstrap_path):
        raise RuntimeError(f"bootstrap.py not found at: {bootstrap_path}")

    # *** FIXED HERE ***
    import importlib.util
    spec = importlib.util.spec_from_file_location("bootstrap", bootstrap_path)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)

    project_root = bootstrap.bootstrap_paths()
    print(f"[BOOTSTRAP] Project root added: {project_root}")

    assert project_root in sys.path, "Bootstrap path insertion failed."


_bootstrap()


# ============================================================================
# 2. IMPORTS AFTER BOOTSTRAP
# ============================================================================
from modules.emtac_ai.intent_ner.engine.llm_engine import LLMENTityEngine
from modules.services.ai_models_embedding_service import AIModelsEmbeddingService


# ============================================================================
# 3. UTIL HELPERS
# ============================================================================
def _print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# 4. TESTS
# ============================================================================
def test_engine_loads_tokenizer_and_model():
    _print_header("ENGINE LOAD TEST")

    try:
        engine = LLMENTityEngine(mode="intent")

        assert engine.tokenizer is not None
        print("✔ Tokenizer loaded")

        assert engine.model is not None
        print(f"✔ Model loaded ({engine.model_type})")

        assert engine.model_type in ("causal_lm", "encoder")
        print(f"✔ Valid model type: {engine.model_type}")

    except Exception:
        _print_header("ENGINE LOAD TEST FAILED ✗")
        traceback.print_exc()
        raise


def test_encoder_model_rejects_generate_if_not_causal():
    _print_header("ENCODER SAFETY TEST")

    engine = LLMENTityEngine(mode="intent")

    if engine.model_type == "causal_lm":
        pytest.skip("Model is causal LM — skipping encoder safety test.")

    with pytest.raises(RuntimeError):
        engine._generate("hello")

    print("✔ Encoder-only model correctly rejected _generate()")


def test_causal_model_generate_if_available():
    _print_header("CAUSAL LM GENERATION TEST")

    model_name = AIModelsEmbeddingService.get_current_model_name()
    print("DB Model:", model_name)

    engine = LLMENTityEngine(mode="intent")

    if engine.model_type != "causal_lm":
        pytest.skip("Encoder-only model — cannot test generate()")

    output = engine._generate("Say hello:", max_tokens=8)
    print("Generated:", output)

    assert isinstance(output, str) and len(output) > 0
    print("✔ Causal LM generate() passed")


# ============================================================================
# 5. ALLOW DIRECT EXECUTION
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-q"])
