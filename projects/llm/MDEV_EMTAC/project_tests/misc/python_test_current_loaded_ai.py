# python_test_current_loaded_ai.py

from modules.services import DBServices
from modules.ai.config.models_config import ModelsConfig

print("\n==============================")
print("=== AI SERVICE LOAD TEST =====")
print("==============================\n")

# ----------------------------------------------------------------
# 1. Check what the DB says the CURRENT_MODEL is for the LLM
# ----------------------------------------------------------------
llm_name = ModelsConfig.get_config_value("ai", "CURRENT_MODEL", default="NoAIModel")
print("DB says LLM model =", llm_name)

# ----------------------------------------------------------------
# 2. Load LLM through service layer
# ----------------------------------------------------------------
svc = DBServices().ai

try:
    llm = svc._load_model()
    if llm is None:
        print("\n✘ AIModelsService FAILED to load LLM")
    else:
        print(f"✔ AIModelsService loaded LLM: {llm.__class__.__name__}")

        # Optional: simple test prompt
        response = svc.answer("Say hello", "")
        print("\nModel Response:\n", response)

except Exception as e:
    print("\n✘ LLM load failed:", e)


# =================================================================
# === EMBEDDING MODEL TEST — VIA SERVICE LAYER ONLY ==============
# =================================================================

from modules.services import DBServices  # ensures access to service façade

print("\n==============================")
print("=== EMBEDDING MODEL LOAD TEST ===")
print("==============================\n")

# Access embedding service layer
embed_service = DBServices().embeddings

# Check DB config
embed_name = ModelsConfig.get_config_value("embedding", "CURRENT_MODEL", default="NoEmbeddingModel")
print("DB says embedding model =", embed_name)

try:
    embed_model = embed_service._load_model()

    if embed_model is None:
        print("\n✘ Embedding model FAILED to load (service layer returned None)")
    else:
        print(f"✔ Loaded embedding model: {embed_model.__class__.__name__}")
        print("Model Path:", getattr(embed_model, "model_path", "N/A"))
        print("Model Available:", embed_model.is_available())

        # Test vector generation
        vec = embed_service.get_embeddings("embedding test")
        print("Embedding vector length:", len(vec))

except Exception as e:
    print("\n✘ Embedding model load failed:", e)
