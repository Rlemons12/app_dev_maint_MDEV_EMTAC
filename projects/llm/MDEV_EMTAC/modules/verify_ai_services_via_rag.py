# modules/verify_ai_services_via_rag.py

"""
End-to-end verification script for EMTAC AI services.

This script:
- Uses RAGPipeline as the entry point
- Exercises:
    - Embedding service
    - Retrieval (pgvector)
    - LLM answer generation
    - Image service (direct call)
- Works with both local and gpu_service backends
"""

from configuration.log_config import (
    info_id,
    error_id,
    debug_id,
    set_request_id,
    get_request_id,
)

from emtac_ai.search.rag_core import RAGPipeline
from plugins.ai_modules.services.ai_model_image_service import AIModelImageService
from plugins.ai_modules.services.ai_models_embedding_service import (
    AIModelsEmbeddingService,
)
from modules.ai.config.models_config import ModelsConfig


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------
# MAIN TEST
# ---------------------------------------------------------
def main():
    # Create request context
    set_request_id()
    rid = get_request_id()

    banner("EMTAC AI SERVICES VERIFICATION (RAG ENTRY POINT)")
    info_id("Starting verification run", rid)

    # -----------------------------------------------------
    # Show current DB configuration
    # -----------------------------------------------------
    ai_backend = ModelsConfig.get_execution_backend("ai")
    emb_backend = ModelsConfig.get_execution_backend("embedding")
    img_backend = ModelsConfig.get_execution_backend("image")

    info_id(f"[CONFIG] AI backend        = {ai_backend}", rid)
    info_id(f"[CONFIG] Embedding backend = {emb_backend}", rid)
    info_id(f"[CONFIG] Image backend     = {img_backend}", rid)

    info_id(
        f"[CONFIG] AI model        = {ModelsConfig.get_config_value('ai', 'CURRENT_MODEL')}",
        rid,
    )
    info_id(
        f"[CONFIG] Embedding model = {ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL')}",
        rid,
    )
    info_id(
        f"[CONFIG] Image model     = {ModelsConfig.get_config_value('image', 'CURRENT_MODEL')}",
        rid,
    )

    # -----------------------------------------------------
    # 1. RAG PIPELINE TEST (TEXT + EMBEDDINGS)
    # -----------------------------------------------------
    banner("1) RAG PIPELINE TEST")

    rag = RAGPipeline()

    question = "What does this equipment do and what is its primary function?"

    try:
        result = rag.run(
            question=question,
            top_k=3,
            request_id=rid,
        )
    except Exception as e:
        error_id(f"[RAG TEST] FAILED: {e}", rid)
        raise

    answer = result.get("answer", "")
    used_chunks = result.get("used_chunks", [])

    info_id("[RAG TEST] Completed successfully", rid)
    info_id(f"[RAG TEST] Answer length = {len(answer)} chars", rid)
    info_id(f"[RAG TEST] Retrieved chunks = {len(used_chunks)}", rid)

    print("\n--- RAG ANSWER PREVIEW ---")
    print(answer[:500])
    print("--- END ANSWER PREVIEW ---")

    # -----------------------------------------------------
    # 2. DIRECT EMBEDDING TEST
    # -----------------------------------------------------
    banner("2) DIRECT EMBEDDING SERVICE TEST")

    test_text = "Hydraulic pressure sensor mounted near the pump assembly."

    try:
        vec = AIModelsEmbeddingService.get_embeddings(
            test_text,
            request_id=rid,
        )
    except Exception as e:
        error_id(f"[EMBEDDING TEST] FAILED: {e}", rid)
        raise

    info_id(f"[EMBEDDING TEST] Vector length = {len(vec)}", rid)

    # -----------------------------------------------------
    # 3. IMAGE SERVICE TEST (NON-STRICT)
    # -----------------------------------------------------
    banner("3) IMAGE SERVICE TEST")

    # These paths do not need to exist to validate routing,
    # but real images will produce better results.
    image_a = "tests/assets/sample_image_a.jpg"
    image_b = "tests/assets/sample_image_b.jpg"

    try:
        desc = AIModelImageService.generate_description(
            image_a,
            request_id=rid,
        )
        info_id("[IMAGE TEST] generate_description() OK", rid)
        debug_id(f"[IMAGE TEST] description={desc}", rid)
    except Exception as e:
        error_id(f"[IMAGE TEST] generate_description FAILED: {e}", rid)

    try:
        cmp_result = AIModelImageService.compare_images(
            image_a,
            image_b,
            request_id=rid,
        )
        info_id("[IMAGE TEST] compare_images() OK", rid)
        debug_id(f"[IMAGE TEST] result={cmp_result}", rid)
    except Exception as e:
        error_id(f"[IMAGE TEST] compare_images FAILED: {e}", rid)

    # -----------------------------------------------------
    # DONE
    # -----------------------------------------------------
    banner("VERIFICATION COMPLETE")
    info_id("All verification steps executed", rid)


if __name__ == "__main__":
    main()
