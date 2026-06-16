"""
EMTAC RAG Diagnostic — verifies each stage of the RAG pipeline:

  ✔ Embedding model loads from DB (TinyLlamaEmbeddingModel)
  ✔ Query embedding vector produced
  ✔ pgvector retrieves rows correctly
  ✔ ContextBuilder constructs final context block
  ✔ Answer generator responds using DB-configured model (TinyLlamaModel, Qwen, etc.)
  ✔ Full pipeline runs end-to-end

Run:
    python test_rag_pipeline.py
"""

import pytest

from modules.emtac_ai.search.rag_core.embedder import DBConfiguredEmbedder
from modules.emtac_ai.search.rag_core.retriever import PgVectorRetriever
from modules.emtac_ai.search.rag_core.context_builder import ContextBuilder
from modules.emtac_ai.search.rag_core.answer_generator import DBConfiguredAnswerGenerator
from modules.emtac_ai.search.rag_core.rag_pipeline import RAGPipeline


# ------------------------------------------------------------
# Test question used for all stages
# ------------------------------------------------------------
TEST_QUESTION = "What steps are needed to replace a fill nozzle?"


# ------------------------------------------------------------
# FIXTURES FOR PYTEST
# ------------------------------------------------------------
@pytest.fixture
def query_vec():
    """Fixture: produce an embedding vector from current model."""
    embedder = DBConfiguredEmbedder()
    return embedder.embed_query(TEST_QUESTION)


@pytest.fixture
def chunks(query_vec):
    """Fixture: run retriever and return retrieved chunks."""
    retriever = PgVectorRetriever()
    return retriever.retrieve(query_embedding=query_vec, top_k=5)


@pytest.fixture
def context(chunks):
    """Fixture: generate context from chunks."""
    cb = ContextBuilder(max_tokens=4000)     # UPDATED
    ctx = cb.build_context(chunks)
    return ctx["context"]


# ------------------------------------------------------------
# TEST 1 — EMBEDDER
# ------------------------------------------------------------
def test_embedder():
    print("\n=== TEST 1: EMBEDDER (DBConfiguredEmbedder) ===")

    embedder = DBConfiguredEmbedder()
    vec = embedder.embed_query(TEST_QUESTION)

    print("Vector length:", len(vec))
    assert len(vec) > 0, "Embedding vector is empty"

    print("✔ Embedding vector produced successfully")
    return vec


# ------------------------------------------------------------
# TEST 2 — RETRIEVER
# ------------------------------------------------------------
def test_retriever(query_vec):
    print("\n=== TEST 2: RETRIEVER (PgVectorRetriever) ===")

    retriever = PgVectorRetriever()
    chunks = retriever.retrieve(query_embedding=query_vec, top_k=5)

    print(f"Retrieved chunks: {len(chunks)}")

    for i, ch in enumerate(chunks[:3]):
        print(f"  {i+1}. DocID={ch.get('document_id')}, distance={ch.get('distance')}")

    assert chunks is not None, "Retriever returned None"
    assert len(chunks) > 0, "Retriever returned 0 results"

    print("✔ Retriever returned results")
    return chunks


# ------------------------------------------------------------
# TEST 3 — CONTEXT BUILDER
# ------------------------------------------------------------
def test_context_builder(chunks):
    print("\n=== TEST 3: CONTEXT BUILDER ===")

    cb = ContextBuilder(max_tokens=4000)     # UPDATED
    ctx_obj = cb.build_context(chunks)

    context = ctx_obj["context"]
    used = ctx_obj["used_chunks"]

    print(f"Context length (chars): {len(context)}")
    print(f"Chunks used: {len(used)}")

    assert len(context) > 0, "Context is empty"
    print("✔ Context built successfully")

    return context


# ------------------------------------------------------------
# TEST 4 — ANSWER GENERATOR
# ------------------------------------------------------------
def test_answer_generator(context):
    print("\n=== TEST 4: ANSWER GENERATOR (LOCAL MODEL) ===")

    ag = DBConfiguredAnswerGenerator()
    result = ag.generate_answer(TEST_QUESTION, context=context)

    answer = result.get("answer", "")
    print("\nAnswer from model:\n", answer[:500], "\n")

    assert answer is not None
    assert answer.strip() != "", "Answer generator returned empty output"

    print("✔ Answer produced successfully")
    return answer


# ------------------------------------------------------------
# TEST 5 — FULL PIPELINE
# ------------------------------------------------------------
def test_full_pipeline():
    print("\n=== TEST 5: FULL PIPELINE (RAGPipeline.run()) ===")

    rag = RAGPipeline()
    out = rag.run(TEST_QUESTION, top_k=5)

    print("\nFinal Answer:\n", out["answer"][:500])
    print("\nContext chars:", len(out["context"]))
    print("Used chunks:", len(out["used_chunks"]))

    assert out["answer"], "Pipeline returned empty answer"
    assert len(out["used_chunks"]) > 0, "Pipeline retrieved no chunks"

    print("✔ Full RAG pipeline test complete")


# ------------------------------------------------------------
# MANUAL RUNNER
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n==================================================")
    print("        EMTAC RAG PIPELINE DIAGNOSTIC TEST        ")
    print("==================================================")

    vec = test_embedder()
    chunks = test_retriever(vec)
    context = test_context_builder(chunks)
    answer = test_answer_generator(context)
    test_full_pipeline()

    print("\nDONE.")
