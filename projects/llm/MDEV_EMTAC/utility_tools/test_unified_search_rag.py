"""
UnifiedSearch RAG Integration Test
----------------------------------

This test verifies:

  ✔ UnifiedSearch initializes correctly
  ✔ RAG backend is called
  ✔ UnifiedSearch returns:
        - rag_answer
        - rag_chunks
        - results_by_type["documents"]
        - search_method="rag"

Run:
    pytest test_unified_search_rag.py -q
"""

import pytest

from modules.emtac_ai.search import UnifiedSearch


# ------------------------------------------------------------------------------
# FIXTURES
# ------------------------------------------------------------------------------
@pytest.fixture
def unified():
    """
    Build UnifiedSearch with:
      - orchestrator disabled
      - vector disabled
      - FTS disabled
      - RAG only
    """
    return UnifiedSearch(
        db_session=None,
        enable_vector=False,
        enable_fts=False,
        enable_orchestrator=False
    )


@pytest.fixture
def question():
    return "How do I replace a fill nozzle?"


# ------------------------------------------------------------------------------
# TEST 1 — Initialization
# ------------------------------------------------------------------------------
def test_unified_search_initializes(unified):
    assert unified is not None
    assert "rag" in unified.backends, "RAG backend was not initialized."


# ------------------------------------------------------------------------------
# TEST 2 — Execute search (RAG backend)
# ------------------------------------------------------------------------------
def test_unified_search_rag_pipeline(unified, question):
    result = unified.execute_unified_search(question)

    # --- basic checks ---
    assert result["status"] == "success"
    assert result["search_method"] == "rag"
    assert "rag_answer" in result
    assert "rag_chunks" in result

    # --- rag answer ---
    answer = result["rag_answer"]
    assert answer is not None
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0

    # --- chunks ---
    chunks = result["rag_chunks"]
    assert isinstance(chunks, list)
    assert len(chunks) > 0

    # Every chunk must have document metadata
    for ch in chunks:
        assert ch.get("type") == "document"
        assert "content" in ch
        assert "distance" in ch

    # --- unified bucket mapping ---
    buckets = result["results_by_type"]
    assert "documents" in buckets
    assert len(buckets["documents"]) == len(chunks)

    # --- final ---
    print("\nUnifiedSearch RAG output:")
    print("Answer:", answer[:250])
    print("Chunks:", len(chunks))
    print("Docs bucket:", len(buckets["documents"]))

    assert True  # if we reach this point, test passes
