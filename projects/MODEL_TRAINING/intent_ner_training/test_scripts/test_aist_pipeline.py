# tests/test_aist_pipeline.py
import pytest
from modules.emtac_ai.aist_manager import get_or_create_aist_manager

@pytest.fixture(scope="module")
def aist_manager():
    # Global AistManager (may or may not have DB session)
    return get_or_create_aist_manager()

@pytest.mark.integration
def test_pipeline_from_aist_manager(aist_manager):
    question = "Show me part 789 in area C"
    user_id = "pipeline-tester"

    # --- Run full pipeline
    result = aist_manager.answer_question(user_id=user_id, question=question)

    # --- Print/log for visibility
    intent_val = result.get("intent") or result.get("detected_intent")
    entities = result.get("entities", [])

    print(f"\n[AIST Pipeline Test] Question: {question}")
    print(f"[AIST Pipeline Test] Detected intent: {intent_val}")
    print(f"[AIST Pipeline Test] Extracted entities: {entities}")

    # --- Assertions to prove wiring
    assert isinstance(result, dict)
    assert intent_val is not None, "Intent was not detected"
    assert isinstance(entities, (list, dict)), "Entities missing"
    assert any("789" in str(e) or "area" in str(e).lower() for e in entities)

    # --- Tracking (stub or DB-backed)
    if getattr(aist_manager, "query_tracker", None):
        tracker = aist_manager.query_tracker
        if hasattr(tracker, "_queries"):  # stub tracker
            assert tracker._queries, "Tracker did not log queries"
        else:
            # DB-backed tracker is used → just confirm no crash
            assert tracker is not None
