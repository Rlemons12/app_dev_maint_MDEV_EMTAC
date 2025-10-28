# tests/chat_flow/test_aist_manager_flow.py
import pytest

from modules.emtac_ai.aist_manager import AistManager
from modules.emtac_ai.search.UnifiedSearch import UnifiedSearch


@pytest.mark.parametrize("query", [
    "find TEST-123",
    "search valve in area 2",
])
def test_aist_manager_end_to_end(monkeypatch, query):
    """
    End-to-end sanity check:
    Question -> AistManager -> NLP -> Search -> Formatter -> Answer
    """

    # --- Stub backend search to avoid real DB calls ---
    def fake_search(self, text, *args, **kwargs):
        return {
            "status": "success",
            "answer": f"Found fake result for {text}",
            "results": [{"type": "part", "part_number": "TEST-123"}],
            "search_method": "orchestrator",
            "total_results": 1,
        }

    # ✅ Patch the actual class instead of string path
    monkeypatch.setattr(UnifiedSearch, "search", fake_search)

    # --- Initialize manager ---
    manager = AistManager()

    # --- Run query ---
    result = manager.answer_question(query, user_id="tester")

    # --- Assertions ---
    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert "answer" in result
    assert "results" in result
    assert result["results"]["total_results"] == 1
    assert result["results"]["search_method"] == "orchestrator"
