# tests/test_chatbot_flow_mocked.py
import pytest
from unittest.mock import MagicMock, patch
from modules.emtac_ai.aist_manager import AistManager


@pytest.fixture
def mocked_manager():
    # Patch UnifiedSearch.execute_unified_search so we don't hit DB/models
    with patch.object(AistManager, "execute_unified_search", return_value={
        "status": "success",
        "answer": "Found 1 test result",
        "results": [{"type": "part", "part_number": "TEST-123"}],
        "search_method": "orchestrator",
        "total_results": 1
    }) as mock_search:
        manager = AistManager(ai_model=MagicMock(), db_session=MagicMock())
        yield manager, mock_search


def test_question_follows_path(mocked_manager):
    manager, mock_search = mocked_manager

    # --- Act ---
    result = manager.answer_question(user_id="tester", question="find TEST-123")

    # --- Assert pipeline flow ---
    mock_search.assert_called_once()  # ensures UnifiedSearch was used
    assert result["status"] == "success"
    assert "answer" in result
    assert "results" in result
    assert result["method"] in ["orchestrator", "vector", "fts", "regex"]

    print("\n--- MOCKED PIPELINE RESULT ---")
    print(result)
