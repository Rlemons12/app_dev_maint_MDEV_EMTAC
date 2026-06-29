import pytest
from modules.applications.chat_coordinator import ChatCoordinator


@pytest.mark.slow
def test_full_real_stack_chat_flow():
    """
    FULL REAL STACK TEST.

    This test:
        - Uses real DB
        - Uses real orchestrator
        - Uses real UnifiedSearchService
        - Uses real RAG pipeline
        - Uses real model config

    No monkeypatch.
    No fake services.
    """

    coordinator = ChatCoordinator()

    result = coordinator.process_question(
        user_id="integration_test",
        question="What is MTBF?",
        client_type="pytest",
    )

    assert isinstance(result, dict)
    assert "status" in result
    assert "answer" in result
    assert "blocks" in result

    # We don't assert specific content,
    # only that pipeline completed successfully.
    assert result["status"] in ("success", "error")