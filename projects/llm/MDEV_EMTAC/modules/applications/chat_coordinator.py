# modules/application/chat_coordinator.py

from typing import Dict, Any
from modules.configuration.log_config import (
    with_request_id,
    info_id,
)
from modules.orchestrators.chat_orchestrator import ChatOrchestrator


class ChatCoordinator:
    """
    Application-layer coordinator for chatbot requests.

    Responsibilities:
        - Input validation
        - Control command handling
        - Response normalization
        - Delegation to orchestrator

    Does NOT:
        - Open sessions
        - Commit/rollback
        - Access ORM
    """

    MIN_QUESTION_LENGTH = 3

    def __init__(self):
        self.orchestrator = ChatOrchestrator()

    # ---------------------------------------------------------
    # Public Entry Point
    # ---------------------------------------------------------

    @with_request_id
    def process_question(
        self,
        *,
        user_id: str,
        question: str,
        client_type: str,
        request_id: str = None,
    ) -> Dict[str, Any]:

        info_id("ChatCoordinator.process_question called", request_id)

        question = (question or "").strip()

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        if len(question) < self.MIN_QUESTION_LENGTH:
            return self._normalize_response({
                "status": "invalid_input",
                "answer": "Please provide a more detailed question.",
            })

        # --------------------------------------------------
        # Control Commands
        # --------------------------------------------------
        if question.lower() == "end session please":
            return self._normalize_response({
                "status": "session_ended",
                "answer": "Session ended. Thank you for using the chatbot.",
                "redirect": "/logout",  # parity with legacy behavior
            })

        # --------------------------------------------------
        # Delegate to Orchestrator
        # --------------------------------------------------
        result = self.orchestrator.handle_question(
            user_id=user_id,
            question=question,
            client_type=client_type,
            request_id=request_id,
        )

        return self._normalize_response(result)

    # ---------------------------------------------------------
    # Response Normalization
    # ---------------------------------------------------------

    def _normalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Guarantees UI contract parity:
            - Always has blocks
            - Always has status
            - Never missing containers
        """

        if not isinstance(response, dict):
            response = {
                "status": "error",
                "answer": "Invalid response format.",
            }

        response.setdefault("status", "success")
        response.setdefault("answer", "")

        # Ensure blocks exist
        if "blocks" not in response or not isinstance(response["blocks"], dict):
            response["blocks"] = {}

        for key in (
            "parts-container",
            "images-container",
            "documents-container",
            "drawings-container",
        ):
            response["blocks"].setdefault(key, [])

        return response