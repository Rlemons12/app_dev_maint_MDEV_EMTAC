from __future__ import annotations

from typing import Dict, Any, Optional

from modules.configuration.log_config import (
    with_request_id,
    info_id,
    warning_id,
)

from modules.orchestrators.chat_orchestrator import ChatOrchestrator


class ChatCoordinator:
    """
    Application-layer coordinator for chatbot answer requests.

    This coordinator is responsible for the ANSWER-FIRST pathway.

    Responsibilities:
        - Input validation
        - Control command handling
        - Conversation ID pass-through for conversational memory
        - Delegation to ChatOrchestrator.handle_question()
        - Response normalization for the frontend

    Does NOT:
        - Open sessions
        - Commit/rollback
        - Access ORM directly
        - Import analytics/tracking services
        - Build documents/images/parts/drawings payload
        - Call ChatPayloadOrchestrator directly

    Flow:
        /ask
            -> ChatCoordinator.process_question()
            -> ChatOrchestrator.handle_question()
            -> returns answer immediately with empty payload containers
            -> also returns conversation_id for continued memory

        /ask/payload
            -> ChatPayloadCoordinator or route-level payload handler
            -> ChatPayloadOrchestrator.load_payload()
            -> returns documents/images/parts/drawings
    """

    MIN_QUESTION_LENGTH = 3

    EMPTY_BLOCKS = {
        "documents-container": [],
        "parts-container": [],
        "images-container": [],
        "drawings-container": [],
    }

    TOP_LEVEL_TO_BLOCK_KEY = {
        "documents": "documents-container",
        "parts": "parts-container",
        "images": "images-container",
        "drawings": "drawings-container",
    }

    CONTROL_COMMANDS = {
        "end session please",
        "end session",
        "logout",
        "log out",
    }

    def __init__(
        self,
        *,
        orchestrator: Optional[ChatOrchestrator] = None,
    ):
        """
        Dependency injection is optional.

        Production usage:
            ChatCoordinator()

        Test usage:
            ChatCoordinator(orchestrator=fake_orchestrator)
        """

        self.orchestrator = orchestrator or ChatOrchestrator()

    @with_request_id
    def process_question(
        self,
        *,
        user_id: str,
        question: str,
        client_type: str,
        request_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user question.

        conversation_id is optional:
            - If provided, the orchestrator should continue that conversation.
            - If missing, the orchestrator should create a new conversation/session.
            - The response should always return the active conversation_id.
        """

        info_id("ChatCoordinator.process_question called", request_id)

        normalized_user_id = (user_id or "anonymous").strip() or "anonymous"
        normalized_question = (question or "").strip()
        normalized_client_type = (client_type or "web").strip().lower() or "web"
        normalized_conversation_id = (conversation_id or "").strip() or None

        if len(normalized_question) < self.MIN_QUESTION_LENGTH:
            warning_id(
                "ChatCoordinator rejected question: input too short",
                request_id,
            )

            return self._normalize_response({
                "status": "invalid_input",
                "answer": "Please provide a more detailed question.",
                "request_id": request_id,
                "conversation_id": normalized_conversation_id,
                "payload_status": "unavailable",
                "payload_endpoint": None,
            })

        if normalized_question.lower() in self.CONTROL_COMMANDS:
            info_id(
                "ChatCoordinator received session termination command",
                request_id,
            )

            return self._normalize_response({
                "status": "session_ended",
                "answer": "Session ended. Thank you for using the chatbot.",
                "redirect": "/logout",
                "request_id": request_id,
                "conversation_id": normalized_conversation_id,
                "payload_status": "unavailable",
                "payload_endpoint": None,
            })

        result = self.orchestrator.handle_question(
            user_id=normalized_user_id,
            question=normalized_question,
            client_type=normalized_client_type,
            request_id=request_id,
            conversation_id=normalized_conversation_id,
        )

        return self._normalize_response(result)

    def _normalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize the answer-first response shape expected by the frontend.

        Expected answer-first response:

        {
            "status": "success",
            "answer": "...",
            "request_id": "...",
            "conversation_id": "...",
            "payload_status": "pending",
            "payload_endpoint": "/ask/payload",
            "blocks": {
                "documents-container": [],
                "parts-container": [],
                "images-container": [],
                "drawings-container": []
            },
            "documents": [],
            "parts": [],
            "images": [],
            "drawings": []
        }

        This method also preserves backward compatibility if an older service
        still returns top-level documents/images/parts/drawings.
        """

        if not isinstance(response, dict):
            response = {
                "status": "error",
                "answer": "Invalid response format.",
                "request_id": None,
                "conversation_id": None,
                "payload_status": "unavailable",
                "payload_endpoint": None,
            }

        response.setdefault("status", "success")
        response.setdefault("answer", "")

        if response.get("answer") is None:
            response["answer"] = ""

        response.setdefault("request_id", None)
        response.setdefault("conversation_id", None)

        response.setdefault("payload_status", "pending")
        response.setdefault("payload_endpoint", "/ask/payload")

        if response.get("status") in {"error", "invalid_input", "session_ended"}:
            response["payload_status"] = response.get("payload_status") or "unavailable"

            if response["payload_status"] == "unavailable":
                response["payload_endpoint"] = None

        if "blocks" not in response or not isinstance(response["blocks"], dict):
            response["blocks"] = {}

        for block_key in self.EMPTY_BLOCKS:
            response["blocks"].setdefault(block_key, [])

            if response["blocks"][block_key] is None:
                response["blocks"][block_key] = []

        for top_level_key in self.TOP_LEVEL_TO_BLOCK_KEY:
            response.setdefault(top_level_key, [])

            if response[top_level_key] is None:
                response[top_level_key] = []

        # Backward compatibility:
        # If an older service still returns top-level payload data, promote it
        # into the frontend blocks. In the new answer-first path, these should
        # normally remain empty until /ask/payload is called.
        for top_level_key, block_key in self.TOP_LEVEL_TO_BLOCK_KEY.items():
            top_level_items = response.get(top_level_key) or []

            if top_level_items and not response["blocks"].get(block_key):
                response["blocks"][block_key] = top_level_items

        return response