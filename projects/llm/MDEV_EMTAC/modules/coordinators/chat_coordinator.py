from __future__ import annotations

import inspect
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
        - Conversation ON/OFF pass-through for single-turn mode
        - Document scope pass-through for document-scoped conversation mode
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
            -> also returns conversation_id when conversation memory is enabled

        /ask/payload
            -> ChatPayloadCoordinator or route-level payload handler
            -> ChatPayloadOrchestrator.load_payload()
            -> returns documents/images/parts/drawings

    Document mode flow:
        /ask receives document_scope from the frontend
            -> ChatCoordinator.process_question(document_scope=...)
            -> ChatOrchestrator.handle_question(document_scope=...)
            -> later layers restrict retrieval to selected complete_document_id

    Conversation mode flow:
        conversation_enabled=True
            -> pass conversation_id normally
            -> orchestrator may create/use chat_sessions memory

        conversation_enabled=False
            -> force conversation_id=None
            -> orchestrator should skip chat_sessions memory
            -> every question is treated as single-turn / NEW_TOPIC
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
        document_scope: Optional[Dict[str, Any]] = None,
        conversation_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a user question.

        conversation_enabled:
            - True: normal current-session conversation memory is allowed.
            - False: force single-turn mode and do not pass conversation_id.

        conversation_id is optional:
            - If provided and conversation_enabled=True, the orchestrator should
              continue that conversation.
            - If missing and conversation_enabled=True, the orchestrator may create
              a new conversation/session.
            - If conversation_enabled=False, conversation_id is forced to None.

        document_scope is optional:
            - If provided, the orchestrator should pass it to the AI/RAG pathway.
            - The actual retrieval filter is applied later by UnifiedSearchService/RAG.
            - This coordinator validates only the lightweight shape needed for pass-through.
        """

        info_id("ChatCoordinator.process_question called", request_id)

        normalized_user_id = (user_id or "anonymous").strip() or "anonymous"
        normalized_question = (question or "").strip()
        normalized_client_type = (client_type or "web").strip().lower() or "web"
        normalized_conversation_enabled = bool(conversation_enabled)

        normalized_conversation_id = (
            (conversation_id or "").strip() or None
        )

        if not normalized_conversation_enabled:
            normalized_conversation_id = None

        normalized_document_scope = self._normalize_document_scope(document_scope)

        info_id(
            (
                "[ChatCoordinator] Conversation mode "
                f"conversation_enabled={normalized_conversation_enabled} "
                f"conversation_mode="
                f"{'conversation' if normalized_conversation_enabled else 'single_turn'} "
                f"conversation_id={normalized_conversation_id}"
            ),
            request_id,
        )

        if normalized_document_scope:
            info_id(
                (
                    "[ChatCoordinator] Document scope active "
                    f"complete_document_id={normalized_document_scope.get('complete_document_id')} "
                    f"document_name={normalized_document_scope.get('document_name')}"
                ),
                request_id,
            )

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
                "conversation_enabled": normalized_conversation_enabled,
                "conversation_mode": (
                    "conversation"
                    if normalized_conversation_enabled
                    else "single_turn"
                ),
                "memory_enabled": False,
                "memory_context_used": False,
                "document_scope": normalized_document_scope,
                "document_scope_enabled": bool(normalized_document_scope),
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
                "conversation_enabled": normalized_conversation_enabled,
                "conversation_mode": (
                    "conversation"
                    if normalized_conversation_enabled
                    else "single_turn"
                ),
                "memory_enabled": False,
                "memory_context_used": False,
                "document_scope": normalized_document_scope,
                "document_scope_enabled": bool(normalized_document_scope),
                "payload_status": "unavailable",
                "payload_endpoint": None,
            })

        result = self._call_orchestrator_handle_question(
            user_id=normalized_user_id,
            question=normalized_question,
            client_type=normalized_client_type,
            request_id=request_id,
            conversation_id=normalized_conversation_id,
            document_scope=normalized_document_scope,
            conversation_enabled=normalized_conversation_enabled,
        )

        if isinstance(result, dict):
            result.setdefault("conversation_enabled", normalized_conversation_enabled)
            result.setdefault(
                "conversation_mode",
                "conversation" if normalized_conversation_enabled else "single_turn",
            )

            if not normalized_conversation_enabled:
                result["conversation_id"] = None
                result.setdefault("memory_enabled", False)
                result.setdefault("memory_context_used", False)

            result.setdefault("document_scope", normalized_document_scope)
            result.setdefault("document_scope_enabled", bool(normalized_document_scope))

        return self._normalize_response(result)

    def _call_orchestrator_handle_question(
        self,
        *,
        user_id: str,
        question: str,
        client_type: str,
        request_id: Optional[str],
        conversation_id: Optional[str],
        document_scope: Optional[Dict[str, Any]],
        conversation_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Call ChatOrchestrator.handle_question with document_scope and
        conversation_enabled when supported.

        Safe step-by-step rollout:
            - If ChatOrchestrator supports conversation_enabled, forward it.
            - If not, force conversation_id=None when disabled and continue.
            - If ChatOrchestrator supports document_scope, forward it.
            - If not, continue without document_scope and log a warning.
        """

        normalized_conversation_enabled = bool(conversation_enabled)
        normalized_conversation_id = (
            conversation_id if normalized_conversation_enabled else None
        )

        kwargs: Dict[str, Any] = {
            "user_id": user_id,
            "question": question,
            "client_type": client_type,
            "request_id": request_id,
            "conversation_id": normalized_conversation_id,
        }

        try:
            signature = inspect.signature(self.orchestrator.handle_question)
            parameters = signature.parameters

            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )

            if "conversation_enabled" in parameters or accepts_kwargs:
                kwargs["conversation_enabled"] = normalized_conversation_enabled

                info_id(
                    (
                        "[ChatCoordinator] Forwarding conversation_enabled "
                        "to ChatOrchestrator "
                        f"conversation_enabled={normalized_conversation_enabled} "
                        f"conversation_id={normalized_conversation_id}"
                    ),
                    request_id,
                )

            elif not normalized_conversation_enabled:
                warning_id(
                    (
                        "[ChatCoordinator] conversation_enabled=False received but "
                        "ChatOrchestrator.handle_question does not accept "
                        "conversation_enabled yet. Continuing with conversation_id=None only."
                    ),
                    request_id,
                )

            if document_scope and ("document_scope" in parameters or accepts_kwargs):
                kwargs["document_scope"] = document_scope

                info_id(
                    (
                        "[ChatCoordinator] Forwarding document_scope to ChatOrchestrator "
                        f"complete_document_id={document_scope.get('complete_document_id')} "
                        f"document_name={document_scope.get('document_name')}"
                    ),
                    request_id,
                )

            elif document_scope:
                warning_id(
                    (
                        "[ChatCoordinator] document_scope received but "
                        "ChatOrchestrator.handle_question does not accept document_scope yet. "
                        "Continuing without scoped backend retrieval. "
                        f"complete_document_id={document_scope.get('complete_document_id')} "
                        f"document_name={document_scope.get('document_name')}"
                    ),
                    request_id,
                )

            return self.orchestrator.handle_question(**kwargs)

        except TypeError as type_error:
            warning_id(
                (
                    "[ChatCoordinator] ChatOrchestrator rejected extended call. "
                    "Retrying with legacy-safe kwargs. "
                    f"error={type_error}"
                ),
                request_id,
            )

            legacy_kwargs: Dict[str, Any] = {
                "user_id": user_id,
                "question": question,
                "client_type": client_type,
                "request_id": request_id,
                "conversation_id": normalized_conversation_id,
            }

            return self.orchestrator.handle_question(**legacy_kwargs)

        except (ValueError, AttributeError) as signature_error:
            warning_id(
                (
                    "[ChatCoordinator] Could not inspect ChatOrchestrator.handle_question "
                    "signature. Continuing with legacy-safe kwargs. "
                    f"error={signature_error}"
                ),
                request_id,
            )

            legacy_kwargs = {
                "user_id": user_id,
                "question": question,
                "client_type": client_type,
                "request_id": request_id,
                "conversation_id": normalized_conversation_id,
            }

            return self.orchestrator.handle_question(**legacy_kwargs)

    def _normalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize the answer-first response shape expected by the frontend.

        Expected answer-first response:

        {
            "status": "success",
            "answer": "...",
            "request_id": "...",
            "conversation_id": "...",
            "conversation_enabled": true | false,
            "conversation_mode": "conversation" | "single_turn",
            "memory_enabled": true | false,
            "memory_context_used": true | false,
            "document_scope": {...} | None,
            "document_scope_enabled": true | false,
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
                "conversation_enabled": True,
                "conversation_mode": "conversation",
                "memory_enabled": False,
                "memory_context_used": False,
                "document_scope": None,
                "document_scope_enabled": False,
                "payload_status": "unavailable",
                "payload_endpoint": None,
            }

        response.setdefault("status", "success")
        response.setdefault("answer", "")

        if response.get("answer") is None:
            response["answer"] = ""

        response.setdefault("request_id", None)

        raw_conversation_enabled = response.get(
            "conversation_enabled",
            response.get("conversationEnabled", None),
        )

        raw_conversation_mode = str(
            response.get("conversation_mode")
            or response.get("conversationMode")
            or ""
        ).strip().lower()

        if raw_conversation_enabled is None:
            conversation_enabled = raw_conversation_mode not in {
                "single_turn",
                "single-turn",
                "new_topic",
                "new-topic",
                "off",
                "disabled",
            }
        else:
            conversation_enabled = bool(raw_conversation_enabled)

        if raw_conversation_mode in {
            "single_turn",
            "single-turn",
            "new_topic",
            "new-topic",
            "off",
            "disabled",
        }:
            conversation_enabled = False

        response["conversation_enabled"] = conversation_enabled
        response["conversation_mode"] = (
            "conversation" if conversation_enabled else "single_turn"
        )

        if not conversation_enabled:
            response["conversation_id"] = None
            response["memory_enabled"] = False
            response["memory_context_used"] = False
        else:
            response.setdefault("conversation_id", None)
            response.setdefault("memory_enabled", bool(response.get("conversation_id")))
            response.setdefault("memory_context_used", False)

        normalized_document_scope = self._normalize_document_scope(
            response.get("document_scope") or response.get("documentScope")
        )

        response["document_scope"] = normalized_document_scope
        response["document_scope_enabled"] = bool(normalized_document_scope)

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

    @staticmethod
    def _normalize_document_scope(
        document_scope: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize document_scope before passing it deeper into the backend.

        Expected shape:
            {
                "enabled": true,
                "scope_type": "complete_document",
                "document_id": 29,
                "complete_document_id": 29,
                "document_name": "Document #29"
            }
        """

        if not document_scope or not isinstance(document_scope, dict):
            return None

        enabled = document_scope.get("enabled", True)

        if enabled is False:
            return None

        scope_type = (
            document_scope.get("scope_type")
            or document_scope.get("scopeType")
            or "complete_document"
        )

        scope_type = str(scope_type or "").strip() or "complete_document"

        if scope_type != "complete_document":
            return None

        complete_document_id = (
            document_scope.get("complete_document_id")
            or document_scope.get("completed_document_id")
            or document_scope.get("completeDocumentId")
            or document_scope.get("completeDocumentID")
        )

        complete_document_id = ChatCoordinator._coerce_int_or_none(
            complete_document_id
        )

        if complete_document_id is None:
            return None

        document_id = (
            document_scope.get("document_id")
            or document_scope.get("documentId")
        )

        document_id = ChatCoordinator._coerce_int_or_none(document_id)

        document_name = (
            document_scope.get("document_name")
            or document_scope.get("documentName")
            or document_scope.get("name")
            or document_scope.get("title")
            or f"Document #{complete_document_id}"
        )

        document_name = (
            str(document_name or "").strip()
            or f"Document #{complete_document_id}"
        )

        return {
            "enabled": True,
            "scope_type": "complete_document",
            "document_id": document_id,
            "complete_document_id": complete_document_id,
            "document_name": document_name,
        }

    @staticmethod
    def _coerce_int_or_none(value: Any) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        try:
            text = str(value).strip()

            if not text:
                return None

            return int(text)

        except (TypeError, ValueError):
            return None