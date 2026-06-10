# modules/services/intent/chat_intent_prompt_service.py

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ChatIntentPromptService:
    """
    Compatibility service for chat intent context preparation.

    IMPORTANT:
    The production intent pathway now uses the local DistilBERT classifier.

    This service no longer builds a JSON-only LLM prompt because intent
    classification should not call:
      - AIModelsService
      - AIStewardManagerService
      - RAG/search
      - local LLM generation

    It remains here so older imports do not break.
    """

    MAX_RECENT_MESSAGES = 6
    MAX_CHARS_PER_MESSAGE = 500
    MAX_SUMMARY_ITEMS = 4
    MAX_SUMMARY_QUESTION_CHARS = 350
    MAX_SUMMARY_ANSWER_CHARS = 450

    def build_prompt(
        self,
        *,
        question: str,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        conversation_summary: Optional[List[Dict[str, Any]]] = None,
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Backward-compatible method.

        DistilBERT does not need this prompt. Return only the cleaned question
        so any old caller still gets a safe string instead of the old giant
        JSON classifier prompt.
        """

        return (question or "").strip()

    def build_context(
        self,
        *,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        conversation_summary: Optional[List[Dict[str, Any]]] = None,
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Optional helper for the DistilBERT classifier/orchestrator.

        The classifier mainly uses the raw question, but these fields are useful
        for rule checks and future context-aware routing.
        """

        return {
            "recent_messages": self._safe_list(recent_messages),
            "conversation_summary": self._safe_list(conversation_summary),
            "document_scope": document_scope if isinstance(document_scope, dict) else None,
            "recent_messages_text": self._render_recent_messages(
                self._safe_list(recent_messages)
            ),
            "conversation_summary_text": self._render_summary(
                self._safe_list(conversation_summary)
            ),
            "document_scope_text": self._render_document_scope(document_scope),
        }

    def _render_recent_messages(self, messages: List[Dict[str, Any]]) -> str:
        items = messages[-self.MAX_RECENT_MESSAGES:]
        lines: List[str] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            role = str(item.get("role") or "unknown").strip() or "unknown"
            content = str(item.get("content") or "").strip()

            if not content:
                continue

            lines.append(f"{role}: {self._clip(content, self.MAX_CHARS_PER_MESSAGE)}")

        return "\n".join(lines) if lines else "(none)"

    def _render_summary(self, summaries: List[Dict[str, Any]]) -> str:
        lines: List[str] = []

        for item in summaries[-self.MAX_SUMMARY_ITEMS:]:
            if not isinstance(item, dict):
                continue

            q = self._clip(
                str(item.get("question") or ""),
                self.MAX_SUMMARY_QUESTION_CHARS,
            )
            a = self._clip(
                str(
                    item.get("answer_preview")
                    or item.get("answer")
                    or item.get("summary")
                    or ""
                ),
                self.MAX_SUMMARY_ANSWER_CHARS,
            )

            if q or a:
                lines.append(f"- Question: {q}\n  Answer summary: {a}")

        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def _render_document_scope(document_scope: Optional[Dict[str, Any]]) -> str:
        if not isinstance(document_scope, dict):
            return "(none)"

        enabled = bool(document_scope.get("enabled", True))
        document_name = (
            document_scope.get("document_name")
            or document_scope.get("documentName")
            or document_scope.get("name")
            or document_scope.get("title")
        )
        complete_document_id = (
            document_scope.get("complete_document_id")
            or document_scope.get("completeDocumentId")
            or document_scope.get("completeDocumentID")
            or document_scope.get("document_id")
            or document_scope.get("documentId")
        )

        return (
            f"active={enabled}, "
            f"document_name={document_name}, "
            f"complete_document_id={complete_document_id}"
        )

    @staticmethod
    def _safe_list(value: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if isinstance(value, list):
            return list(value)
        return []

    @staticmethod
    def _clip(value: str, max_chars: int) -> str:
        value = (value or "").strip()

        if len(value) <= max_chars:
            return value

        return value[: max_chars - 3].rstrip() + "..."