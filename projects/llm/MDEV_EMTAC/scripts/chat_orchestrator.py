from __future__ import annotations

import inspect
import json
import time
import re
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from uuid import UUID

from modules.observability.high_end_tracer import tracer
from modules.orchestrators.base_orchestrator import BaseOrchestrator

from modules.services.ai_steward_manager_service import AIStewardManagerService
from modules.services.qanda_service import QandAService
from modules.services.qanda_embedding_service import QandAEmbeddingService

from modules.coordinators.chat_intent_coordinator import ChatIntentCoordinator
from modules.intent.intent_types import ChatIntent, ChatIntentDecision

from modules.ai.search_pathway.audit import (
    SearchAuditService,
    SearchPathwayName,
    get_search_audit_log_manager,
)

from modules.configuration.config import (
    FORCE_DEBUG_CHUNK,
    FORCE_DEBUG_CHUNK_ID,
)

from modules.configuration.log_config import (
    with_request_id,
    info_id,
    warning_id,
    error_id,
    debug_id,
)

from modules.emtacdb.emtacdb_fts import QandA, ChatSession


class ChatOrchestrator(BaseOrchestrator):
    """
    Answer-only chat orchestrator with intent-controlled memory routing.
    """

    DEFAULT_METHOD = "rag"
    AUDIT_PATHWAY_NAME = SearchPathwayName.RAG.value
    AUDIT_PATHWAY_VERSION = "1.1-intent-memory-routing"

    CHAT_SESSION_MAX_MESSAGES = 100
    MEMORY_RECENT_MESSAGE_LIMIT = 10
    MEMORY_SUMMARY_LIMIT = 8
    MEMORY_PREVIEW_CHARS = 1000
    SUMMARY_ANSWER_PREVIEW_CHARS = 1500
    SUMMARY_MAX_ITEMS = 20

    RECALL_SESSION_SENTINEL = "NO_SESSION_MEMORY_MATCH"
    RECALL_QANDA_SENTINEL = "NO_QANDA_RECALL_MATCH"
    QANDA_RECALL_TOP_K = 5
    QANDA_RECALL_MIN_SIMILARITY = 0.45

    EMPTY_BLOCKS = {
        "documents-container": [],
        "parts-container": [],
        "images-container": [],
        "drawings-container": [],
    }

    def __init__(
        self,
        *,
        ai_service: Optional[AIStewardManagerService] = None,
        qanda_service: Optional[QandAService] = None,
        qanda_embedding_service: Optional[QandAEmbeddingService] = None,
        intent_coordinator: Optional[ChatIntentCoordinator] = None,
    ):
        super().__init__()

        self.ai_service = ai_service or AIStewardManagerService()
        self.qanda_service = qanda_service or QandAService()
        self.qanda_embedding_service = qanda_embedding_service or QandAEmbeddingService()
        self.intent_coordinator = intent_coordinator or ChatIntentCoordinator()
        self.audit_log_manager = get_search_audit_log_manager()

    @with_request_id
    def handle_question(
        self,
        *,
        user_id: str,
        question: str,
        client_type: str,
        request_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Request-level chat answer orchestration.

        Important:
        - Do NOT use @trace_entrypoint here.
        - The Flask route owns the root TraceSession.
        - This method creates a child span so the dashboard can map the
          answer-first pathway without creating duplicate root traces.
        """

        normalized_user_id = (user_id or "anonymous").strip() or "anonymous"
        normalized_question = (question or "").strip()
        normalized_client_type = (client_type or "web").strip().lower() or "web"
        normalized_conversation_id = (conversation_id or "").strip() or None
        normalized_document_scope = self._normalize_document_scope(document_scope)
        document_scope_enabled = bool(normalized_document_scope)

        with tracer.span(
            "chat.orchestrator.handle_question",
            meta={
                "request_id": request_id,
                "user_id": normalized_user_id,
                "client_type": normalized_client_type,
                "incoming_conversation_id": normalized_conversation_id,
                "question_chars": len(normalized_question),
                "document_scope_enabled": document_scope_enabled,
                "complete_document_id": (
                    normalized_document_scope.get("complete_document_id")
                    if normalized_document_scope
                    else None
                ),
                "document_name": (
                    normalized_document_scope.get("document_name")
                    if normalized_document_scope
                    else None
                ),
            },
        ):
            try:
                tracer.event(
                    "chat_orchestrator_input_received",
                    {
                        "request_id": request_id,
                        "user_id": normalized_user_id,
                        "client_type": normalized_client_type,
                        "incoming_conversation_id": normalized_conversation_id,
                        "question_chars": len(normalized_question),
                        "document_scope_enabled": document_scope_enabled,
                        "complete_document_id": (
                            normalized_document_scope.get("complete_document_id")
                            if normalized_document_scope
                            else None
                        ),
                        "document_name": (
                            normalized_document_scope.get("document_name")
                            if normalized_document_scope
                            else None
                        ),
                    },
                )
            except Exception:
                pass

            result = self._handle_question_impl(
                user_id=normalized_user_id,
                question=normalized_question,
                client_type=normalized_client_type,
                request_id=request_id,
                conversation_id=normalized_conversation_id,
                document_scope=normalized_document_scope,
            )

            try:
                result_document_scope = self._normalize_document_scope(
                    result.get("document_scope")
                    if isinstance(result, dict)
                    else None
                )

                tracer.event(
                    "chat_orchestrator_response_ready",
                    {
                        "request_id": request_id,
                        "status": (
                            result.get("status")
                            if isinstance(result, dict)
                            else None
                        ),
                        "conversation_id": (
                            result.get("conversation_id")
                            if isinstance(result, dict)
                            else None
                        ),
                        "method": (
                            result.get("method")
                            if isinstance(result, dict)
                            else None
                        ),
                        "strategy": (
                            result.get("strategy")
                            if isinstance(result, dict)
                            else None
                        ),
                        "payload_status": (
                            result.get("payload_status")
                            if isinstance(result, dict)
                            else None
                        ),
                        "document_scope_enabled": bool(result_document_scope),
                        "complete_document_id": (
                            result_document_scope.get("complete_document_id")
                            if result_document_scope
                            else None
                        ),
                    },
                )
            except Exception:
                pass

            return result

    def _handle_question_impl(
        self,
        *,
        user_id: str,
        question: str,
        client_type: str,
        request_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        request_start = time.perf_counter()

        memory_time = 0.0
        ai_time = 0.0
        persist_time = 0.0
        embedding_time = 0.0
        audit_time = 0.0

        ai_result: Dict[str, Any] = {}
        audit_summary: Dict[str, Any] = {}
        qanda_id: Optional[Any] = None
        embeddings_updated = False

        active_conversation_id: Optional[str] = None
        memory_context_text = ""
        memory_context_used = False
        intent_decision: Optional[ChatIntentDecision] = None

        normalized_user_id = (user_id or "anonymous").strip() or "anonymous"
        normalized_question = (question or "").strip()
        normalized_client_type = (client_type or "web").strip().lower() or "web"
        normalized_conversation_id = (conversation_id or "").strip() or None
        normalized_document_scope = self._normalize_document_scope(document_scope)
        document_scope_enabled = bool(normalized_document_scope)

        self.audit_log_manager.log_run_start(
            request_id=request_id or "unknown",
            pathway_name=self.AUDIT_PATHWAY_NAME,
            question=normalized_question,
        )

        if normalized_document_scope:
            info_id(
                f"[ChatOrchestrator] Document scope active "
                f"complete_document_id={normalized_document_scope.get('complete_document_id')} "
                f"document_name={normalized_document_scope.get('document_name')}",
                request_id,
            )

        try:
            forced_chunk_id = self._resolve_forced_chunk_id(request_id=request_id)

            memory_start = time.perf_counter()

            try:
                with self.transaction() as session:
                    with tracer.span(
                        "chat.orchestrator.memory_prepare",
                        meta={
                            "request_id": request_id,
                            "incoming_conversation_id": normalized_conversation_id,
                            "user_id": normalized_user_id,
                            "document_scope_enabled": document_scope_enabled,
                            "complete_document_id": (
                                normalized_document_scope.get("complete_document_id")
                                if normalized_document_scope
                                else None
                            ),
                        },
                    ):
                        chat_session, created_session = self._get_or_create_chat_session(
                            session=session,
                            conversation_id=normalized_conversation_id,
                            user_id=normalized_user_id,
                            request_id=request_id,
                        )

                        active_conversation_id = str(chat_session.session_id)

                        self._append_chat_message(
                            chat_session=chat_session,
                            role="user",
                            content=normalized_question,
                            request_id=request_id,
                            metadata={
                                "client_type": normalized_client_type,
                                "created_session": created_session,
                                "document_scope": normalized_document_scope,
                                "document_scope_enabled": document_scope_enabled,
                            },
                        )

                        try:
                            with tracer.span(
                                "chat.orchestrator.intent_classify",
                                meta={
                                    "request_id": request_id,
                                    "conversation_id": active_conversation_id,
                                    "document_scope_enabled": document_scope_enabled,
                                    "complete_document_id": (
                                        normalized_document_scope.get("complete_document_id")
                                        if normalized_document_scope
                                        else None
                                    ),
                                },
                            ):
                                intent_decision = self.intent_coordinator.classify_question(
                                    question=normalized_question,
                                    chat_session=chat_session,
                                    document_scope=normalized_document_scope,
                                    request_id=request_id,
                                )
                                try:
                                    tracer.event(
                                        "chat_intent_decision",
                                        {
                                            "request_id": request_id,
                                            "conversation_id": active_conversation_id,
                                            "question": normalized_question,
                                            "intent": intent_decision.intent.value if intent_decision else None,
                                            "confidence": (
                                                float(intent_decision.confidence)
                                                if intent_decision and intent_decision.confidence is not None
                                                else None
                                            ),
                                            "needs_current_session_memory": (
                                                bool(intent_decision.needs_current_session_memory)
                                                if intent_decision
                                                else False
                                            ),
                                            "needs_semantic_chat_recall": (
                                                bool(intent_decision.needs_semantic_chat_recall)
                                                if intent_decision
                                                else False
                                            ),
                                            "needs_document_scope": (
                                                bool(intent_decision.needs_document_scope)
                                                if intent_decision
                                                else False
                                            ),
                                            "rewritten_question": (
                                                intent_decision.rewritten_question
                                                if intent_decision
                                                else None
                                            ),
                                            "reason": (
                                                intent_decision.reason
                                                if intent_decision
                                                else None
                                            ),
                                            "document_scope_enabled": document_scope_enabled,
                                            "complete_document_id": (
                                                normalized_document_scope.get("complete_document_id")
                                                if normalized_document_scope
                                                else None
                                            ),
                                        },
                                    )
                                except Exception:
                                    pass


                            debug_id(
                                "[ChatOrchestrator] Intent pathway returned "
                                f"intent={intent_decision.intent.value} "
                                f"confidence={intent_decision.confidence:.2f} "
                                f"needs_current_session_memory={intent_decision.needs_current_session_memory} "
                                f"needs_semantic_chat_recall={intent_decision.needs_semantic_chat_recall} "
                                f"needs_document_scope={intent_decision.needs_document_scope} "
                                f"rewritten_question={intent_decision.rewritten_question!r} "
                                f"reason={intent_decision.reason!r}",
                                request_id,
                            )

                        except Exception as intent_error:
                            warning_id(
                                "[ChatOrchestrator] Intent classification failed; "
                                "continuing as NEW_TOPIC. "
                                f"Error: {intent_error}",
                                request_id,
                                exc_info=True,
                            )
                            intent_decision = ChatIntentDecision.fallback_new_topic(
                                normalized_question
                            )

                        should_use_memory = self._should_use_memory_for_intent(
                            intent_decision=intent_decision,
                            document_scope_enabled=document_scope_enabled,
                        )

                        if should_use_memory:
                            memory_context_text = self._build_memory_context_text(
                                chat_session=chat_session,
                            )
                            memory_context_used = bool(memory_context_text.strip())
                        else:
                            memory_context_text = ""
                            memory_context_used = False

                        memory_context_chars = len(memory_context_text or "")
                        estimated_memory_tokens = (
                            int(memory_context_chars / 4)
                            if memory_context_chars
                            else 0
                        )

                        session_messages = self._safe_list(
                            getattr(chat_session, "session_data", None)
                        )
                        summary_items = self._safe_list(
                            getattr(chat_session, "conversation_summary", None)
                        )

                        debug_id(
                            "[ChatOrchestrator] Intent-controlled memory decision "
                            f"conversation_id={active_conversation_id} "
                            f"created_session={created_session} "
                            f"intent={intent_decision.intent.value if intent_decision else None} "
                            f"should_use_memory={should_use_memory} "
                            f"memory_context_used={memory_context_used} "
                            f"memory_context_chars={memory_context_chars} "
                            f"estimated_memory_tokens={estimated_memory_tokens} "
                            f"session_messages_count={len(session_messages)} "
                            f"summary_items_count={len(summary_items)} "
                            f"document_scope_enabled={document_scope_enabled}",
                            request_id,
                        )

                        if memory_context_text:
                            debug_id(
                                "[ChatOrchestrator] Conversation memory preview "
                                f"conversation_id={active_conversation_id} "
                                f"preview={memory_context_text[:1500]!r}",
                                request_id,
                            )

                        self._touch_chat_session(chat_session)

            except Exception as memory_error:
                warning_id(
                    "Chat memory preparation failed; answer generation will continue without memory: "
                    f"{memory_error}",
                    request_id,
                    exc_info=True,
                )

                active_conversation_id = normalized_conversation_id
                memory_context_text = ""
                memory_context_used = False
                intent_decision = ChatIntentDecision.fallback_new_topic(
                    normalized_question
                )

            memory_time = time.perf_counter() - memory_start

            ai_start = time.perf_counter()

            question_for_ai = self._select_question_for_ai(
                normalized_question=normalized_question,
                intent_decision=intent_decision,
            )

            debug_id(
                "[ChatOrchestrator] Question selected for AI "
                f"original={normalized_question!r} "
                f"question_for_ai={question_for_ai!r} "
                f"intent={intent_decision.intent.value if intent_decision else None} "
                f"memory_context_used={memory_context_used} "
                f"document_scope_enabled={document_scope_enabled}",
                request_id,
            )

            with self.transaction() as session:
                ai_result = None

                # ------------------------------------------------------------
                # Personal memory update pathway
                # ------------------------------------------------------------
                # This is terminal and must run before recall cascade and before
                # normal AI/RAG execution.
                #
                # Example:
                #   User: "Hi my name is John"
                #
                # Correct behavior:
                #   - Store the user message in ChatSession
                #   - Return a short acknowledgement
                #   - Do NOT call AIStewardManagerService.execute()
                #   - Do NOT call RAGPipeline
                #   - Do NOT search document_embedding
                # ------------------------------------------------------------

                if (
                    intent_decision
                    and intent_decision.intent == ChatIntent.PERSONAL_MEMORY_UPDATE
                ):
                    with tracer.span(
                        "chat.orchestrator.personal_memory_update_no_rag",
                        meta={
                            "request_id": request_id,
                            "conversation_id": active_conversation_id,
                            "user_id": normalized_user_id,
                            "intent": intent_decision.intent.value,
                            "intent_confidence": intent_decision.confidence,
                            "document_scope_enabled": document_scope_enabled,
                        },
                    ):
                        answer = self._build_personal_memory_update_acknowledgement(
                            question_for_ai
                        )

                        ai_result = self._personal_memory_update_result(
                            answer=answer,
                            conversation_id=active_conversation_id,
                        )

                        memory_context_text = ""
                        memory_context_used = False

                        debug_id(
                            "[ChatOrchestrator] Personal memory update handled without RAG",
                            request_id,
                        )

                # ------------------------------------------------------------
                # Memory / recall pathway
                # ------------------------------------------------------------
                # This must run before normal AI/RAG execution.
                #
                # Correct behavior:
                #   If current-session memory is required and memory exists,
                #   let the recall cascade answer from ChatSession memory first.
                #
                # Safety:
                #   Do not run this branch during Ask This Document mode.
                #   Selected-document scope should remain authoritative.
                # ------------------------------------------------------------

                should_try_recall_cascade = False

                if ai_result is None and intent_decision and not document_scope_enabled:
                    if intent_decision.intent == ChatIntent.RECALL_PRIOR_CONVERSATION:
                        should_try_recall_cascade = True

                    elif (
                        intent_decision.intent == ChatIntent.FOLLOW_UP_CURRENT_SESSION
                        and bool(intent_decision.needs_current_session_memory)
                        and bool((memory_context_text or "").strip())
                    ):
                        should_try_recall_cascade = True

                if should_try_recall_cascade:
                    with tracer.span(
                        "chat.orchestrator.recall_cascade",
                        meta={
                            "request_id": request_id,
                            "conversation_id": active_conversation_id,
                            "memory_context_used": memory_context_used,
                            "memory_context_chars": len(memory_context_text or ""),
                            "user_id": normalized_user_id,
                            "intent": (
                                intent_decision.intent.value
                                if intent_decision
                                else None
                            ),
                            "intent_confidence": (
                                intent_decision.confidence
                                if intent_decision
                                else None
                            ),
                            "document_scope_enabled": document_scope_enabled,
                        },
                    ):
                        try:
                            debug_id(
                                "[ChatOrchestrator] Trying recall cascade before RAG "
                                f"intent={intent_decision.intent.value if intent_decision else None} "
                                f"memory_context_used={memory_context_used} "
                                f"memory_context_chars={len(memory_context_text or '')} "
                                f"conversation_id={active_conversation_id}",
                                request_id,
                            )

                            ai_result = self._handle_recall_cascade(
                                session=session,
                                question_for_ai=question_for_ai,
                                memory_context_text=memory_context_text,
                                conversation_id=active_conversation_id,
                                user_id=normalized_user_id,
                                request_id=request_id,
                            )

                        except AttributeError:
                            warning_id(
                                "[ChatOrchestrator] _handle_recall_cascade is not available yet. "
                                "Falling back to normal AI/RAG path.",
                                request_id,
                                exc_info=True,
                            )
                            ai_result = None

                        except Exception as recall_error:
                            warning_id(
                                "[ChatOrchestrator] Recall cascade failed. "
                                "Falling back to normal AI/RAG path. "
                                f"Error: {recall_error}",
                                request_id,
                                exc_info=True,
                            )
                            ai_result = None

                if ai_result is None:
                    with tracer.span(
                        "chat.orchestrator.ai_answer_execute",
                        meta={
                            "user_id": normalized_user_id,
                            "client_type": normalized_client_type,
                            "forced_chunk_id": forced_chunk_id,
                            "include_payload": False,
                            "audit_pathway": self.AUDIT_PATHWAY_NAME,
                            "conversation_id": active_conversation_id,
                            "memory_context_used": memory_context_used,
                            "intent": (
                                intent_decision.intent.value
                                if intent_decision
                                else None
                            ),
                            "intent_confidence": (
                                intent_decision.confidence
                                if intent_decision
                                else None
                            ),
                            "original_question": normalized_question,
                            "question_for_ai": question_for_ai,
                            "document_scope_enabled": document_scope_enabled,
                            "complete_document_id": (
                                normalized_document_scope.get("complete_document_id")
                                if normalized_document_scope
                                else None
                            ),
                            "document_name": (
                                normalized_document_scope.get("document_name")
                                if normalized_document_scope
                                else None
                            ),
                        },
                    ):
                        ai_result = self._execute_ai_service(
                            session=session,
                            user_id=normalized_user_id,
                            question=question_for_ai,
                            client_type=normalized_client_type,
                            request_id=request_id,
                            forced_chunk_id=forced_chunk_id,
                            memory_context_text=memory_context_text,
                            conversation_id=active_conversation_id,
                            document_scope=normalized_document_scope,
                        )

            ai_time = time.perf_counter() - ai_start

            if not isinstance(ai_result, dict):
                raise ValueError("AI service returned invalid response format")

            self._apply_debug_metadata(
                ai_result=ai_result,
                forced_chunk_id=forced_chunk_id,
            )

            ai_result.setdefault("payload_status", "pending")
            ai_result["conversation_id"] = active_conversation_id
            ai_result["memory_enabled"] = bool(active_conversation_id)
            ai_result["memory_context_used"] = memory_context_used
            ai_result["original_question"] = normalized_question
            ai_result["question_for_ai"] = question_for_ai
            ai_result["document_scope"] = (
                ai_result.get("document_scope")
                or ai_result.get("documentScope")
                or normalized_document_scope
            )
            ai_result["document_scope_enabled"] = bool(
                self._normalize_document_scope(ai_result.get("document_scope"))
            )

            if intent_decision:
                ai_result["intent"] = self._intent_decision_to_dict(intent_decision)

            persist_start = time.perf_counter()

            try:
                with self.transaction() as session:
                    with tracer.span(
                        "chat.orchestrator.persist_qanda_seed",
                        meta={
                            "request_id": request_id,
                            "conversation_id": active_conversation_id,
                            "document_scope_enabled": document_scope_enabled,
                            "complete_document_id": (
                                normalized_document_scope.get("complete_document_id")
                                if normalized_document_scope
                                else None
                            ),
                            "intent": (
                                intent_decision.intent.value
                                if intent_decision
                                else None
                            ),
                            "original_question": normalized_question,
                            "question_for_ai": question_for_ai,
                        },
                    ):
                        qanda_record = self._create_qanda_interaction_compat(
                            session=session,
                            user_id=normalized_user_id,
                            question=normalized_question,
                            answer=ai_result.get("answer", ""),
                            request_id=request_id,
                            processing_time_ms=int(ai_time * 1000),
                            raw_response=ai_result,
                            intent_decision=intent_decision,
                        )

                        session.flush()
                        qanda_id = self._extract_record_id(qanda_record)

                        if not qanda_id:
                            qanda_id = self._resolve_qanda_id_by_request_id(
                                session=session,
                                request_id=request_id,
                                user_id=normalized_user_id,
                            )

                        debug_id(
                            f"[ChatOrchestrator] QandA seed persisted "
                            f"request_id={request_id} qanda_id={qanda_id} "
                            f"record_type={type(qanda_record).__name__}",
                            request_id,
                        )

                    audit_start = time.perf_counter()

                    with tracer.span(
                        "chat.orchestrator.audit_answer_search_pathway",
                        meta={
                            "request_id": request_id,
                            "pathway_name": self.AUDIT_PATHWAY_NAME,
                            "qanda_id": str(qanda_id) if qanda_id else None,
                            "conversation_id": active_conversation_id,
                            "document_scope_enabled": document_scope_enabled,
                            "complete_document_id": (
                                normalized_document_scope.get("complete_document_id")
                                if normalized_document_scope
                                else None
                            ),
                            "intent": (
                                intent_decision.intent.value
                                if intent_decision
                                else None
                            ),
                            "original_question": normalized_question,
                            "question_for_ai": question_for_ai,
                        },
                    ):
                        audit_summary = SearchAuditService.record_search_result(
                            session=session,
                            request_id=request_id or "unknown",
                            user_id=normalized_user_id,
                            session_id=active_conversation_id,
                            qanda_id=self._coerce_uuid_or_none(qanda_id),
                            question=normalized_question,
                            answer=ai_result.get("answer", ""),
                            response=ai_result,
                            pathway_name=self.AUDIT_PATHWAY_NAME,
                            pathway_version=self.AUDIT_PATHWAY_VERSION,
                            duration_ms=int(
                                (time.perf_counter() - request_start) * 1000
                            ),
                            model_name=ai_result.get("model_name"),
                        )

                    audit_time = time.perf_counter() - audit_start

            except Exception as persist_error:
                warning_id(
                    "Q&A seed persistence or answer audit failed but answer response will continue: "
                    f"{persist_error}",
                    request_id,
                    exc_info=True,
                )

            persist_time = time.perf_counter() - persist_start

            assistant_memory_start = time.perf_counter()

            if active_conversation_id:
                try:
                    with self.transaction() as session:
                        with tracer.span(
                            "chat.orchestrator.memory_store_assistant_answer",
                            meta={
                                "request_id": request_id,
                                "conversation_id": active_conversation_id,
                                "qanda_id": str(qanda_id) if qanda_id else None,
                                "document_scope_enabled": document_scope_enabled,
                                "complete_document_id": (
                                    normalized_document_scope.get("complete_document_id")
                                    if normalized_document_scope
                                    else None
                                ),
                                "intent": (
                                    intent_decision.intent.value
                                    if intent_decision
                                    else None
                                ),
                            },
                        ):
                            self._store_assistant_memory(
                                session=session,
                                conversation_id=active_conversation_id,
                                answer=ai_result.get("answer", ""),
                                question=normalized_question,
                                request_id=request_id,
                                qanda_id=qanda_id,
                                client_type=normalized_client_type,
                                document_scope=normalized_document_scope,
                                intent_decision=intent_decision,
                            )

                except Exception as memory_store_error:
                    warning_id(
                        "Assistant memory update failed but answer response will continue: "
                        f"{memory_store_error}",
                        request_id,
                        exc_info=True,
                    )

            memory_time += time.perf_counter() - assistant_memory_start

            embedding_start = time.perf_counter()

            if qanda_id:
                try:
                    with self.transaction() as session:
                        with tracer.span(
                            "chat.orchestrator.qanda_embedding_update",
                            meta={
                                "request_id": request_id,
                                "qanda_id": str(qanda_id),
                                "conversation_id": active_conversation_id,
                                "document_scope_enabled": document_scope_enabled,
                                "complete_document_id": (
                                    normalized_document_scope.get("complete_document_id")
                                    if normalized_document_scope
                                    else None
                                ),
                                "intent": (
                                    intent_decision.intent.value
                                    if intent_decision
                                    else None
                                ),
                            },
                        ):
                            debug_id(
                                f"[ChatOrchestrator] Calling QandAEmbeddingService "
                                f"request_id={request_id} qanda_id={qanda_id}",
                                request_id,
                            )

                            embeddings_updated = (
                                self.qanda_embedding_service.embed_existing_qanda(
                                    session=session,
                                    qa_id=qanda_id,
                                    question=normalized_question,
                                    answer=ai_result.get("answer", ""),
                                    embed_question=True,
                                    embed_answer=True,
                                    request_id=request_id,
                                    skip_existing=True,
                                    commit=False,
                                )
                            )

                    if embeddings_updated:
                        info_id(
                            f"[ChatOrchestrator] QandA embeddings committed "
                            f"qanda_id={qanda_id} request_id={request_id}",
                            request_id,
                        )
                    else:
                        warning_id(
                            f"[ChatOrchestrator] QandA embedding service completed "
                            f"but reported no update qanda_id={qanda_id} "
                            f"request_id={request_id}",
                            request_id,
                        )

                except Exception as embedding_error:
                    embeddings_updated = False
                    warning_id(
                        "Q&A embedding update failed but answer response will continue: "
                        f"{embedding_error}",
                        request_id,
                        exc_info=True,
                    )

            else:
                warning_id(
                    "[ChatOrchestrator] Skipping QandA embedding update because "
                    "qanda_id is missing after flush and request_id fallback.",
                    request_id,
                )

            embedding_time = time.perf_counter() - embedding_start

            total_time = time.perf_counter() - request_start

            response = self._answer_response(
                ai_result=ai_result,
                request_id=request_id,
                client_type=normalized_client_type,
                total_time=total_time,
                memory_time=memory_time,
                ai_time=ai_time,
                persist_time=persist_time,
                embedding_time=embedding_time,
                audit_time=audit_time,
                audit_summary=audit_summary,
                qanda_id=qanda_id,
                conversation_id=active_conversation_id,
                memory_context_used=memory_context_used,
                embeddings_updated=embeddings_updated,
                document_scope=normalized_document_scope,
                intent_decision=intent_decision,
            )

            self.audit_log_manager.log_run_success(
                request_id=request_id or "unknown",
                pathway_name=self.AUDIT_PATHWAY_NAME,
                duration_ms=int(total_time * 1000),
                counts=(audit_summary or {}).get("counts"),
            )

            info_id(
                f"Chat answer processed in {total_time:.3f}s "
                f"(Memory: {memory_time:.3f}s | AI: {ai_time:.3f}s | "
                f"Persist seed: {persist_time:.3f}s | "
                f"Embeddings: {embedding_time:.3f}s | Audit: {audit_time:.3f}s | "
                f"conversation_id={active_conversation_id} | "
                f"memory_context_used={memory_context_used} | "
                f"intent={intent_decision.intent.value if intent_decision else None} | "
                f"intent_confidence={intent_decision.confidence if intent_decision else None} | "
                f"question_for_ai={question_for_ai!r} | "
                f"document_scope_enabled={document_scope_enabled} | "
                f"complete_document_id="
                f"{normalized_document_scope.get('complete_document_id') if normalized_document_scope else None} | "
                f"qanda_id={qanda_id} | embeddings_updated={embeddings_updated})",
                request_id,
            )

            return response

        except Exception as e:
            total_time = time.perf_counter() - request_start

            self.audit_log_manager.log_run_failure(
                request_id=request_id or "unknown",
                pathway_name=self.AUDIT_PATHWAY_NAME,
                error=e,
                duration_ms=int(total_time * 1000),
            )

            error_id(
                f"ChatOrchestrator answer failure after {total_time:.3f}s: {e}",
                request_id,
                exc_info=True,
            )

            return self._error_response(
                request_id=request_id,
                conversation_id=active_conversation_id or normalized_conversation_id,
                total_time=total_time,
                memory_time=memory_time,
                ai_time=ai_time,
                persist_time=persist_time,
                embedding_time=embedding_time,
                audit_time=audit_time,
                document_scope=normalized_document_scope,
                intent_decision=intent_decision,
            )

    def _execute_ai_service(
        self,
        *,
        session,
        user_id: str,
        question: str,
        client_type: str,
        request_id: Optional[str],
        forced_chunk_id: Optional[int],
        memory_context_text: str,
        conversation_id: Optional[str],
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        normalized_document_scope = self._normalize_document_scope(document_scope)
        memory_context_text = (memory_context_text or "").strip()
        normalized_conversation_id = (
            str(conversation_id).strip()
            if conversation_id is not None and str(conversation_id).strip()
            else None
        )

        execute_kwargs: Dict[str, Any] = {
            "session": session,
            "user_id": user_id,
            "question": question,
            "client_type": client_type,
            "request_id": request_id,
            "forced_chunk_id": forced_chunk_id,
            "include_payload": False,
        }

        try:
            signature = inspect.signature(self.ai_service.execute)
            parameters = signature.parameters

            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )

            supports_document_scope = "document_scope" in parameters or accepts_kwargs
            supports_memory_context = "memory_context" in parameters or accepts_kwargs
            supports_conversation_context = "conversation_context" in parameters or accepts_kwargs
            supports_conversation_id = "conversation_id" in parameters or accepts_kwargs

            debug_id(
                "[ChatOrchestrator] AI service signature inspected "
                f"request_id={request_id} "
                f"accepts_kwargs={accepts_kwargs} "
                f"accepts_document_scope={supports_document_scope} "
                f"accepts_memory_context={supports_memory_context} "
                f"accepts_conversation_context={supports_conversation_context} "
                f"accepts_conversation_id={supports_conversation_id}",
                request_id,
            )

            if normalized_conversation_id and supports_conversation_id:
                execute_kwargs["conversation_id"] = normalized_conversation_id

                debug_id(
                    "[ChatOrchestrator] Passing conversation_id to AI service "
                    f"request_id={request_id} "
                    f"conversation_id={normalized_conversation_id}",
                    request_id,
                )

            if normalized_document_scope:
                if supports_document_scope:
                    execute_kwargs["document_scope"] = normalized_document_scope

                    debug_id(
                        "[ChatOrchestrator] Passing document_scope to AI service "
                        f"complete_document_id={normalized_document_scope.get('complete_document_id')}",
                        request_id,
                    )

                else:
                    warning_id(
                        "[ChatOrchestrator] document_scope active but "
                        "AIStewardManagerService.execute does not accept document_scope yet. "
                        "Continuing without scoped retrieval. "
                        f"complete_document_id={normalized_document_scope.get('complete_document_id')}",
                        request_id,
                    )

            if memory_context_text:
                if supports_memory_context:
                    execute_kwargs["memory_context"] = memory_context_text

                    debug_id(
                        "[ChatOrchestrator] Passing conversational memory as memory_context "
                        f"request_id={request_id} "
                        f"conversation_id={normalized_conversation_id} "
                        f"memory_context_chars={len(memory_context_text)}",
                        request_id,
                    )

                elif supports_conversation_context:
                    execute_kwargs["conversation_context"] = memory_context_text

                    debug_id(
                        "[ChatOrchestrator] Passing conversational memory as conversation_context "
                        f"request_id={request_id} "
                        f"conversation_id={normalized_conversation_id} "
                        f"memory_context_chars={len(memory_context_text)}",
                        request_id,
                    )

                else:
                    execute_kwargs["question"] = self._build_memory_augmented_question(
                        question=question,
                        memory_context_text=memory_context_text,
                    )

                    warning_id(
                        "[ChatOrchestrator] AI service does not accept memory kwargs. "
                        "Using memory-augmented question compatibility fallback. "
                        f"request_id={request_id} "
                        f"conversation_id={normalized_conversation_id} "
                        f"memory_context_chars={len(memory_context_text)} "
                        f"augmented_question_chars={len(str(execute_kwargs.get('question') or ''))}",
                        request_id,
                    )

        except (TypeError, ValueError):
            if normalized_document_scope:
                warning_id(
                    "[ChatOrchestrator] Could not inspect AI service signature. "
                    "Continuing without document_scope pass-through.",
                    request_id,
                    exc_info=True,
                )

            if memory_context_text:
                execute_kwargs["question"] = self._build_memory_augmented_question(
                    question=question,
                    memory_context_text=memory_context_text,
                )

                warning_id(
                    "[ChatOrchestrator] Could not inspect AI service signature. "
                    "Using memory-augmented question compatibility fallback. "
                    f"request_id={request_id} "
                    f"conversation_id={normalized_conversation_id} "
                    f"memory_context_chars={len(memory_context_text)} "
                    f"augmented_question_chars={len(str(execute_kwargs.get('question') or ''))}",
                    request_id,
                    exc_info=True,
                )

        try:
            debug_id(
                "[ChatOrchestrator] AI service execute kwargs "
                f"request_id={request_id} "
                f"has_memory_context={'memory_context' in execute_kwargs} "
                f"has_conversation_context={'conversation_context' in execute_kwargs} "
                f"has_conversation_id={'conversation_id' in execute_kwargs} "
                f"conversation_id={execute_kwargs.get('conversation_id')} "
                f"has_document_scope={'document_scope' in execute_kwargs} "
                f"question_chars={len(str(execute_kwargs.get('question') or ''))} "
                f"memory_context_chars={len(str(execute_kwargs.get('memory_context') or execute_kwargs.get('conversation_context') or ''))} "
                f"document_scope_enabled={bool(normalized_document_scope)} "
                f"execute_keys={sorted(execute_kwargs.keys())}",
                request_id,
            )

            return self.ai_service.execute(**execute_kwargs)

        except TypeError as type_error:
            type_error_text = str(type_error)
            unexpected_kwarg = "unexpected keyword argument" in type_error_text

            if unexpected_kwarg:
                removed_keys: List[str] = []

                for key in (
                    "document_scope",
                    "memory_context",
                    "conversation_context",
                    "conversation_id",
                ):
                    if key in execute_kwargs:
                        execute_kwargs.pop(key, None)
                        removed_keys.append(key)

                if removed_keys:
                    warning_id(
                        "[ChatOrchestrator] AI service rejected kwargs "
                        f"{removed_keys}; retrying with compatibility fallback. "
                        f"error={type_error}",
                        request_id,
                        exc_info=True,
                    )

                    if memory_context_text:
                        execute_kwargs["question"] = self._build_memory_augmented_question(
                            question=question,
                            memory_context_text=memory_context_text,
                        )

                    debug_id(
                        "[ChatOrchestrator] AI service retry kwargs "
                        f"request_id={request_id} "
                        f"execute_keys={sorted(execute_kwargs.keys())}",
                        request_id,
                    )

                    return self.ai_service.execute(**execute_kwargs)

            raise

    def _answer_from_session_memory(
        self,
        *,
        question: str,
        memory_context_text: str,
        request_id: Optional[str],
    ) -> str:

        memory_context_text = (memory_context_text or "").strip()
        question = (question or "").strip()

        if not memory_context_text:
            return ""

        if not hasattr(self.ai_service, "answer_from_context"):
            warning_id(
                "[ChatOrchestrator] AIStewardManagerService.answer_from_context is not available yet.",
                request_id,
            )
            return ""

        recall_question = (
            "Answer the user's question using ONLY the conversation memory provided. "
            "The memory is the authoritative record of this conversation. "
            "If the memory only contains the user's recall question itself, or if it does "
            "not contain enough prior conversation to answer, reply with exactly: "
            f"{self.RECALL_SESSION_SENTINEL}\n\n"
            f"User question:\n{question}"
        )

        try:
            answer = self.ai_service.answer_from_context(
                question=recall_question,
                context=memory_context_text,
                request_id=request_id,
            )
        except Exception as exc:
            warning_id(
                f"[ChatOrchestrator] Session memory recall generation failed: {exc}",
                request_id,
                exc_info=True,
            )
            return ""

        answer = str(answer or "").strip()

        if not answer:
            return ""

        if self.RECALL_SESSION_SENTINEL in answer:
            return ""

        return answer

    def _recall_result(
        self,
        *,
        answer: str,
        strategy: str,
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:

        normalized_strategy = (strategy or "recall_session_memory").strip()

        return {
            "status": "success",
            "strategy": normalized_strategy,
            "method": normalized_strategy,
            "answer": answer or "",

            # No RAG evidence for memory-only recall answers.
            "chunks": [],
            "used_chunks": [],
            "documents": [],
            "images": [],
            "drawings": [],
            "parts": [],
            "relationship_map": {},

            # Important:
            # Memory recall does not have a payload to project.
            # This prevents the UI from treating this as a pending
            # document/image/part/drawing payload request.
            "payload_status": "not_applicable",
            "payload_endpoint": None,

            # No vector document retrieval was used for this answer.
            "retriever_top_k": None,
            "query_embedding": [],

            # Conversation memory metadata.
            "conversation_id": conversation_id,
            "memory_enabled": bool(conversation_id),
            "memory_context_used": True,
            "memory_context_mode": "recall_answer",

            # Document scope is intentionally off for recall results.
            "document_scope": None,
            "document_scope_enabled": False,
            "document_scope_mode": "none",

            # Helpful for logs/debug/audit.
            "model_name": "conversation_recall",

            # Keep response shape consistent with normal answers.
            "blocks": {
                "documents-container": [],
                "parts-container": [],
                "images-container": [],
                "drawings-container": [],
            },
        }

    def _handle_recall_cascade(
            self,
            *,
            session,
            question_for_ai: str,
            memory_context_text: str,
            conversation_id: Optional[str],
            user_id: str,
            request_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:

        # --------------------------------------------------
        # 1. Current ChatSession memory recall
        # --------------------------------------------------
        answer = self._answer_from_session_memory(
            question=question_for_ai,
            memory_context_text=memory_context_text,
            request_id=request_id,
        )

        if answer:
            debug_id(
                "[ChatOrchestrator] Recall answered from current ChatSession memory.",
                request_id,
            )

            return self._recall_result(
                answer=answer,
                strategy="recall_session_memory",
                conversation_id=conversation_id,
            )

        debug_id(
            "[ChatOrchestrator] Current ChatSession memory had no usable recall answer. "
            "Trying QandA semantic recall.",
            request_id,
        )

        # --------------------------------------------------
        # 2. Cross-session QandA semantic recall
        # --------------------------------------------------
        answer = self._semantic_recall_qanda(
            session=session,
            question=question_for_ai,
            user_id=user_id,
            request_id=request_id,
        )

        if answer:
            debug_id(
                "[ChatOrchestrator] Recall answered from QandA semantic recall.",
                request_id,
            )

            return self._recall_result(
                answer=answer,
                strategy="recall_qanda_semantic",
                conversation_id=conversation_id,
            )

        # --------------------------------------------------
        # 3. No memory recall match; caller falls back to RAG
        # --------------------------------------------------
        debug_id(
            "[ChatOrchestrator] Recall cascade exhausted. "
            "No ChatSession memory answer and no QandA semantic recall answer. "
            "Falling back to normal RAG.",
            request_id,
        )

        return None

    def _create_qanda_interaction_compat(
        self,
        *,
        session,
        user_id: str,
        question: str,
        answer: str,
        request_id: Optional[str],
        processing_time_ms: int,
        raw_response: Dict[str, Any],
        intent_decision: Optional[ChatIntentDecision],
    ) -> Any:

        base_kwargs: Dict[str, Any] = {
            "session": session,
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "request_id": request_id,
            "processing_time_ms": processing_time_ms,
            "raw_response": raw_response,
        }

        intent_kwargs: Dict[str, Any] = {
            "intent_type": intent_decision.intent.value if intent_decision else None,
            "intent_confidence": (
                float(intent_decision.confidence or 0.0)
                if intent_decision
                else None
            ),
            "intent_reason": intent_decision.reason if intent_decision else None,
            "intent_rewritten_question": (
                intent_decision.rewritten_question if intent_decision else None
            ),
            "intent_needs_current_session_memory": (
                bool(intent_decision.needs_current_session_memory)
                if intent_decision
                else False
            ),
            "intent_needs_semantic_chat_recall": (
                bool(intent_decision.needs_semantic_chat_recall)
                if intent_decision
                else False
            ),
            "intent_needs_document_scope": (
                bool(intent_decision.needs_document_scope)
                if intent_decision
                else False
            ),
        }

        try:
            signature = inspect.signature(self.qanda_service.create_interaction)
            parameters = signature.parameters

            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )

            if accepts_kwargs:
                return self.qanda_service.create_interaction(
                    **base_kwargs,
                    **intent_kwargs,
                )

            filtered_intent_kwargs = {
                key: value
                for key, value in intent_kwargs.items()
                if key in parameters
            }

            return self.qanda_service.create_interaction(
                **base_kwargs,
                **filtered_intent_kwargs,
            )

        except (TypeError, ValueError):
            warning_id(
                "[ChatOrchestrator] Could not inspect QandAService.create_interaction; "
                "retrying without intent fields.",
                request_id,
                exc_info=True,
            )
            return self.qanda_service.create_interaction(**base_kwargs)

    @staticmethod
    def _should_use_memory_for_intent(
        *,
        intent_decision: Optional[ChatIntentDecision],
        document_scope_enabled: bool,
    ) -> bool:
        if intent_decision is None:
            return False

        if intent_decision.intent == ChatIntent.NEW_TOPIC:
            return False

        if intent_decision.intent == ChatIntent.DOCUMENT_SCOPED_FOLLOW_UP:
            return bool(intent_decision.needs_current_session_memory)

        if document_scope_enabled and not intent_decision.needs_current_session_memory:
            return False

        return bool(intent_decision.needs_current_session_memory)

    @staticmethod
    def _select_question_for_ai(
        *,
        normalized_question: str,
        intent_decision: Optional[ChatIntentDecision],
    ) -> str:
        if intent_decision is None:
            return normalized_question

        rewritten = (intent_decision.rewritten_question or "").strip()

        if not rewritten:
            return normalized_question

        if intent_decision.intent in {
            ChatIntent.FOLLOW_UP_CURRENT_SESSION,
            ChatIntent.CLARIFICATION,
            ChatIntent.DOCUMENT_SCOPED_FOLLOW_UP,
            ChatIntent.RECALL_PRIOR_CONVERSATION,
            ChatIntent.PERSONAL_MEMORY_UPDATE,
        }:
            return rewritten


        return normalized_question

    def _get_or_create_chat_session(
        self,
        *,
        session,
        conversation_id: Optional[str],
        user_id: str,
        request_id: Optional[str],
    ) -> Tuple[ChatSession, bool]:

        parsed_conversation_id = self._coerce_uuid_or_none(conversation_id)

        if parsed_conversation_id is not None:
            existing_session = session.get(ChatSession, parsed_conversation_id)

            if existing_session is not None:
                existing_user_id = str(getattr(existing_session, "user_id", "") or "")
                requested_user_id = str(user_id or "")

                if existing_user_id == requested_user_id:
                    return existing_session, False

                warning_id(
                    "[ChatOrchestrator] Conversation ID belongs to a different user. "
                    f"incoming_conversation_id={conversation_id} "
                    f"existing_user_id={existing_user_id} requested_user_id={requested_user_id}. "
                    "Creating a new ChatSession.",
                    request_id,
                )

            else:
                warning_id(
                    "[ChatOrchestrator] Conversation ID was supplied but no ChatSession was found. "
                    f"incoming_conversation_id={conversation_id}. Creating a new ChatSession.",
                    request_id,
                )

        elif conversation_id:
            warning_id(
                f"[ChatOrchestrator] Invalid conversation_id supplied: {conversation_id!r}. "
                "Creating a new ChatSession.",
                request_id,
            )

        now = self._utc_iso()

        chat_session = ChatSession(
            user_id=str(user_id or "anonymous"),
            start_time=now,
            last_interaction=now,
            session_data=[],
            conversation_summary=[],
        )

        session.add(chat_session)
        session.flush()

        return chat_session, True

    def _store_assistant_memory(
        self,
        *,
        session,
        conversation_id: str,
        answer: str,
        question: str,
        request_id: Optional[str],
        qanda_id: Optional[Any],
        client_type: str,
        document_scope: Optional[Dict[str, Any]] = None,
        intent_decision: Optional[ChatIntentDecision] = None,
    ) -> None:

        parsed_conversation_id = self._coerce_uuid_or_none(conversation_id)
        normalized_document_scope = self._normalize_document_scope(document_scope)

        if parsed_conversation_id is None:
            warning_id(
                "[ChatOrchestrator] Cannot store assistant memory because conversation_id "
                f"is invalid: {conversation_id!r}",
                request_id,
            )
            return

        chat_session = session.get(ChatSession, parsed_conversation_id)

        if chat_session is None:
            warning_id(
                "[ChatOrchestrator] Cannot store assistant memory because ChatSession "
                f"was not found. conversation_id={conversation_id}",
                request_id,
            )
            return

        self._append_chat_message(
            chat_session=chat_session,
            role="assistant",
            content=answer or "",
            request_id=request_id,
            metadata={
                "client_type": client_type,
                "qanda_id": str(qanda_id) if qanda_id else None,
                "document_scope": normalized_document_scope,
                "document_scope_enabled": bool(normalized_document_scope),
                "intent": (
                    self._intent_decision_to_dict(intent_decision)
                    if intent_decision
                    else None
                ),
            },
        )

        self._append_conversation_summary_item(
            chat_session=chat_session,
            question=question,
            answer=answer or "",
            request_id=request_id,
            qanda_id=qanda_id,
            document_scope=normalized_document_scope,
            intent_decision=intent_decision,
        )

        self._update_chat_session_summary_embedding(
            chat_session=chat_session,
            request_id=request_id,
        )

        self._touch_chat_session(chat_session)

    def _append_chat_message(
        self,
        *,
        chat_session: ChatSession,
        role: str,
        content: str,
        request_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        messages = self._safe_list(getattr(chat_session, "session_data", None))

        messages.append(
            {
                "role": role,
                "content": content or "",
                "request_id": request_id,
                "created_at": self._utc_iso(),
                "metadata": metadata or {},
            }
        )

        if len(messages) > self.CHAT_SESSION_MAX_MESSAGES:
            messages = messages[-self.CHAT_SESSION_MAX_MESSAGES:]

        chat_session.session_data = messages

    def _append_conversation_summary_item(
        self,
        *,
        chat_session: ChatSession,
        question: str,
        answer: str,
        request_id: Optional[str],
        qanda_id: Optional[Any],
        document_scope: Optional[Dict[str, Any]] = None,
        intent_decision: Optional[ChatIntentDecision] = None,
    ) -> None:

        normalized_document_scope = self._normalize_document_scope(document_scope)
        summaries = self._safe_list(getattr(chat_session, "conversation_summary", None))

        summaries.append(
            {
                "request_id": request_id,
                "qanda_id": str(qanda_id) if qanda_id else None,
                "created_at": self._utc_iso(),
                "question": self._preview_text(question, self.MEMORY_PREVIEW_CHARS),
                "answer_preview": self._preview_text(
                    answer,
                    self.SUMMARY_ANSWER_PREVIEW_CHARS,
                ),
                "document_scope": normalized_document_scope,
                "document_scope_enabled": bool(normalized_document_scope),
                "intent": (
                    self._intent_decision_to_dict(intent_decision)
                    if intent_decision
                    else None
                ),
            }
        )

        if len(summaries) > self.SUMMARY_MAX_ITEMS:
            summaries = summaries[-self.SUMMARY_MAX_ITEMS:]

        chat_session.conversation_summary = summaries

    def _update_chat_session_summary_embedding(
        self,
        *,
        chat_session: ChatSession,
        request_id: Optional[str],
    ) -> bool:

        summaries = self._safe_list(getattr(chat_session, "conversation_summary", None))

        summary_text = "\n\n".join(
            (
                f"Question: {str(item.get('question') or '').strip()}\n"
                f"Answer: {str(item.get('answer_preview') or '').strip()}"
            )
            for item in summaries
            if isinstance(item, dict)
            and (
                str(item.get("question") or "").strip()
                or str(item.get("answer_preview") or "").strip()
            )
        ).strip()

        if not summary_text:
            chat_session.summary_embedding = None
            warning_id(
                "[ChatOrchestrator] Chat session summary embedding skipped because summary text is empty. "
                f"conversation_id={getattr(chat_session, 'session_id', None)}",
                request_id,
            )
            return False

        try:
            embedding = self.qanda_embedding_service.embed_text(
                text=summary_text,
                request_id=request_id,
            )

            if not embedding:
                warning_id(
                    "[ChatOrchestrator] Chat session summary embedding returned empty. "
                    f"conversation_id={getattr(chat_session, 'session_id', None)} "
                    f"summary_items={len(summaries)} "
                    f"summary_text_chars={len(summary_text)}",
                    request_id,
                )
                return False

            chat_session.summary_embedding = embedding

            info_id(
                "[ChatOrchestrator] Chat session summary embedding updated "
                f"conversation_id={chat_session.session_id} "
                f"summary_items={len(summaries)} "
                f"summary_text_chars={len(summary_text)} "
                f"embedding_dims={len(embedding)}",
                request_id,
            )

            return True

        except Exception as exc:
            error_id(
                "[ChatOrchestrator] Failed to update chat session summary embedding: "
                f"{type(exc).__name__}: {exc}",
                request_id,
                exc_info=True,
            )
            return False

    def _build_memory_context_text(
        self,
        *,
        chat_session: ChatSession,
    ) -> str:

        sections: List[str] = []

        summaries = self._safe_list(getattr(chat_session, "conversation_summary", None))
        recent_summaries = summaries[-self.MEMORY_SUMMARY_LIMIT:]

        if recent_summaries:
            rendered_summary_items: List[str] = []

            for item in recent_summaries:
                if not isinstance(item, dict):
                    continue

                question = self._preview_text(
                    str(item.get("question") or ""),
                    self.MEMORY_PREVIEW_CHARS,
                )
                answer_preview = self._preview_text(
                    str(item.get("answer_preview") or ""),
                    self.MEMORY_PREVIEW_CHARS,
                )

                summary_scope = self._normalize_document_scope(
                    item.get("document_scope")
                )

                scope_line = ""
                if summary_scope:
                    scope_line = (
                        f"\n  Prior document scope: "
                        f"{summary_scope.get('document_name')} "
                        f"(complete_document_id={summary_scope.get('complete_document_id')})"
                    )

                if question or answer_preview:
                    rendered_summary_items.append(
                        f"- Prior question: {question}\n"
                        f"  Prior answer summary: {answer_preview}"
                        f"{scope_line}"
                    )

            if rendered_summary_items:
                sections.append(
                    "Rolling conversation summary:\n"
                    + "\n".join(rendered_summary_items)
                )

        if self.MEMORY_RECENT_MESSAGE_LIMIT > 0:
            messages = self._safe_list(getattr(chat_session, "session_data", None))
            recent_messages = messages[-self.MEMORY_RECENT_MESSAGE_LIMIT:]
        else:
            recent_messages = []

        if recent_messages:
            rendered_messages: List[str] = []

            for item in recent_messages:
                if not isinstance(item, dict):
                    continue

                role = str(item.get("role") or "unknown").strip() or "unknown"
                content = self._preview_text(
                    str(item.get("content") or ""),
                    self.MEMORY_PREVIEW_CHARS,
                )

                metadata = (
                    item.get("metadata")
                    if isinstance(item.get("metadata"), dict)
                    else {}
                )
                message_scope = self._normalize_document_scope(
                    metadata.get("document_scope")
                )

                scope_suffix = ""
                if message_scope:
                    scope_suffix = (
                        f" [document_scope={message_scope.get('document_name')} "
                        f"complete_document_id={message_scope.get('complete_document_id')}]"
                    )

                if content:
                    rendered_messages.append(f"{role}{scope_suffix}: {content}")

            if rendered_messages:
                sections.append(
                    "Recent conversation messages:\n"
                    + "\n".join(rendered_messages)
                )

        return "\n\n".join(sections).strip()

    @staticmethod
    def _build_memory_augmented_question(
        *,
        question: str,
        memory_context_text: str,
    ) -> str:

        return (
            "Use the conversation memory below only to understand context, references, "
            "and follow-up wording. The current user question remains the main request. "
            "Do not let memory override retrieved database/document evidence.\n\n"
            f"Conversation memory:\n{memory_context_text.strip()}\n\n"
            f"Current user question:\n{question.strip()}"
        )

    @staticmethod
    def _touch_chat_session(chat_session: ChatSession) -> None:
        chat_session.last_interaction = ChatOrchestrator._utc_iso()

    @staticmethod
    def _safe_list(value: Any) -> List[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return list(value)

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, str):
            text = value.strip()

            if not text:
                return []

            try:
                parsed = json.loads(text)
            except Exception:
                return []

            if isinstance(parsed, list):
                return parsed

            return []

        return []

    @staticmethod
    def _preview_text(value: str, max_chars: int) -> str:
        text = (value or "").strip()

        if len(text) <= max_chars:
            return text

        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _utc_iso() -> str:
        return datetime.utcnow().isoformat()

    def _resolve_forced_chunk_id(
        self,
        *,
        request_id: Optional[str],
    ) -> Optional[int]:

        if not FORCE_DEBUG_CHUNK:
            return None

        if FORCE_DEBUG_CHUNK_ID is None:
            warning_id(
                "FORCE_DEBUG_CHUNK is enabled but FORCE_DEBUG_CHUNK_ID is None",
                request_id,
            )
            return None

        try:
            forced_chunk_id = int(FORCE_DEBUG_CHUNK_ID)
        except (TypeError, ValueError):
            warning_id(
                f"FORCE_DEBUG_CHUNK_ID is invalid: {FORCE_DEBUG_CHUNK_ID!r}",
                request_id,
            )
            return None

        debug_id(
            f"[ChatOrchestrator] DEBUG MODE ENABLED -> chunk={forced_chunk_id}",
            request_id,
        )

        return forced_chunk_id

    @staticmethod
    def _apply_debug_metadata(
        *,
        ai_result: Dict[str, Any],
        forced_chunk_id: Optional[int],
    ) -> None:

        if forced_chunk_id is not None:
            ai_result.setdefault("debug_mode", True)
            ai_result.setdefault("debug_chunk_id", forced_chunk_id)
        else:
            ai_result.setdefault("debug_mode", False)
            ai_result.setdefault("debug_chunk_id", None)

    def _answer_response(
        self,
        *,
        ai_result: Dict[str, Any],
        request_id: Optional[str],
        client_type: str,
        total_time: float,
        memory_time: float,
        ai_time: float,
        persist_time: float,
        embedding_time: float,
        audit_time: float,
        audit_summary: Optional[Dict[str, Any]] = None,
        qanda_id: Optional[Any] = None,
        conversation_id: Optional[str] = None,
        memory_context_used: bool = False,
        embeddings_updated: bool = False,
        document_scope: Optional[Dict[str, Any]] = None,
        intent_decision: Optional[ChatIntentDecision] = None,
    ) -> Dict[str, Any]:

        method = ai_result.get("method") or ai_result.get("strategy") or self.DEFAULT_METHOD
        strategy = ai_result.get("strategy") or method

        # ------------------------------------------------------------
        # Payload status
        # ------------------------------------------------------------
        # Normal RAG answers usually return payload_status="pending"
        # because /ask/payload should load documents/images/parts/drawings.
        #
        # Memory-only answers should return payload_status="not_applicable"
        # because there is no payload to project.
        #
        # Examples:
        #   recall_session_memory           -> not_applicable
        #   recall_qanda_semantic           -> not_applicable
        #   personal_memory_update_no_rag   -> not_applicable
        #   normal rag                      -> pending
        # ------------------------------------------------------------

        payload_status = str(ai_result.get("payload_status") or "pending").strip() or "pending"

        if payload_status == "not_applicable":
            payload_endpoint = None
        else:
            payload_endpoint = ai_result.get("payload_endpoint") or "/ask/payload"

        # ------------------------------------------------------------
        # Document scope
        # ------------------------------------------------------------
        # If ai_result explicitly says document_scope_enabled=False and
        # document_scope=None, honor that. This matters for no-RAG memory
        # paths that may happen while a document tab is active.
        # ------------------------------------------------------------

        ai_document_scope = self._normalize_document_scope(ai_result.get("document_scope"))

        if (
            ai_result.get("document_scope_enabled") is False
            and ai_document_scope is None
        ):
            normalized_document_scope = None
        else:
            normalized_document_scope = (
                ai_document_scope
                or self._normalize_document_scope(document_scope)
            )

        response: Dict[str, Any] = {
            "status": ai_result.get("status", "success"),
            "answer": ai_result.get("answer", ""),
            "method": method,
            "strategy": strategy,
            "request_id": request_id,
            "conversation_id": conversation_id,
            "document_scope": normalized_document_scope,
            "document_scope_enabled": bool(normalized_document_scope),
            "intent": (
                self._intent_decision_to_dict(intent_decision)
                if intent_decision
                else None
            ),
            "qanda_id": str(qanda_id) if qanda_id else None,
            "response_time": total_time,

            # Updated: do not force memory-only paths into pending payload mode.
            "payload_status": payload_status,
            "payload_endpoint": payload_endpoint,

            "debug_mode": bool(ai_result.get("debug_mode", False)),
            "debug_chunk_id": ai_result.get("debug_chunk_id"),
            "retriever_top_k": ai_result.get("retriever_top_k"),
            "used_chunks_count": len(ai_result.get("used_chunks") or []),
            "memory_enabled": bool(conversation_id),
            "memory_context_used": bool(memory_context_used),
            "embeddings_updated": bool(embeddings_updated),

            # Answer route remains lightweight. Full payload, when applicable,
            # is loaded by /ask/payload.
            "blocks": ai_result.get("blocks") or {
                "documents-container": [],
                "parts-container": [],
                "images-container": [],
                "drawings-container": [],
            },
            "documents": ai_result.get("documents") or [],
            "parts": ai_result.get("parts") or [],
            "images": ai_result.get("images") or [],
            "drawings": ai_result.get("drawings") or [],
        }

        if client_type == "debug":
            response["performance"] = {
                "total_time": total_time,
                "memory_time": memory_time,
                "ai_time": ai_time,
                "persist_time": persist_time,
                "embedding_time": embedding_time,
                "audit_time": audit_time,
                "method": method,
                "strategy": strategy,
                "debug_mode": response["debug_mode"],
                "debug_chunk_id": response["debug_chunk_id"],
                "memory_enabled": bool(conversation_id),
                "memory_context_used": bool(memory_context_used),
                "embeddings_updated": bool(embeddings_updated),
                "document_scope_enabled": bool(normalized_document_scope),
                "complete_document_id": (
                    normalized_document_scope.get("complete_document_id")
                    if normalized_document_scope
                    else None
                ),
                "payload_status": payload_status,
                "payload_endpoint": payload_endpoint,
                "intent": (
                    self._intent_decision_to_dict(intent_decision)
                    if intent_decision
                    else None
                ),
            }

            response["audit"] = {
                "pathway_name": self.AUDIT_PATHWAY_NAME,
                "pathway_version": self.AUDIT_PATHWAY_VERSION,
                "summary": audit_summary or {},
            }

        else:
            response["performance"] = {
                "total_time": total_time,
                "memory_time": memory_time,
                "method": method,
                "strategy": strategy,
                "memory_enabled": bool(conversation_id),
                "memory_context_used": bool(memory_context_used),
                "document_scope_enabled": bool(normalized_document_scope),
                "payload_status": payload_status,
                "payload_endpoint": payload_endpoint,
                "intent": (
                    self._intent_decision_to_dict(intent_decision)
                    if intent_decision
                    else None
                ),
            }

        return response

    def _error_response(
        self,
        *,
        request_id: Optional[str],
        conversation_id: Optional[str],
        total_time: float,
        memory_time: float,
        ai_time: float,
        persist_time: float,
        embedding_time: float,
        audit_time: float,
        document_scope: Optional[Dict[str, Any]] = None,
        intent_decision: Optional[ChatIntentDecision] = None,
    ) -> Dict[str, Any]:

        normalized_document_scope = self._normalize_document_scope(document_scope)

        return {
            "status": "error",
            "answer": "An unexpected error occurred while processing your request.",
            "method": "error",
            "strategy": "error",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "document_scope": normalized_document_scope,
            "document_scope_enabled": bool(normalized_document_scope),
            "intent": (
                self._intent_decision_to_dict(intent_decision)
                if intent_decision
                else None
            ),
            "qanda_id": None,
            "response_time": total_time,
            "payload_status": "unavailable",
            "payload_endpoint": None,
            "debug_mode": False,
            "debug_chunk_id": None,
            "memory_enabled": bool(conversation_id),
            "memory_context_used": False,
            "embeddings_updated": False,
            "blocks": {
                "documents-container": [],
                "parts-container": [],
                "images-container": [],
                "drawings-container": [],
            },
            "documents": [],
            "parts": [],
            "images": [],
            "drawings": [],
            "used_chunks_count": 0,
            "retriever_top_k": None,
            "performance": {
                "total_time": total_time,
                "memory_time": memory_time,
                "ai_time": ai_time,
                "persist_time": persist_time,
                "embedding_time": embedding_time,
                "audit_time": audit_time,
                "method": "error",
                "strategy": "error",
                "document_scope_enabled": bool(normalized_document_scope),
                "intent": (
                    self._intent_decision_to_dict(intent_decision)
                    if intent_decision
                    else None
                ),
            },
        }

    @staticmethod
    def _intent_decision_to_dict(
        intent_decision: Optional[ChatIntentDecision],
    ) -> Optional[Dict[str, Any]]:

        if intent_decision is None:
            return None

        return {
            "intent": intent_decision.intent.value,
            "confidence": float(intent_decision.confidence or 0.0),
            "needs_current_session_memory": bool(
                intent_decision.needs_current_session_memory
            ),
            "needs_semantic_chat_recall": bool(
                intent_decision.needs_semantic_chat_recall
            ),
            "needs_document_scope": bool(intent_decision.needs_document_scope),
            "rewritten_question": intent_decision.rewritten_question or "",
            "reason": intent_decision.reason or "",
        }

    @staticmethod
    def _extract_record_id(record: Any) -> Optional[Any]:

        if record is None:
            return None

        if isinstance(record, dict):
            return (
                record.get("id")
                or record.get("qanda_id")
                or record.get("qa_id")
                or record.get("record_id")
                or record.get("qandaId")
            )

        return (
            getattr(record, "id", None)
            or getattr(record, "qanda_id", None)
            or getattr(record, "qa_id", None)
            or getattr(record, "record_id", None)
        )

    @staticmethod
    def _resolve_qanda_id_by_request_id(
        *,
        session,
        request_id: Optional[str],
        user_id: Optional[str],
    ) -> Optional[Any]:

        if not request_id:
            return None

        try:
            query = session.query(QandA).filter(QandA.request_id == request_id)

            if user_id is not None:
                query = query.filter(QandA.user_id == str(user_id))

            record = query.order_by(QandA.timestamp.desc()).first()

            if record is None:
                return None

            return getattr(record, "id", None)

        except Exception as exc:
            warning_id(
                f"[ChatOrchestrator] Failed to resolve QandA ID by request_id={request_id}: {exc}",
                request_id,
                exc_info=True,
            )
            return None

    @staticmethod
    def _normalize_document_scope(
        document_scope: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:

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

        complete_document_id = ChatOrchestrator._coerce_int_or_none(
            complete_document_id
        )

        if complete_document_id is None:
            return None

        document_id = document_scope.get("document_id") or document_scope.get(
            "documentId"
        )

        document_id = ChatOrchestrator._coerce_int_or_none(document_id)

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

    @staticmethod
    def _coerce_uuid_or_none(value: Any) -> Optional[UUID]:

        if value is None:
            return None

        if isinstance(value, UUID):
            return value

        try:
            return UUID(str(value))
        except (TypeError, ValueError, AttributeError):
            return None

    def _semantic_recall_qanda(
            self,
            *,
            session,
            question: str,
            user_id: str,
            request_id: Optional[str],
    ) -> str:

        question = (question or "").strip()

        if not question:
            return ""

        if not hasattr(self.ai_service, "answer_from_context"):
            warning_id(
                "[ChatOrchestrator] AIStewardManagerService.answer_from_context is not available yet.",
                request_id,
            )
            return ""

        try:
            query_embedding = self.qanda_embedding_service.embed_text(
                text=question,
                request_id=request_id,
            )
        except Exception as exc:
            warning_id(
                f"[ChatOrchestrator] QandA semantic recall embedding failed: {exc}",
                request_id,
                exc_info=True,
            )
            return ""

        if not query_embedding:
            warning_id(
                "[ChatOrchestrator] QandA semantic recall skipped because query embedding is empty.",
                request_id,
            )
            return ""

        try:
            matches = self.qanda_service.find_similar_questions(
                session=session,
                query_embedding=query_embedding,
                user_id=user_id,
                limit=self.QANDA_RECALL_TOP_K,
                similarity_threshold=self.QANDA_RECALL_MIN_SIMILARITY,
            )
        except Exception as exc:
            warning_id(
                f"[ChatOrchestrator] QandA semantic recall search failed: {exc}",
                request_id,
                exc_info=True,
            )
            return ""

        if not matches:
            debug_id(
                "[ChatOrchestrator] QandA semantic recall found no matching prior QandA rows.",
                request_id,
            )
            return ""

        rendered_items: List[str] = []

        for qa_record, similarity in matches:
            prior_question = str(getattr(qa_record, "question", "") or "").strip()
            prior_answer = str(getattr(qa_record, "answer", "") or "").strip()

            if not prior_question and not prior_answer:
                continue

            try:
                similarity_value = float(similarity)
            except (TypeError, ValueError):
                similarity_value = 0.0

            rendered_items.append(
                f"Similarity: {similarity_value:.3f}\n"
                f"Earlier question: {prior_question}\n"
                f"Earlier answer: {prior_answer}"
            )

        if not rendered_items:
            return ""

        recall_context = (
                "Prior recalled QandA records:\n\n"
                + "\n\n---\n\n".join(rendered_items)
        )

        recall_question = (
            "Answer the user's question using ONLY the prior recalled QandA records provided. "
            "These records are from earlier saved chatbot interactions. "
            "If the records do not contain enough relevant information to answer, reply with exactly: "
            f"{self.RECALL_QANDA_SENTINEL}\n\n"
            f"User question:\n{question}"
        )

        try:
            answer = self.ai_service.answer_from_context(
                question=recall_question,
                context=recall_context,
                request_id=request_id,
            )
        except Exception as exc:
            warning_id(
                f"[ChatOrchestrator] QandA semantic recall answer generation failed: {exc}",
                request_id,
                exc_info=True,
            )
            return ""

        answer = str(answer or "").strip()

        if not answer:
            return ""

        if self.RECALL_QANDA_SENTINEL in answer:
            return ""

        debug_id(
            "[ChatOrchestrator] QandA semantic recall produced answer "
            f"matches={len(matches)} "
            f"context_chars={len(recall_context)} "
            f"answer_chars={len(answer)}",
            request_id,
        )

        return answer

    def _personal_memory_update_result(
        self,
        *,
        answer: str,
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            "strategy": "chat.orchestrator.personal_memory_update_no_rag",
            "method": "chat.orchestrator.personal_memory_update_no_rag",
            "answer": answer,
            "chunks": [],
            "used_chunks": [],
            "documents": [],
            "images": [],
            "drawings": [],
            "parts": [],
            "relationship_map": {},
            "payload_status": "not_applicable",
            "retriever_top_k": None,
            "query_embedding": [],
            "conversation_id": conversation_id,
            "memory_enabled": bool(conversation_id),
            "memory_context_used": False,
            "memory_context_mode": "personal_memory_update",
            "document_scope": None,
            "document_scope_enabled": False,
            "document_scope_mode": "none",
            "model_name": "conversation_memory_update",
        }

    def _build_personal_memory_update_acknowledgement(self, question: str) -> str:
        q = (question or "").strip()

        name_patterns = [
            r"^\s*(?:hi|hello|hey)?\s*,?\s*my\s+name\s+is\s+(.+?)\s*[.!]?\s*$",
            r"^\s*call\s+me\s+(.+?)\s*[.!]?\s*$",
            r"^\s*you\s+can\s+call\s+me\s+(.+?)\s*[.!]?\s*$",
            r"^\s*remember\s+(?:that\s+)?my\s+name\s+is\s+(.+?)\s*[.!]?\s*$",
        ]

        for pattern in name_patterns:
            match = re.match(pattern, q, flags=re.IGNORECASE)
            if match:
                name = self._clean_memory_value(match.group(1))
                if name:
                    return f"Nice to meet you, {name}."

        work_match = re.match(
            r"^\s*i\s+work\s+(?:on|in)\s+(.+?)\s*[.!]?\s*$",
            q,
            flags=re.IGNORECASE,
        )
        if work_match:
            area = self._clean_memory_value(work_match.group(1), title_case=False)
            if area:
                return f"Got it — you work on {area}."

        shift_match = re.match(
            r"^\s*i(?:\s+am|'m)\s+on\s+(.+?)\s*[.!]?\s*$",
            q,
            flags=re.IGNORECASE,
        )
        if shift_match:
            shift = self._clean_memory_value(shift_match.group(1), title_case=False)
            if shift:
                return f"Got it — you are on {shift}."

        working_match = re.match(
            r"^\s*i(?:\s+am|'m)\s+working\s+on\s+(.+?)\s*[.!]?\s*$",
            q,
            flags=re.IGNORECASE,
        )
        if working_match:
            item = self._clean_memory_value(working_match.group(1), title_case=False)
            if item:
                return f"Got it — you are working on {item}."

        troubleshooting_match = re.match(
            r"^\s*i(?:\s+am|'m)\s+troubleshooting\s+(.+?)\s*[.!]?\s*$",
            q,
            flags=re.IGNORECASE,
        )
        if troubleshooting_match:
            item = self._clean_memory_value(
                troubleshooting_match.group(1),
                title_case=False,
            )
            if item:
                return f"Got it — you are troubleshooting {item}."

        assigned_match = re.match(
            r"^\s*(?:remember\s+that\s+)?i(?:\s+am|'m)\s+assigned\s+to\s+(.+?)\s*[.!]?\s*$",
            q,
            flags=re.IGNORECASE,
        )
        if assigned_match:
            item = self._clean_memory_value(assigned_match.group(1), title_case=False)
            if item:
                return f"Got it — you are assigned to {item}."

        return "Got it — I’ll keep that in this conversation."

    @staticmethod
    def _clean_memory_value(value: str, *, title_case: bool = True) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" .,!?:;\"'")

        if not text:
            return ""

        if not title_case:
            return text

        parts = []

        for part in text.split(" "):
            if part.isupper() and len(part) <= 5:
                parts.append(part)
            else:
                parts.append(part[:1].upper() + part[1:].lower())

        return " ".join(parts)