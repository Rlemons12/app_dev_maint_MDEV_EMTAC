from __future__ import annotations

import inspect
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from uuid import UUID

from modules.observability.high_end_tracer import tracer
from modules.decorators.trace_decorator import trace_entrypoint
from modules.orchestrators.base_orchestrator import BaseOrchestrator

from modules.services.ai_steward_manager_service import AIStewardManagerService
from modules.services.qanda_service import QandAService
from modules.services.qanda_embedding_service import QandAEmbeddingService

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
    Answer-only chat orchestrator.

    Responsibilities:
        - Own the database transaction/session boundary for answer generation
        - Create/load ChatSession conversational memory
        - Store user/assistant messages in ChatSession.session_data
        - Build lightweight conversation memory context
        - Pass document_scope through the answer-first AI/RAG pathway
        - Coordinate RAG / AI answer generation
        - Persist Q&A seed data for later payload loading
        - Update Q&A question/answer embeddings on the same QandA row
        - Record search-pathway audit summary for answer-first path
        - Return the text answer quickly

    Does NOT:
        - Build documents/images/parts/drawings UI payload
        - Render frontend HTML
        - Read Flask request objects
        - Own route validation
        - Apply final RAG database filtering itself

    Notes:
        - conversation_id maps to ChatSession.session_id.
        - If conversation_id is supplied and valid, the existing ChatSession is reused.
        - If conversation_id is missing, invalid, not found, or owned by a different user,
          a new ChatSession is created.
        - The active conversation_id is always returned in the response when available.
        - document_scope is normalized here and passed deeper when supported.
    """

    DEFAULT_METHOD = "rag"
    AUDIT_PATHWAY_NAME = SearchPathwayName.RAG.value
    AUDIT_PATHWAY_VERSION = "1.0"

    CHAT_SESSION_MAX_MESSAGES = 30
    MEMORY_RECENT_MESSAGE_LIMIT = 0
    MEMORY_SUMMARY_LIMIT = 3
    MEMORY_PREVIEW_CHARS = 350
    SUMMARY_ANSWER_PREVIEW_CHARS = 500
    SUMMARY_MAX_ITEMS = 20

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
    ):
        super().__init__()

        self.ai_service = ai_service or AIStewardManagerService()
        self.qanda_service = qanda_service or QandAService()
        self.qanda_embedding_service = qanda_embedding_service or QandAEmbeddingService()
        self.audit_log_manager = get_search_audit_log_manager()

    @with_request_id
    @trace_entrypoint(
        name="chat_answer_pipeline",
        deep_profile=False,
        capture_args=False,
        capture_return=False,
    )
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

            # --------------------------------------------------
            # 1. Load/create conversation session and store user message
            # --------------------------------------------------
            memory_start = time.perf_counter()

            try:
                with self.transaction() as session:
                    with tracer.span(
                        "chat_memory_prepare",
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

                        if document_scope_enabled:
                            memory_context_text = ""
                            memory_context_used = False

                            debug_id(
                                "[ChatOrchestrator] Document scope active; "
                                "conversation memory disabled for answer generation "
                                f"conversation_id={active_conversation_id} "
                                f"complete_document_id={normalized_document_scope.get('complete_document_id')}",
                                request_id,
                            )
                        else:
                            memory_context_text = self._build_memory_context_text(
                                chat_session=chat_session,
                            )
                            memory_context_used = bool(memory_context_text.strip())

                        memory_context_chars = len(memory_context_text or "")
                        estimated_memory_tokens = int(memory_context_chars / 4) if memory_context_chars else 0

                        session_messages = self._safe_list(getattr(chat_session, "session_data", None))
                        summary_items = self._safe_list(getattr(chat_session, "conversation_summary", None))

                        debug_id(
                            "[ChatOrchestrator] Conversation memory size "
                            f"conversation_id={active_conversation_id} "
                            f"created_session={created_session} "
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

                        self._touch_chat_session(chat_session)

                        debug_id(
                            f"[ChatOrchestrator] Chat memory prepared "
                            f"conversation_id={active_conversation_id} "
                            f"created_session={created_session} "
                            f"memory_context_used={memory_context_used} "
                            f"document_scope_enabled={document_scope_enabled}",
                            request_id,
                        )

            except Exception as memory_error:
                warning_id(
                    f"Chat memory preparation failed; answer generation will continue without memory: "
                    f"{memory_error}",
                    request_id,
                    exc_info=True,
                )

                active_conversation_id = normalized_conversation_id
                memory_context_text = ""
                memory_context_used = False

            memory_time = time.perf_counter() - memory_start

            # --------------------------------------------------
            # 2. Generate answer only
            # --------------------------------------------------
            ai_start = time.perf_counter()

            with self.transaction() as session:
                with tracer.span(
                    "ai_answer_execute",
                    meta={
                        "user_id": normalized_user_id,
                        "client_type": normalized_client_type,
                        "forced_chunk_id": forced_chunk_id,
                        "include_payload": False,
                        "audit_pathway": self.AUDIT_PATHWAY_NAME,
                        "conversation_id": active_conversation_id,
                        "memory_context_used": memory_context_used,
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
                        question=normalized_question,
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
            ai_result["document_scope"] = (
                ai_result.get("document_scope")
                or ai_result.get("documentScope")
                or normalized_document_scope
            )
            ai_result["document_scope_enabled"] = bool(
                self._normalize_document_scope(ai_result.get("document_scope"))
            )

            # --------------------------------------------------
            # 3. Persist Q&A seed and audit
            # --------------------------------------------------
            persist_start = time.perf_counter()

            try:
                with self.transaction() as session:
                    with tracer.span(
                        "persist_qanda_seed",
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
                        qanda_record = self.qanda_service.create_interaction(
                            session=session,
                            user_id=normalized_user_id,
                            question=normalized_question,
                            answer=ai_result.get("answer", ""),
                            request_id=request_id,
                            processing_time_ms=int(ai_time * 1000),
                            raw_response=ai_result,
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
                        "audit_answer_search_pathway",
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
                            duration_ms=int((time.perf_counter() - request_start) * 1000),
                            model_name=ai_result.get("model_name"),
                        )

                    audit_time = time.perf_counter() - audit_start

            except Exception as persist_error:
                warning_id(
                    f"Q&A seed persistence or answer audit failed but answer response will continue: "
                    f"{persist_error}",
                    request_id,
                    exc_info=True,
                )

            persist_time = time.perf_counter() - persist_start

            # --------------------------------------------------
            # 4. Store assistant memory
            # --------------------------------------------------
            assistant_memory_start = time.perf_counter()

            if active_conversation_id:
                try:
                    with self.transaction() as session:
                        with tracer.span(
                            "chat_memory_store_assistant_answer",
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
                            )

                except Exception as memory_store_error:
                    warning_id(
                        f"Assistant memory update failed but answer response will continue: "
                        f"{memory_store_error}",
                        request_id,
                        exc_info=True,
                    )

            memory_time += time.perf_counter() - assistant_memory_start

            # --------------------------------------------------
            # 5. Embed Q&A
            # --------------------------------------------------
            embedding_start = time.perf_counter()

            if qanda_id:
                try:
                    with self.transaction() as session:
                        with tracer.span(
                            "qanda_embedding_update",
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
                            },
                        ):
                            debug_id(
                                f"[ChatOrchestrator] Calling QandAEmbeddingService "
                                f"request_id={request_id} qanda_id={qanda_id}",
                                request_id,
                            )

                            embeddings_updated = self.qanda_embedding_service.embed_existing_qanda(
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
                        f"Q&A embedding update failed but answer response will continue: "
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

            # --------------------------------------------------
            # 6. Return response
            # --------------------------------------------------
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
        """
        Execute the AI service with memory and document-scope support.

        Diagnostic purpose:
            This version keeps the same behavior as the existing implementation,
            but logs the final execute_kwargs shape before calling
            AIStewardManagerService.execute().

        This helps confirm whether conversational memory is being passed as:
            - memory_context
            - conversation_context
            - prepended directly into the question

        This method intentionally does NOT:
            - Change document-scope behavior
            - Add retry behavior
            - Disable memory
            - Modify the returned AI result
        """

        normalized_document_scope = self._normalize_document_scope(document_scope)
        memory_context_text = (memory_context_text or "").strip()

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

            debug_id(
                "[ChatOrchestrator] AI service signature inspected "
                f"request_id={request_id} "
                f"accepts_kwargs={accepts_kwargs} "
                f"accepts_document_scope={'document_scope' in parameters or accepts_kwargs} "
                f"accepts_memory_context={'memory_context' in parameters or accepts_kwargs} "
                f"accepts_conversation_context={'conversation_context' in parameters} "
                f"accepts_conversation_id={'conversation_id' in parameters or accepts_kwargs}",
                request_id,
            )

            if normalized_document_scope:
                if "document_scope" in parameters or accepts_kwargs:
                    execute_kwargs["document_scope"] = normalized_document_scope

                    debug_id(
                        f"[ChatOrchestrator] Passing document_scope to AI service "
                        f"complete_document_id={normalized_document_scope.get('complete_document_id')}",
                        request_id,
                    )

                else:
                    warning_id(
                        f"[ChatOrchestrator] document_scope active but "
                        f"AIStewardManagerService.execute does not accept document_scope yet. "
                        f"Continuing without scoped retrieval. "
                        f"complete_document_id={normalized_document_scope.get('complete_document_id')}",
                        request_id,
                    )

            if memory_context_text:
                if "memory_context" in parameters or accepts_kwargs:
                    execute_kwargs["memory_context"] = memory_context_text
                    execute_kwargs["conversation_id"] = conversation_id

                    debug_id(
                        "[ChatOrchestrator] Passing conversational memory as memory_context "
                        f"request_id={request_id} "
                        f"conversation_id={conversation_id} "
                        f"memory_context_chars={len(memory_context_text)}",
                        request_id,
                    )

                elif "conversation_context" in parameters:
                    execute_kwargs["conversation_context"] = memory_context_text
                    execute_kwargs["conversation_id"] = conversation_id

                    debug_id(
                        "[ChatOrchestrator] Passing conversational memory as conversation_context "
                        f"request_id={request_id} "
                        f"conversation_id={conversation_id} "
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
                        f"conversation_id={conversation_id} "
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
                    f"conversation_id={conversation_id} "
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
                f"has_document_scope={'document_scope' in execute_kwargs} "
                f"question_chars={len(str(execute_kwargs.get('question') or ''))} "
                f"memory_context_chars={len(str(execute_kwargs.get('memory_context') or execute_kwargs.get('conversation_context') or ''))} "
                f"document_scope_enabled={bool(normalized_document_scope)} "
                f"execute_keys={sorted(execute_kwargs.keys())}",
                request_id,
            )

            return self.ai_service.execute(**execute_kwargs)

        except TypeError as type_error:
            # Defensive fallback if the service signature looked compatible but
            # the actual implementation rejects newer kwargs.
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
                        f"[ChatOrchestrator] AI service rejected kwargs {removed_keys}; "
                        f"retrying with compatibility fallback. error={type_error}",
                        request_id,
                        exc_info=True,
                    )

                    if memory_context_text:
                        execute_kwargs["question"] = self._build_memory_augmented_question(
                            question=question,
                            memory_context_text=memory_context_text,
                        )

                    debug_id(
                        "[ChatOrchestrator] AI service execute kwargs after TypeError fallback "
                        f"request_id={request_id} "
                        f"removed_keys={removed_keys} "
                        f"has_memory_context={'memory_context' in execute_kwargs} "
                        f"has_conversation_context={'conversation_context' in execute_kwargs} "
                        f"has_conversation_id={'conversation_id' in execute_kwargs} "
                        f"has_document_scope={'document_scope' in execute_kwargs} "
                        f"question_chars={len(str(execute_kwargs.get('question') or ''))} "
                        f"memory_context_chars={len(str(execute_kwargs.get('memory_context') or execute_kwargs.get('conversation_context') or ''))} "
                        f"document_scope_enabled={bool(normalized_document_scope)} "
                        f"execute_keys={sorted(execute_kwargs.keys())}",
                        request_id,
                    )

                    return self.ai_service.execute(**execute_kwargs)

            raise

    def _get_or_create_chat_session(
        self,
        *,
        session,
        conversation_id: Optional[str],
        user_id: str,
        request_id: Optional[str],
    ) -> Tuple[ChatSession, bool]:
        """
        Load an existing ChatSession or create a new one.

        Returns:
            tuple(ChatSession, created_session)
        """

        parsed_conversation_id = self._coerce_uuid_or_none(conversation_id)

        if parsed_conversation_id is not None:
            existing_session = session.get(ChatSession, parsed_conversation_id)

            if existing_session is not None:
                existing_user_id = str(getattr(existing_session, "user_id", "") or "")
                requested_user_id = str(user_id or "")

                if existing_user_id == requested_user_id:
                    return existing_session, False

                warning_id(
                    f"[ChatOrchestrator] Conversation ID belongs to a different user. "
                    f"incoming_conversation_id={conversation_id} "
                    f"existing_user_id={existing_user_id} requested_user_id={requested_user_id}. "
                    f"Creating a new ChatSession.",
                    request_id,
                )

            else:
                warning_id(
                    f"[ChatOrchestrator] Conversation ID was supplied but no ChatSession was found. "
                    f"incoming_conversation_id={conversation_id}. Creating a new ChatSession.",
                    request_id,
                )

        elif conversation_id:
            warning_id(
                f"[ChatOrchestrator] Invalid conversation_id supplied: {conversation_id!r}. "
                f"Creating a new ChatSession.",
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
    ) -> None:
        """
        Store assistant answer and update lightweight rolling conversation summary.
        """

        parsed_conversation_id = self._coerce_uuid_or_none(conversation_id)
        normalized_document_scope = self._normalize_document_scope(document_scope)

        if parsed_conversation_id is None:
            warning_id(
                f"[ChatOrchestrator] Cannot store assistant memory because conversation_id "
                f"is invalid: {conversation_id!r}",
                request_id,
            )
            return

        chat_session = session.get(ChatSession, parsed_conversation_id)

        if chat_session is None:
            warning_id(
                f"[ChatOrchestrator] Cannot store assistant memory because ChatSession "
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
            },
        )

        self._append_conversation_summary_item(
            chat_session=chat_session,
            question=question,
            answer=answer or "",
            request_id=request_id,
            qanda_id=qanda_id,
            document_scope=normalized_document_scope,
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
        """
        Append one message to ChatSession.session_data.

        session_data is stored as JSON, so we reassign the list after mutation
        to make sure SQLAlchemy tracks the change.
        """

        messages = self._safe_list(getattr(chat_session, "session_data", None))

        messages.append({
            "role": role,
            "content": content or "",
            "request_id": request_id,
            "created_at": self._utc_iso(),
            "metadata": metadata or {},
        })

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
    ) -> None:
        """
        Append a lightweight turn summary.

        This is not an AI-generated summary yet. It is a compact rolling memory
        item that gives the bot useful prior-turn context without loading the
        entire raw conversation forever.
        """

        normalized_document_scope = self._normalize_document_scope(document_scope)
        summaries = self._safe_list(getattr(chat_session, "conversation_summary", None))

        summaries.append({
            "request_id": request_id,
            "qanda_id": str(qanda_id) if qanda_id else None,
            "created_at": self._utc_iso(),
            "question": self._preview_text(question, self.MEMORY_PREVIEW_CHARS),
            "answer_preview": self._preview_text(answer, self.SUMMARY_ANSWER_PREVIEW_CHARS),
            "document_scope": normalized_document_scope,
            "document_scope_enabled": bool(normalized_document_scope),
        })

        if len(summaries) > self.SUMMARY_MAX_ITEMS:
            summaries = summaries[-self.SUMMARY_MAX_ITEMS:]

        chat_session.conversation_summary = summaries

    def _build_memory_context_text(
        self,
        *,
        chat_session: ChatSession,
    ) -> str:
        """
        Build prompt-ready memory context from ChatSession.

        Uses:
            - Recent rolling conversation_summary items
            - Recent raw session_data messages

        This is intentionally lightweight and safe for first implementation.
        """

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

                summary_scope = self._normalize_document_scope(item.get("document_scope"))

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

                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                message_scope = self._normalize_document_scope(metadata.get("document_scope"))

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
        """
        Compatibility fallback when the AI service does not accept a separate
        memory_context argument.
        """

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
    ) -> Dict[str, Any]:

        method = ai_result.get("method") or ai_result.get("strategy") or self.DEFAULT_METHOD
        strategy = ai_result.get("strategy") or method
        normalized_document_scope = (
            self._normalize_document_scope(ai_result.get("document_scope"))
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
            "qanda_id": str(qanda_id) if qanda_id else None,
            "response_time": total_time,
            "payload_status": "pending",
            "payload_endpoint": "/ask/payload",
            "debug_mode": bool(ai_result.get("debug_mode", False)),
            "debug_chunk_id": ai_result.get("debug_chunk_id"),
            "retriever_top_k": ai_result.get("retriever_top_k"),
            "used_chunks_count": len(ai_result.get("used_chunks") or []),
            "memory_enabled": bool(conversation_id),
            "memory_context_used": bool(memory_context_used),
            "embeddings_updated": bool(embeddings_updated),
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
            },
        }

    @staticmethod
    def _extract_record_id(record: Any) -> Optional[Any]:
        """
        Extract an id from a Q&A record returned by QandAService.

        Supports:
            - ORM object with .id
            - ORM/object with .qanda_id
            - dictionary with "id"
            - dictionary with "qanda_id"
            - dictionary with "qa_id"
            - dictionary with "record_id"
            - None
        """

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
        """
        Resolve QandA.id from the database when create_interaction() returns an ORM
        object before the primary key is available from the returned object.
        """

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

        complete_document_id = ChatOrchestrator._coerce_int_or_none(complete_document_id)

        if complete_document_id is None:
            return None

        document_id = (
            document_scope.get("document_id")
            or document_scope.get("documentId")
        )

        document_id = ChatOrchestrator._coerce_int_or_none(document_id)

        document_name = (
            document_scope.get("document_name")
            or document_scope.get("documentName")
            or document_scope.get("name")
            or document_scope.get("title")
            or f"Document #{complete_document_id}"
        )

        document_name = str(document_name or "").strip() or f"Document #{complete_document_id}"

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
        """
        Coerce a value to UUID if possible.

        Used for:
            - ChatSession.session_id
            - QandA.id when SearchAuditService expects UUID | None

        If a future service returns a non-UUID ID, this safely returns None.
        """

        if value is None:
            return None

        if isinstance(value, UUID):
            return value

        try:
            return UUID(str(value))
        except (TypeError, ValueError, AttributeError):
            return None