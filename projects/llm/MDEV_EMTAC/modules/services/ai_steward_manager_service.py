# modules/services/ai_steward_manager_service.py

from __future__ import annotations

import inspect
from typing import Dict, Any, Optional, List, Tuple

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    with_request_id,
    error_id,
    debug_id,
    warning_id,
)

from modules.services.unified_search_service import UnifiedSearchService
from modules.observability.high_end_tracer import tracer


class AIStewardManagerService:
    """
    Stateless AI execution service.

    Responsibilities:
        - Call UnifiedSearchService
        - Pass conversational memory into the search/RAG pathway when available
        - Pass document_scope into the search/RAG pathway when available
        - Return raw/enriched domain result
        - Optionally project supporting UI payload
        - No persistence
        - No transaction ownership
        - No session creation
        - No final UI formatting

    Notes:
        - ChatOrchestrator owns the answer transaction.
        - ChatPayloadOrchestrator owns the supporting payload transaction.
        - This service bridges UnifiedSearchService/RAG and document UI payload projection.

    Conversational memory flow:
        ChatOrchestrator
            -> builds memory_context from ChatSession.session_data / conversation_summary
            -> passes memory_context + conversation_id here

        AIStewardManagerService
            -> passes memory_context + conversation_id into UnifiedSearchService when supported

        UnifiedSearchService / RAG
            -> should pass memory_context into the final prompt builder as its own
               section, separate from the user's actual question.

    Document-scoped conversation mode flow:
        Frontend
            -> sends document_scope with /chatbot/ask

        Route / Coordinator / Orchestrator
            -> validate and forward document_scope

        AIStewardManagerService
            -> passes document_scope into UnifiedSearchService when supported

        UnifiedSearchService / RAG
            -> should restrict retrieval to the selected complete_document_id
            -> should tell the model it may answer only from the selected document
    """

    def __init__(self):
        self.search_service = UnifiedSearchService(
            enable_intent=False,
            enable_vector=True,
            enable_fts=True,
        )

    # ---------------------------------------------------------
    # Main Execution
    # ---------------------------------------------------------

    @with_request_id
    def execute(
        self,
        *,
        session: Session,
        user_id: str,
        question: str,
        client_type: Optional[str] = None,
        request_id: Optional[str] = None,
        rag_only: bool = True,
        forced_chunk_id: Optional[int] = None,
        include_payload: bool = True,
        memory_context: Optional[str] = None,
        conversation_id: Optional[str] = None,
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the RAG/search pathway.

        include_payload=True:
            Legacy/full behavior.
            Runs search/RAG and allows supporting UI payload generation.

        include_payload=False:
            Answer-first behavior.
            Runs search/RAG and returns the answer plus raw seed fields only.
            Skips heavy UI payload projection so /ask can return faster.

        Conversational memory:
            memory_context:
                Prompt-ready conversational memory text built by ChatOrchestrator.

            conversation_id:
                Active ChatSession.session_id returned to the browser and sent back
                on follow-up questions.

        Document-scoped conversation mode:
            document_scope:
                {
                    "enabled": true,
                    "scope_type": "complete_document",
                    "document_id": 29,
                    "complete_document_id": 29,
                    "document_name": "Document #29"
                }

            This service only forwards the scope. Actual retrieval filtering belongs
            in UnifiedSearchService / retriever / RAG pipeline.

        Important:
            include_payload=False must still preserve:
                - chunks
                - used_chunks
                - retriever_top_k
                - debug metadata
                - conversation_id
                - memory flags
                - document_scope metadata

            The payload route later rebuilds relationship_map and UI payload
            through UnifiedSearchService.build_payload_from_seed().
        """

        question = (question or "").strip()
        normalized_memory_context = (memory_context or "").strip()
        normalized_conversation_id = (
            str(conversation_id).strip()
            if conversation_id is not None and str(conversation_id).strip()
            else None
        )
        normalized_document_scope = self._normalize_document_scope(document_scope)

        if not question:
            return self._empty_response(
                strategy="invalid_input",
                answer="Please provide a more detailed question.",
                model_name=None,
                payload_status="unavailable",
                conversation_id=normalized_conversation_id,
                memory_context_used=bool(normalized_memory_context),
                memory_context_mode="not_used_invalid_input",
                document_scope=normalized_document_scope,
            )

        try:
            # --------------------------------------------------
            # 1. Execute search / RAG
            # --------------------------------------------------
            with tracer.span(
                "unified_search_execute",
                meta={
                    "rag_only": rag_only,
                    "client_type": client_type,
                    "forced_chunk_id": forced_chunk_id,
                    "include_payload": include_payload,
                    "conversation_id": normalized_conversation_id,
                    "memory_context_present": bool(normalized_memory_context),
                    "document_scope_enabled": bool(normalized_document_scope),
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
                result, context_modes = self._execute_unified_search(
                    session=session,
                    user_id=user_id,
                    question=question,
                    request_id=request_id,
                    rag_only=rag_only,
                    forced_chunk_id=forced_chunk_id,
                    include_payload=include_payload,
                    memory_context=normalized_memory_context,
                    conversation_id=normalized_conversation_id,
                    document_scope=normalized_document_scope,
                )

            if not isinstance(result, dict):
                warning_id(
                    "[AIStewardManagerService] UnifiedSearchService returned "
                    "non-dict result; replacing with empty result.",
                    request_id,
                )
                result = {}

            # --------------------------------------------------
            # 2. Normalize raw result shape
            # --------------------------------------------------
            result = self._normalize_raw_result(
                result=result,
                fallback_strategy="rag",
            )

            memory_context_mode = context_modes.get("memory_context_mode", "none")
            document_scope_mode = context_modes.get("document_scope_mode", "none")

            # --------------------------------------------------
            # 3. Preserve memory metadata
            # --------------------------------------------------
            result["conversation_id"] = normalized_conversation_id
            result["memory_enabled"] = bool(normalized_conversation_id)
            result["memory_context_used"] = bool(normalized_memory_context)
            result["memory_context_mode"] = memory_context_mode

            # --------------------------------------------------
            # 4. Preserve document scope metadata
            # --------------------------------------------------
            result["document_scope"] = (
                self._normalize_document_scope(result.get("document_scope"))
                or self._normalize_document_scope(result.get("documentScope"))
                or normalized_document_scope
            )
            result["document_scope_enabled"] = bool(result["document_scope"])
            result["document_scope_mode"] = document_scope_mode

            # --------------------------------------------------
            # 5. Preserve forced chunk/debug metadata
            # --------------------------------------------------
            if forced_chunk_id is not None:
                result.setdefault("debug_mode", True)
                result.setdefault("debug_chunk_id", forced_chunk_id)
            else:
                result.setdefault("debug_mode", False)
                result.setdefault("debug_chunk_id", None)

            # --------------------------------------------------
            # 6. Inject model_name safely
            # --------------------------------------------------
            result["model_name"] = self._resolve_model_name(
                fallback=result.get("model_name"),
                request_id=request_id,
            )

            # --------------------------------------------------
            # 7. Final answer-path payload status
            # --------------------------------------------------
            if include_payload:
                # UnifiedSearchService should already have built the payload.
                # This safety projection remains for older/legacy result shapes
                # where relationship_map exists but documents/images/parts/drawings
                # were not populated.
                if self._should_apply_projection_safety_net(result):
                    with tracer.span("ai_result_enrichment_safety_net"):
                        result = self._apply_relationship_projection(
                            session=session,
                            result=result,
                            request_id=request_id,
                        )

                result["payload_status"] = "complete"

            else:
                # Answer-first mode:
                # Do not build documents/images/parts/drawings here.
                # Keep chunks/used_chunks intact so the payload route can
                # build the supporting panels later.
                result["payload_status"] = "pending"
                result["documents"] = []
                result["images"] = []
                result["drawings"] = []
                result["parts"] = []

            debug_id(
                "[AIStewardManagerService] Result ready "
                f"(strategy={result.get('strategy')}, "
                f"method={result.get('method')}, "
                f"payload_status={result.get('payload_status')}, "
                f"include_payload={include_payload}, "
                f"forced_chunk_id={forced_chunk_id}, "
                f"conversation_id={normalized_conversation_id}, "
                f"memory_context_used={bool(normalized_memory_context)}, "
                f"memory_context_mode={memory_context_mode}, "
                f"document_scope_enabled={bool(result.get('document_scope'))}, "
                f"document_scope_mode={document_scope_mode}, "
                f"complete_document_id="
                f"{result.get('document_scope', {}).get('complete_document_id') if isinstance(result.get('document_scope'), dict) else None}, "
                f"chunks={len(result.get('chunks') or [])}, "
                f"used_chunks={len(result.get('used_chunks') or [])}, "
                f"relationship_map={'yes' if isinstance(result.get('relationship_map'), dict) and result.get('relationship_map') else 'no'}, "
                f"docs={len(result.get('documents') or [])}, "
                f"images={len(result.get('images') or [])}, "
                f"parts={len(result.get('parts') or [])}, "
                f"drawings={len(result.get('drawings') or [])})",
                request_id,
            )

            return result

        except Exception as e:
            error_id(
                f"AIStewardManagerService failure: {e}",
                request_id,
                exc_info=True,
            )

            return self._empty_response(
                strategy="error",
                answer="AI processing error.",
                model_name=None,
                payload_status="unavailable",
                conversation_id=normalized_conversation_id,
                memory_context_used=bool(normalized_memory_context),
                memory_context_mode="error",
                document_scope=normalized_document_scope,
            )

    # ---------------------------------------------------------
    # Unified Search Execution Adapter
    # ---------------------------------------------------------

    def _execute_unified_search(
        self,
        *,
        session: Session,
        user_id: str,
        question: str,
        request_id: Optional[str],
        rag_only: bool,
        forced_chunk_id: Optional[int],
        include_payload: bool,
        memory_context: Optional[str],
        conversation_id: Optional[str],
        document_scope: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Calls UnifiedSearchService.execute() while staying backward-compatible.

        Preferred path:
            UnifiedSearchService.execute(
                ...,
                memory_context=...,
                conversation_id=...,
                document_scope=...
            )

        Safe fallback:
            If UnifiedSearchService.execute() does not yet accept memory_context,
            this method DOES NOT prepend memory to the question.

            If UnifiedSearchService.execute() does not yet accept document_scope,
            this method DOES NOT pretend scoped retrieval happened. It logs a
            warning and continues with normal search.

        Why:
            Prepending memory into the question can leak internal memory instructions
            into the final user-visible answer.

            Document scope must be enforced as a retrieval filter, not by adding
            plain text to the question.
        """

        execute_kwargs: Dict[str, Any] = {
            "session": session,
            "user_id": user_id,
            "question": question,
            "request_id": request_id,
            "rag_only": rag_only,
            "forced_chunk_id": forced_chunk_id,
            "include_payload": include_payload,
        }

        normalized_memory_context = (memory_context or "").strip()
        normalized_conversation_id = (
            str(conversation_id).strip()
            if conversation_id is not None and str(conversation_id).strip()
            else None
        )
        normalized_document_scope = self._normalize_document_scope(document_scope)

        memory_context_mode = "none"
        document_scope_mode = "none"

        support = self._get_callable_support(self.search_service.execute)

        # --------------------------------------------------
        # Memory context
        # --------------------------------------------------
        # Only pass memory if UnifiedSearchService accepts it separately.
        # Do NOT inject memory into the question as a fallback.
        if normalized_memory_context and (
            support["accepts_kwargs"] or support["supports_memory_context"]
        ):
            execute_kwargs["memory_context"] = normalized_memory_context
            memory_context_mode = "separate_memory_context"

        elif normalized_memory_context:
            memory_context_mode = "not_passed_unified_search_missing_support"

            warning_id(
                "[AIStewardManagerService] UnifiedSearchService.execute() does not accept "
                "memory_context yet. Memory was NOT injected into the question fallback "
                "because that can leak conversation-memory instructions into the final answer. "
                "Update UnifiedSearchService/RAG to accept memory_context as a separate prompt section.",
                request_id,
            )

        # --------------------------------------------------
        # Conversation ID
        # --------------------------------------------------
        # conversation_id is safe metadata. Pass it only if the next layer accepts it.
        if normalized_conversation_id and (
            support["accepts_kwargs"] or support["supports_conversation_id"]
        ):
            execute_kwargs["conversation_id"] = normalized_conversation_id

            if memory_context_mode == "none":
                memory_context_mode = "conversation_id_only"

        elif normalized_conversation_id:
            debug_id(
                "[AIStewardManagerService] UnifiedSearchService.execute() does not "
                "accept conversation_id yet. Continuing without passing it downstream.",
                request_id,
            )

        # --------------------------------------------------
        # Document scope
        # --------------------------------------------------
        # document_scope must be passed as structured data.
        # Do NOT append it to question text as a fallback.
        if normalized_document_scope and (
            support["accepts_kwargs"] or support["supports_document_scope"]
        ):
            execute_kwargs["document_scope"] = normalized_document_scope
            document_scope_mode = "separate_document_scope"

            debug_id(
                "[AIStewardManagerService] Passing document_scope to UnifiedSearchService "
                f"complete_document_id={normalized_document_scope.get('complete_document_id')} "
                f"document_name={normalized_document_scope.get('document_name')}",
                request_id,
            )

        elif normalized_document_scope:
            document_scope_mode = "not_passed_unified_search_missing_support"

            warning_id(
                "[AIStewardManagerService] UnifiedSearchService.execute() does not accept "
                "document_scope yet. Document-scoped retrieval has NOT been enforced yet. "
                "Update UnifiedSearchService/RAG retrieval to filter by complete_document_id. "
                f"complete_document_id={normalized_document_scope.get('complete_document_id')} "
                f"document_name={normalized_document_scope.get('document_name')}",
                request_id,
            )

        try:
            result = self.search_service.execute(**execute_kwargs)
            return result, {
                "memory_context_mode": memory_context_mode,
                "document_scope_mode": document_scope_mode,
            }

        except TypeError as type_error:
            # Defensive retry:
            # If signature inspection was wrong or UnifiedSearchService has a
            # wrapped/decorated signature, retry without newer kwargs.
            type_error_text = str(type_error)

            rejected_kwargs = (
                "unexpected keyword argument" in type_error_text
                and (
                    "memory_context" in type_error_text
                    or "conversation_id" in type_error_text
                    or "document_scope" in type_error_text
                )
            )

            if not rejected_kwargs:
                raise

            warning_id(
                "[AIStewardManagerService] UnifiedSearchService rejected optional kwargs. "
                "Retrying without memory/conversation/document-scope kwargs. "
                "Memory was NOT injected into the question fallback, and document scope "
                "was NOT converted into question text. "
                f"Error: {type_error}",
                request_id,
                exc_info=True,
            )

            if "memory_context" in execute_kwargs:
                execute_kwargs.pop("memory_context", None)
                memory_context_mode = "not_passed_after_type_error"

            if "conversation_id" in execute_kwargs:
                execute_kwargs.pop("conversation_id", None)

                if memory_context_mode == "conversation_id_only":
                    memory_context_mode = "none_after_type_error"

            if "document_scope" in execute_kwargs:
                execute_kwargs.pop("document_scope", None)
                document_scope_mode = "not_passed_after_type_error"

            result = self.search_service.execute(**execute_kwargs)
            return result, {
                "memory_context_mode": memory_context_mode,
                "document_scope_mode": document_scope_mode,
            }

    @staticmethod
    def _get_callable_support(callable_obj: Any) -> Dict[str, bool]:
        """
        Inspects a callable to determine whether optional kwargs can be passed.

        Decorators may hide signatures, so this method is intentionally defensive.
        """

        support = {
            "accepts_kwargs": False,
            "supports_memory_context": False,
            "supports_conversation_id": False,
            "supports_document_scope": False,
        }

        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return support

        parameters = signature.parameters

        support["accepts_kwargs"] = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        support["supports_memory_context"] = "memory_context" in parameters
        support["supports_conversation_id"] = "conversation_id" in parameters
        support["supports_document_scope"] = "document_scope" in parameters

        return support

    @staticmethod
    def _build_memory_augmented_question(
        *,
        question: str,
        memory_context: str,
    ) -> str:
        """
        Temporary compatibility fallback.

        This helper is intentionally NOT used when UnifiedSearchService lacks
        memory_context support. Keeping it here avoids breaking older imports,
        but memory should be passed as a separate prompt section.
        """

        return (
            "Use the conversation memory below only to understand context, references, "
            "and follow-up wording. The current user question remains the main request. "
            "Do not let memory override retrieved database/document evidence.\n\n"
            f"Conversation memory:\n{memory_context.strip()}\n\n"
            f"Current user question:\n{question.strip()}"
        )

    # ---------------------------------------------------------
    # Payload Projection Entry Point
    # ---------------------------------------------------------

    @with_request_id
    def project_payload(
        self,
        *,
        session: Session,
        result: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds supporting UI payload from an existing answer/RAG result.

        This is intended for the second request:

            ChatPayloadOrchestrator.load_payload()

        Expected input fields:
            - chunks or used_chunks
            - answer
            - strategy/method
            - optional relationship_map
            - optional conversation_id / memory metadata
            - optional document_scope metadata

        Preferred flow:
            UnifiedSearchService.build_payload_from_seed()

        Fallback flow:
            _apply_relationship_projection()
        """

        if not isinstance(result, dict):
            warning_id(
                "[AIStewardManagerService] project_payload received non-dict result.",
                request_id,
            )
            result = {}

        result = self._normalize_raw_result(
            result=result,
            fallback_strategy="rag",
        )

        with tracer.span(
            "ai_payload_projection",
            meta={
                "document_scope_enabled": bool(
                    self._normalize_document_scope(result.get("document_scope"))
                ),
                "complete_document_id": (
                    self._normalize_document_scope(result.get("document_scope")).get("complete_document_id")
                    if self._normalize_document_scope(result.get("document_scope"))
                    else None
                ),
            },
        ):
            if hasattr(self.search_service, "build_payload_from_seed"):
                result = self.search_service.build_payload_from_seed(
                    session=session,
                    result=result,
                    request_id=request_id,
                )
            else:
                warning_id(
                    "[AIStewardManagerService] UnifiedSearchService does not have "
                    "build_payload_from_seed(); falling back to local projection.",
                    request_id,
                )

                result = self._apply_relationship_projection(
                    session=session,
                    result=result,
                    request_id=request_id,
                )

        if not isinstance(result, dict):
            warning_id(
                "[AIStewardManagerService] project_payload returned non-dict result.",
                request_id,
            )
            result = {}

        result = self._normalize_raw_result(
            result=result,
            fallback_strategy="rag",
        )

        result["payload_status"] = result.get("payload_status") or "complete"

        debug_id(
            "[AIStewardManagerService] Payload projection ready "
            f"payload_status={result.get('payload_status')} "
            f"conversation_id={result.get('conversation_id')} "
            f"document_scope_enabled={bool(result.get('document_scope'))} "
            f"complete_document_id="
            f"{result.get('document_scope', {}).get('complete_document_id') if isinstance(result.get('document_scope'), dict) else None} "
            f"documents={len(result.get('documents') or [])} "
            f"images={len(result.get('images') or [])} "
            f"parts={len(result.get('parts') or [])} "
            f"drawings={len(result.get('drawings') or [])}",
            request_id,
        )

        return result

    # ---------------------------------------------------------
    # Relationship Projection Hook
    # ---------------------------------------------------------

    def _apply_relationship_projection(
        self,
        *,
        session: Session,
        result: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Converts RAG chunks + relationship_map into richer document payloads.

        Required input fields:
            - used_chunks or chunks
            - relationship_map

        Output fields:
            - documents
            - images
            - parts
            - drawings

        Notes:
            - This method does not create relationship_map.
            - UnifiedSearchService.build_payload_from_seed() is preferred for
              second-pass payload loading because it can rebuild relationship_map
              when the answer-first seed intentionally skipped it.
        """

        relationship_map = result.get("relationship_map")

        debug_id(
            "[AIStewardManagerService] projection precheck "
            f"strategy={result.get('strategy')} "
            f"method={result.get('method')} "
            f"used_chunks={len(result.get('used_chunks') or [])} "
            f"chunks={len(result.get('chunks') or [])} "
            f"documents={len(result.get('documents') or [])} "
            f"images={len(result.get('images') or [])} "
            f"relationship_map={'yes' if isinstance(relationship_map, dict) and relationship_map else 'no'}",
            request_id,
        )

        if not isinstance(relationship_map, dict) or not relationship_map:
            debug_id(
                "[AIStewardManagerService] No relationship_map found; "
                "leaving document payload unchanged.",
                request_id,
            )
            return result

        chunks = self._resolve_projection_chunks(result)

        if not chunks:
            debug_id(
                "[AIStewardManagerService] No chunks available for relationship "
                "projection; leaving document payload unchanged.",
                request_id,
            )
            return result

        try:
            from modules.ai.search_pathway.rag_core.chunk_relationship_projection import (
                ChunkRelationshipProjection,
            )
        except Exception as e:
            warning_id(
                "[AIStewardManagerService] ChunkRelationshipProjection import failed; "
                f"leaving payload unchanged. Error: {e}",
                request_id,
            )
            return result

        try:
            projection = ChunkRelationshipProjection(session=session)

            projected = projection.project_chunks_for_ui(
                chunks=chunks,
                relationship_map=relationship_map,
            )

            if not isinstance(projected, dict):
                warning_id(
                    "[AIStewardManagerService] Relationship projection returned "
                    "non-dict payload; leaving payload unchanged.",
                    request_id,
                )
                return result

            documents = projected.get("documents-container")

            if isinstance(documents, list):
                result["documents"] = documents

            summary = projected.get("summary")
            if summary is not None:
                result["relationship_summary"] = summary

            for result_key, projected_key in (
                ("images", "images-container"),
                ("parts", "parts-container"),
                ("drawings", "drawings-container"),
            ):
                items = projected.get(projected_key)

                if isinstance(items, list):
                    result[result_key] = items

            result["images"] = self._promote_unique_items_from_documents(
                documents=result.get("documents") or [],
                field_names=("images", "part_images"),
                existing_items=result.get("images") or [],
            )

            result["parts"] = self._promote_unique_items_from_documents(
                documents=result.get("documents") or [],
                field_names=("parts",),
                existing_items=result.get("parts") or [],
            )

            result["drawings"] = self._promote_unique_items_from_documents(
                documents=result.get("documents") or [],
                field_names=("drawings",),
                existing_items=result.get("drawings") or [],
            )

            debug_id(
                "[AIStewardManagerService] Applied relationship projection "
                f"documents={len(result.get('documents') or [])} "
                f"images={len(result.get('images') or [])} "
                f"parts={len(result.get('parts') or [])} "
                f"drawings={len(result.get('drawings') or [])}",
                request_id,
            )

            return result

        except Exception as e:
            warning_id(
                "[AIStewardManagerService] Relationship projection failed; "
                f"leaving payload unchanged. Error: {e}",
                request_id,
            )
            return result

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_raw_result(
        *,
        result: Dict[str, Any],
        fallback_strategy: str = "rag",
    ) -> Dict[str, Any]:
        """
        Normalizes raw UnifiedSearchService output.

        This intentionally preserves relationship_map, chunks, used_chunks,
        conversation_id, memory metadata, and document_scope metadata because
        those are required for answer-first response handling and second-pass
        payload loading.
        """

        if not isinstance(result, dict):
            result = {}

        result.setdefault("status", "success")
        result.setdefault("strategy", fallback_strategy)
        result.setdefault("method", result.get("strategy", fallback_strategy))
        result.setdefault("answer", "")

        result.setdefault("chunks", [])
        result.setdefault("used_chunks", [])
        result.setdefault("documents", [])
        result.setdefault("images", [])
        result.setdefault("drawings", [])
        result.setdefault("parts", [])

        result.setdefault("conversation_id", None)
        result.setdefault("memory_enabled", False)
        result.setdefault("memory_context_used", False)
        result.setdefault("memory_context_mode", "none")

        normalized_document_scope = AIStewardManagerService._normalize_document_scope(
            result.get("document_scope") or result.get("documentScope")
        )
        result["document_scope"] = normalized_document_scope
        result["document_scope_enabled"] = bool(normalized_document_scope)
        result.setdefault("document_scope_mode", "none")

        if "relationship_map" not in result:
            result["relationship_map"] = {}

        return result

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

        complete_document_id = AIStewardManagerService._coerce_int_or_none(
            complete_document_id
        )

        if complete_document_id is None:
            return None

        document_id = (
            document_scope.get("document_id")
            or document_scope.get("documentId")
        )

        document_id = AIStewardManagerService._coerce_int_or_none(document_id)

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
    def _resolve_projection_chunks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Finds the best chunk list for UI projection.

        Preference:
            1. used_chunks - cleaned/context-selected chunks from RAG
            2. chunks - raw chunk list if used_chunks is unavailable
        """

        used_chunks = result.get("used_chunks")

        if isinstance(used_chunks, list) and used_chunks:
            return used_chunks

        chunks = result.get("chunks")

        if isinstance(chunks, list) and chunks:
            return chunks

        return []

    @staticmethod
    def _promote_unique_items_from_documents(
        *,
        documents: List[Dict[str, Any]],
        field_names: tuple[str, ...],
        existing_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Promotes nested document-level items to top-level result containers.

        Example:
            doc["images"] -> result["images"]
            doc["part_images"] -> result["images"]

        Dedupes by:
            1. item["id"] if available
            2. item["src"] if available
            3. object identity fallback
        """

        promoted: List[Dict[str, Any]] = []
        seen = set()

        def add_item(item: Any) -> None:
            if not isinstance(item, dict):
                return

            item_id = item.get("id")
            item_src = item.get("src")

            if item_id is not None:
                key = ("id", item_id)
            elif item_src:
                key = ("src", item_src)
            else:
                key = ("obj", id(item))

            if key in seen:
                return

            seen.add(key)
            promoted.append(item)

        for item in existing_items or []:
            add_item(item)

        for doc in documents or []:
            if not isinstance(doc, dict):
                continue

            for field_name in field_names:
                nested = doc.get(field_name)

                if isinstance(nested, list):
                    for item in nested:
                        add_item(item)

        return promoted

    @staticmethod
    def _should_apply_projection_safety_net(result: Dict[str, Any]) -> bool:
        """
        Returns True when a legacy result has relationship_map but does not
        yet have any UI payload populated.
        """

        relationship_map = result.get("relationship_map")

        if not isinstance(relationship_map, dict) or not relationship_map:
            return False

        has_payload = any(
            bool(result.get(key))
            for key in ("documents", "images", "parts", "drawings")
        )

        return not has_payload

    def _resolve_model_name(
        self,
        *,
        fallback: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Safely resolves the active model name from the search/RAG stack.

        This must never break the chat flow.
        """

        if fallback:
            return fallback

        try:
            rag_pipeline = getattr(self.search_service, "rag_pipeline", None)

            if not rag_pipeline:
                return None

            answer_generator = getattr(rag_pipeline, "answer_generator", None)

            if not answer_generator:
                return None

            model_name = getattr(answer_generator, "model_name", None)

            if model_name:
                return model_name

            model_config = getattr(answer_generator, "model_config", None)

            if model_config:
                return getattr(model_config, "model_name", None)

            return None

        except Exception as e:
            debug_id(
                f"[AIStewardManagerService] Could not resolve model name: {e}",
                request_id,
            )
            return None

    @staticmethod
    def _empty_response(
        *,
        strategy: str,
        answer: str,
        model_name: Optional[str],
        payload_status: str = "unavailable",
        conversation_id: Optional[str] = None,
        memory_context_used: bool = False,
        memory_context_mode: str = "none",
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        normalized_document_scope = AIStewardManagerService._normalize_document_scope(
            document_scope
        )

        return {
            "status": "error" if strategy == "error" else "success",
            "strategy": strategy,
            "method": strategy,
            "answer": answer,
            "chunks": [],
            "used_chunks": [],
            "relationship_map": {},
            "documents": [],
            "images": [],
            "drawings": [],
            "parts": [],
            "model_name": model_name,
            "payload_status": payload_status,
            "debug_mode": False,
            "debug_chunk_id": None,
            "conversation_id": conversation_id,
            "memory_enabled": bool(conversation_id),
            "memory_context_used": bool(memory_context_used),
            "memory_context_mode": memory_context_mode,
            "document_scope": normalized_document_scope,
            "document_scope_enabled": bool(normalized_document_scope),
            "document_scope_mode": (
                "empty_response_document_scope_preserved"
                if normalized_document_scope
                else "none"
            ),
        }

    @with_request_id
    def answer_from_context(self, *, question: str, context: str,
                            request_id: Optional[str] = None) -> str:
        """Generate an answer from caller-supplied context only (no retrieval)."""
        rag = getattr(self.search_service, "rag_pipeline", None)
        if rag is None or getattr(rag, "answer_generator", None) is None:
            warning_id("[AIStewardManagerService] answer_from_context: no generator.", request_id)
            return ""
        try:
            result = rag.answer_generator.generate_answer(
                question=question, context=context or "", request_id=request_id,
            )
        except Exception as exc:
            warning_id(f"[AIStewardManagerService] answer_from_context failed: {exc}",
                       request_id, exc_info=True)
            return ""
        if isinstance(result, dict):
            return str(result.get("answer", "") or "")
        return result if isinstance(result, str) else ""