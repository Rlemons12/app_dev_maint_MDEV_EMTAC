# modules/services/unified_search_service.py
# Stateless RAG-first unified search domain service.

from __future__ import annotations

import inspect
import time
from typing import Dict, Any, Optional, List, Tuple

from modules.configuration.log_config import (
    logger,
    with_request_id,
    debug_id,
    warning_id,
)

from modules.ai.search_pathway.rag_core.rag_pipeline import get_default_rag
from modules.ai.search_pathway.rag_core.document_ui_payload import DocumentUIPayload
from modules.configuration import config
from modules.observability.high_end_tracer import tracer

from modules.emtacdb.emtacdb_fts import Document, CompleteDocument


class UnifiedSearchService:
    """
    Stateless RAG-first unified search domain service.

    Responsibilities:
        - Execute normal RAG searches.
        - Execute forced chunk debug searches.
        - Accept conversational memory metadata.
        - Accept document_scope metadata for document-scoped conversation mode.
        - Return answer-first results when include_payload=False.
        - Build supporting UI payload when include_payload=True or when
          build_payload_from_seed() is called by ChatPayloadOrchestrator.
        - Build relationship maps from chunks.
        - Project relationship data into UI-ready document payloads.
        - Promote nested document data into top-level containers.

    Does NOT:
        - Own DB session lifecycle.
        - Commit/rollback.
        - Format final chat blocks.
        - Persist Q&A history.

    Notes:
        ChatOrchestrator owns the answer transaction/session.
        ChatPayloadOrchestrator owns the payload transaction/session.
        AIStewardManagerService passes include_payload=False for answer-first mode.

    Document-scoped conversation mode:
        document_scope shape:
            {
                "enabled": true,
                "scope_type": "complete_document",
                "document_id": 29,
                "complete_document_id": 29,
                "document_name": "Document #29"
            }

        Preferred behavior:
            Pass document_scope into rag_pipeline.run() if supported so retrieval
            can filter by complete_document_id before selecting chunks.

        Safety fallback:
            If the lower RAG pipeline does not support document_scope yet, this
            service filters returned chunks to the selected complete_document_id
            and regenerates the answer only from those scoped chunks. This is not
            as good as true retriever-level filtering, but it prevents answering
            from unrelated documents.
    """

    def __init__(
        self,
        *,
        enable_rag: bool = True,
        enable_vector: bool = True,
        enable_fts: bool = True,
        enable_intent: bool = False,
    ):
        self.enable_vector = enable_vector
        self.enable_fts = enable_fts
        self.enable_intent = enable_intent

        self.rag_pipeline = None

        if enable_rag:
            try:
                logger.info("[UnifiedSearchService] Initializing RAG pipeline")
                self.rag_pipeline = get_default_rag()

                if not self.rag_pipeline:
                    raise RuntimeError("RAG pipeline initialization failed")

            except Exception as e:
                logger.error(
                    f"[UnifiedSearchService] RAG init failed: {e}",
                    exc_info=True,
                )
                self.rag_pipeline = None

    # ------------------------------------------------------------------
    # PUBLIC EXECUTION
    # ------------------------------------------------------------------

    @with_request_id
    def execute(
        self,
        *,
        session,
        question: str,
        user_id: str,
        request_id: Optional[str] = None,
        rag_only: bool = True,
        forced_chunk_id: Optional[int] = None,
        include_payload: bool = True,
        memory_context: Optional[str] = None,
        conversation_id: Optional[str] = None,
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main search entrypoint.

        include_payload=True:
            Legacy/full behavior.
            Returns answer plus documents/images/parts/drawings.

        include_payload=False:
            Answer-first behavior.
            Returns answer plus chunks/used_chunks only.

            Skips:
                - relationship_map build
                - ChunkRelationshipProjection
                - DocumentUIPayload aggregation
                - image/part/drawing promotion

            The payload route later calls:
                build_payload_from_seed()

        memory_context:
            Optional conversation memory text. This is passed to the RAG pipeline
            only when the pipeline supports it as a separate argument.

        conversation_id:
            Optional ChatSession.session_id metadata.

        document_scope:
            Optional document-scoped mode metadata. When active, retrieval/answering
            must be limited to the selected complete_document_id.

        Diagnostic additions:
            - Logs normalized request shape.
            - Logs RAG pipeline result type, keys, answer preview, and chunk counts.
            - Warns if the RAG pipeline returns a known placeholder failure answer.
            - Does not change answer behavior.
        """

        question = (question or "").strip()
        normalized_memory_context = (memory_context or "").strip()
        normalized_conversation_id = (
            str(conversation_id).strip()
            if conversation_id is not None and str(conversation_id).strip()
            else None
        )
        normalized_document_scope = self._normalize_document_scope(document_scope)

        debug_id(
            "[UnifiedSearchService] Execute start "
            f"request_id={request_id} "
            f"question_chars={len(question)} "
            f"user_id={user_id} "
            f"include_payload={include_payload} "
            f"rag_only={rag_only} "
            f"forced_chunk_id={forced_chunk_id} "
            f"memory_context_present={bool(normalized_memory_context)} "
            f"memory_context_chars={len(normalized_memory_context)} "
            f"conversation_id={normalized_conversation_id} "
            f"document_scope_enabled={bool(normalized_document_scope)} "
            f"complete_document_id="
            f"{normalized_document_scope.get('complete_document_id') if normalized_document_scope else None}",
            request_id,
        )

        if not question:
            return self._empty_response(
                strategy="invalid_input",
                answer="Please provide a valid question.",
                payload_status="unavailable",
                conversation_id=normalized_conversation_id,
                memory_context_used=bool(normalized_memory_context),
                memory_context_mode="not_used_invalid_input",
                document_scope=normalized_document_scope,
                document_scope_mode=(
                    "not_used_invalid_input"
                    if normalized_document_scope
                    else "none"
                ),
            )

        if not self.rag_pipeline:
            warning_id(
                "[UnifiedSearchService] RAG pipeline unavailable.",
                request_id,
            )

            return self._empty_response(
                strategy="rag_unavailable",
                answer="The AI assistant is temporarily unavailable.",
                payload_status="unavailable",
                conversation_id=normalized_conversation_id,
                memory_context_used=bool(normalized_memory_context),
                memory_context_mode="not_used_rag_unavailable",
                document_scope=normalized_document_scope,
                document_scope_mode=(
                    "not_used_rag_unavailable"
                    if normalized_document_scope
                    else "none"
                ),
            )

        effective_forced_chunk = (
            forced_chunk_id
            if forced_chunk_id is not None
            else (
                config.FORCE_DEBUG_CHUNK_ID
                if config.FORCE_DEBUG_CHUNK
                else None
            )
        )

        if effective_forced_chunk is not None:
            debug_id(
                "[UnifiedSearchService] Routing to forced chunk pipeline "
                f"request_id={request_id} "
                f"chunk_id={effective_forced_chunk}",
                request_id,
            )

            return self._run_forced_chunk_pipeline(
                session=session,
                question=question,
                chunk_id=effective_forced_chunk,
                request_id=request_id,
                include_payload=include_payload,
                memory_context=normalized_memory_context,
                conversation_id=normalized_conversation_id,
                document_scope=normalized_document_scope,
            )

        rag_start = time.perf_counter()

        with tracer.span(
            "rag_run",
            meta={
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
            rag_result, context_modes = self._run_rag_pipeline(
                question=question,
                request_id=request_id,
                memory_context=normalized_memory_context,
                conversation_id=normalized_conversation_id,
                document_scope=normalized_document_scope,
            )

        rag_time = time.perf_counter() - rag_start

        debug_id(
            "[UnifiedSearchService] RAG pipeline returned "
            f"request_id={request_id} "
            f"result_type={type(rag_result).__name__} "
            f"context_modes={context_modes} "
            f"rag_time={rag_time:.3f}s "
            f"memory_context_present={bool(normalized_memory_context)} "
            f"conversation_id={normalized_conversation_id} "
            f"document_scope_enabled={bool(normalized_document_scope)}",
            request_id,
        )

        if isinstance(rag_result, dict):
            rag_answer_preview = str(rag_result.get("answer") or "")
            rag_chunks = rag_result.get("chunks")
            rag_used_chunks = rag_result.get("used_chunks")

            rag_chunks_count = len(rag_chunks) if isinstance(rag_chunks, list) else 0
            rag_used_chunks_count = (
                len(rag_used_chunks) if isinstance(rag_used_chunks, list) else 0
            )

            debug_id(
                "[UnifiedSearchService] RAG result summary "
                f"request_id={request_id} "
                f"answer_preview={rag_answer_preview[:300]!r} "
                f"answer_chars={len(rag_answer_preview)} "
                f"chunks_count={rag_chunks_count} "
                f"used_chunks_count={rag_used_chunks_count} "
                f"retriever_top_k={rag_result.get('retriever_top_k')} "
                f"keys={sorted(str(key) for key in rag_result.keys())}",
                request_id,
            )

            normalized_answer_preview = rag_answer_preview.strip().lower()

            if normalized_answer_preview in {
                "error generating answer",
                "error generating answer.",
                "an unexpected error occurred while processing your request.",
            }:
                warning_id(
                    "[UnifiedSearchService] RAG pipeline returned a placeholder "
                    "failure answer. This means retrieval may have completed, but "
                    "answer generation likely failed inside rag_pipeline.run(). "
                    f"request_id={request_id} "
                    f"chunks_count={rag_chunks_count} "
                    f"used_chunks_count={rag_used_chunks_count} "
                    f"memory_context_present={bool(normalized_memory_context)} "
                    f"conversation_id={normalized_conversation_id} "
                    f"document_scope_enabled={bool(normalized_document_scope)}",
                    request_id,
                )

        if not isinstance(rag_result, dict):
            warning_id(
                "[UnifiedSearchService] RAG pipeline returned non-dict result.",
                request_id,
            )

            return self._empty_response(
                strategy="rag_invalid_result",
                answer="The AI assistant returned an invalid search result.",
                payload_status="unavailable",
                conversation_id=normalized_conversation_id,
                memory_context_used=bool(normalized_memory_context),
                memory_context_mode=context_modes.get("memory_context_mode", "none"),
                document_scope=normalized_document_scope,
                document_scope_mode=context_modes.get("document_scope_mode", "none"),
            )

        used_chunks = self._resolve_seed_chunks(rag_result)
        answer = rag_result.get("answer", "") or ""

        memory_context_mode = context_modes.get("memory_context_mode", "none")
        document_scope_mode = context_modes.get("document_scope_mode", "none")

        # --------------------------------------------------------------
        # Document-scope safety enforcement
        # --------------------------------------------------------------
        # Even if the RAG pipeline says it accepted document_scope, keep this
        # local filter as a safety check. If the selected document is active,
        # downstream chunks must not contain other complete_document_id values.
        if normalized_document_scope:
            scoped_chunks = self._filter_chunks_by_document_scope(
                chunks=used_chunks,
                document_scope=normalized_document_scope,
            )

            if not scoped_chunks:
                warning_id(
                    "[UnifiedSearchService] Document-scoped RAG found no chunks "
                    "from selected document. Returning scoped no-answer response. "
                    f"complete_document_id={normalized_document_scope.get('complete_document_id')} "
                    f"document_name={normalized_document_scope.get('document_name')}",
                    request_id,
                )

                return self._scoped_no_answer_response(
                    question=question,
                    document_scope=normalized_document_scope,
                    payload_status="pending",
                    rag_time=rag_time,
                    retriever_top_k=rag_result.get("retriever_top_k"),
                    query_embedding=rag_result.get("query_embedding", []),
                    memory_context_used=bool(normalized_memory_context),
                    memory_context_mode=memory_context_mode,
                    document_scope_mode=(
                        document_scope_mode
                        if document_scope_mode != "none"
                        else "post_filter_no_matching_chunks"
                    ),
                    conversation_id=normalized_conversation_id,
                )

            if len(scoped_chunks) != len(used_chunks):
                warning_id(
                    "[UnifiedSearchService] Document scope filtered out chunks "
                    "from other documents. Regenerating answer from scoped chunks only. "
                    f"before={len(used_chunks)} after={len(scoped_chunks)} "
                    f"complete_document_id={normalized_document_scope.get('complete_document_id')}",
                    request_id,
                )

                answer, generation_mode = self._generate_answer_from_scoped_chunks(
                    question=question,
                    chunks=scoped_chunks,
                    document_scope=normalized_document_scope,
                    request_id=request_id,
                )

                document_scope_mode = (
                    generation_mode
                    if document_scope_mode in {"none", "not_passed_rag_missing_support"}
                    else f"{document_scope_mode}+{generation_mode}"
                )

                used_chunks = scoped_chunks

            else:
                used_chunks = scoped_chunks

        debug_id(
            "[UnifiedSearchService] Normal RAG result "
            f"include_payload={include_payload} "
            f"used_chunks={len(used_chunks)} "
            f"answer_chars={len(str(answer or ''))} "
            f"rag_time={rag_time:.3f}s "
            f"conversation_id={normalized_conversation_id} "
            f"memory_context_mode={memory_context_mode} "
            f"document_scope_enabled={bool(normalized_document_scope)} "
            f"document_scope_mode={document_scope_mode} "
            f"complete_document_id="
            f"{normalized_document_scope.get('complete_document_id') if normalized_document_scope else None}",
            request_id,
        )

        if not include_payload:
            debug_id(
                "[UnifiedSearchService] Normal RAG answer-first mode: "
                "skipping relationship_map and UI payload projection.",
                request_id,
            )

            return {
                "strategy": "rag_document_scope" if normalized_document_scope else "rag",
                "method": "rag_document_scope" if normalized_document_scope else "rag",
                "answer": answer,
                "chunks": used_chunks,
                "used_chunks": used_chunks,
                "documents": [],
                "drawings": [],
                "images": [],
                "parts": [],
                "relationship_map": {},
                "payload_status": "pending",
                "retriever_top_k": rag_result.get("retriever_top_k"),
                "query_embedding": rag_result.get("query_embedding", []),
                "conversation_id": normalized_conversation_id,
                "memory_enabled": bool(normalized_conversation_id),
                "memory_context_used": bool(normalized_memory_context),
                "memory_context_mode": memory_context_mode,
                "document_scope": normalized_document_scope,
                "document_scope_enabled": bool(normalized_document_scope),
                "document_scope_mode": document_scope_mode,
                "payload_performance": {
                    "rag_time": rag_time,
                    "relationship_map_time": 0.0,
                    "projection_time": 0.0,
                    "fallback_document_time": 0.0,
                    "post_process_time": 0.0,
                    "payload_build_time": 0.0,
                    "fallback_documents_built": False,
                },
            }

        return self._build_full_payload_response(
            session=session,
            strategy="rag_document_scope" if normalized_document_scope else "rag",
            method="rag_document_scope" if normalized_document_scope else "rag",
            answer=answer,
            chunks=used_chunks,
            used_chunks=used_chunks,
            fallback_documents=rag_result.get("documents", []) or [],
            relationship_map=rag_result.get("relationship_map"),
            retriever_top_k=rag_result.get("retriever_top_k"),
            query_embedding=rag_result.get("query_embedding", []),
            request_id=request_id,
            debug_mode=False,
            debug_chunk_id=None,
            base_performance={
                "rag_time": rag_time,
            },
            conversation_id=normalized_conversation_id,
            memory_context_used=bool(normalized_memory_context),
            memory_context_mode=memory_context_mode,
            document_scope=normalized_document_scope,
            document_scope_mode=document_scope_mode,
        )

    # ------------------------------------------------------------------
    # RAG PIPELINE ADAPTER
    # ------------------------------------------------------------------

    def _run_rag_pipeline(
        self,
        *,
        question: str,
        request_id: Optional[str],
        memory_context: Optional[str],
        conversation_id: Optional[str],
        document_scope: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Calls rag_pipeline.run() with optional kwargs when supported.

        Preferred:
            rag_pipeline.run(
                question=...,
                request_id=...,
                memory_context=...,
                conversation_id=...,
                document_scope=...
            )

        Safe fallback:
            If rag_pipeline.run() does not accept document_scope, do not inject
            document details into the question. This service will filter and
            regenerate from scoped chunks after retrieval as a safety fallback.
        """

        run_kwargs: Dict[str, Any] = {
            "question": question,
            "request_id": request_id,
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

        support = self._get_callable_support(self.rag_pipeline.run)

        if normalized_memory_context and (
            support["accepts_kwargs"] or support["supports_memory_context"]
        ):
            run_kwargs["memory_context"] = normalized_memory_context
            memory_context_mode = "separate_memory_context"

        elif normalized_memory_context:
            memory_context_mode = "not_passed_rag_missing_support"
            warning_id(
                "[UnifiedSearchService] rag_pipeline.run() does not accept "
                "memory_context yet. Memory was NOT injected into the question.",
                request_id,
            )

        if normalized_conversation_id and (
            support["accepts_kwargs"] or support["supports_conversation_id"]
        ):
            run_kwargs["conversation_id"] = normalized_conversation_id

            if memory_context_mode == "none":
                memory_context_mode = "conversation_id_only"

        elif normalized_conversation_id:
            debug_id(
                "[UnifiedSearchService] rag_pipeline.run() does not accept "
                "conversation_id yet. Continuing without passing it downstream.",
                request_id,
            )

        if normalized_document_scope and (
            support["accepts_kwargs"] or support["supports_document_scope"]
        ):
            run_kwargs["document_scope"] = normalized_document_scope
            document_scope_mode = "separate_document_scope"

            debug_id(
                "[UnifiedSearchService] Passing document_scope to rag_pipeline.run "
                f"complete_document_id={normalized_document_scope.get('complete_document_id')} "
                f"document_name={normalized_document_scope.get('document_name')}",
                request_id,
            )

        elif normalized_document_scope:
            document_scope_mode = "not_passed_rag_missing_support"

            warning_id(
                "[UnifiedSearchService] rag_pipeline.run() does not accept "
                "document_scope yet. Retriever-level document filtering is not "
                "available in this layer yet. A scoped chunk safety filter will be "
                "applied after retrieval. "
                f"complete_document_id={normalized_document_scope.get('complete_document_id')}",
                request_id,
            )

        try:
            result = self.rag_pipeline.run(**run_kwargs)

            return result, {
                "memory_context_mode": memory_context_mode,
                "document_scope_mode": document_scope_mode,
            }

        except TypeError as type_error:
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
                "[UnifiedSearchService] rag_pipeline.run() rejected optional kwargs. "
                "Retrying without memory/conversation/document-scope kwargs. "
                "Memory and document scope were not injected into the question. "
                f"Error: {type_error}",
                request_id,
                exc_info=True,
            )

            if "memory_context" in run_kwargs:
                run_kwargs.pop("memory_context", None)
                memory_context_mode = "not_passed_after_type_error"

            if "conversation_id" in run_kwargs:
                run_kwargs.pop("conversation_id", None)

                if memory_context_mode == "conversation_id_only":
                    memory_context_mode = "none_after_type_error"

            if "document_scope" in run_kwargs:
                run_kwargs.pop("document_scope", None)
                document_scope_mode = "not_passed_after_type_error"

            result = self.rag_pipeline.run(**run_kwargs)

            return result, {
                "memory_context_mode": memory_context_mode,
                "document_scope_mode": document_scope_mode,
            }

    @staticmethod
    def _get_callable_support(callable_obj: Any) -> Dict[str, bool]:
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

    # ------------------------------------------------------------------
    # PAYLOAD-ONLY ENTRYPOINT
    # ------------------------------------------------------------------

    @with_request_id
    def build_payload_from_seed(
        self,
        *,
        session,
        result: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds supporting UI payload from a saved answer/RAG seed.

        Called by:
            ChatPayloadOrchestrator
                -> AIStewardManagerService.project_payload()
                    -> UnifiedSearchService.build_payload_from_seed()

        Expected seed fields:
            - used_chunks or chunks
            - answer
            - strategy/method
            - optional relationship_map
            - optional document_scope

        If relationship_map is missing, this method builds it now.

        Performance improvement:
            Fallback DocumentUIPayload aggregation is lazy. It is only built
            if projection is skipped or fails.
        """

        payload_start = time.perf_counter()

        if not isinstance(result, dict):
            warning_id(
                "[UnifiedSearchService] build_payload_from_seed received non-dict result.",
                request_id,
            )
            result = {}

        normalized_document_scope = self._normalize_document_scope(
            result.get("document_scope") or result.get("documentScope")
        )

        chunks = self._resolve_seed_chunks(result)

        if normalized_document_scope:
            before_count = len(chunks)
            chunks = self._filter_chunks_by_document_scope(
                chunks=chunks,
                document_scope=normalized_document_scope,
            )

            debug_id(
                "[UnifiedSearchService] Payload seed filtered by document_scope "
                f"before={before_count} after={len(chunks)} "
                f"complete_document_id={normalized_document_scope.get('complete_document_id')}",
                request_id,
            )

        if not chunks:
            warning_id(
                "[UnifiedSearchService] build_payload_from_seed skipped: no chunks in seed.",
                request_id,
            )

            result.setdefault("documents", [])
            result.setdefault("images", [])
            result.setdefault("parts", [])
            result.setdefault("drawings", [])
            result.setdefault("relationship_map", {})
            result["document_scope"] = normalized_document_scope
            result["document_scope_enabled"] = bool(normalized_document_scope)
            result["payload_status"] = "unavailable"
            result["payload_performance"] = {
                "relationship_map_time": 0.0,
                "projection_time": 0.0,
                "fallback_document_time": 0.0,
                "post_process_time": 0.0,
                "payload_build_time": time.perf_counter() - payload_start,
                "fallback_documents_built": False,
            }
            return result

        existing_documents = result.get("documents")
        fallback_documents = (
            existing_documents
            if isinstance(existing_documents, list) and existing_documents
            else None
        )

        relationship_map = result.get("relationship_map")
        relationship_map_time = 0.0

        if not isinstance(relationship_map, dict) or not relationship_map:
            relationship_start = time.perf_counter()

            relationship_map = self._build_relationship_map(
                session=session,
                chunks=chunks,
                request_id=request_id,
            )

            relationship_map_time = time.perf_counter() - relationship_start

        projection_start = time.perf_counter()

        documents, fallback_document_time, fallback_built = self._project_chunks_for_ui(
            session=session,
            chunks=chunks,
            relationship_map=relationship_map,
            fallback_documents=fallback_documents,
            request_id=request_id,
        )

        projection_time = time.perf_counter() - projection_start

        post_process_start = time.perf_counter()
        drawings, images, parts = self._post_process_documents(documents)
        post_process_time = time.perf_counter() - post_process_start

        payload_build_time = time.perf_counter() - payload_start

        result["documents"] = documents
        result["drawings"] = drawings
        result["images"] = images
        result["parts"] = parts
        result["relationship_map"] = relationship_map
        result["document_scope"] = normalized_document_scope
        result["document_scope_enabled"] = bool(normalized_document_scope)
        result["payload_status"] = "complete"
        result["payload_performance"] = {
            "relationship_map_time": relationship_map_time,
            "projection_time": projection_time,
            "fallback_document_time": fallback_document_time,
            "post_process_time": post_process_time,
            "payload_build_time": payload_build_time,
            "fallback_documents_built": fallback_built,
        }

        debug_id(
            "[UnifiedSearchService] Payload built from seed "
            f"documents={len(documents)} "
            f"images={len(images)} "
            f"parts={len(parts)} "
            f"drawings={len(drawings)} "
            f"relationship_map={'yes' if relationship_map else 'no'} "
            f"document_scope_enabled={bool(normalized_document_scope)} "
            f"complete_document_id="
            f"{normalized_document_scope.get('complete_document_id') if normalized_document_scope else None} "
            f"relationship_map_time={relationship_map_time:.3f}s "
            f"projection_time={projection_time:.3f}s "
            f"fallback_document_time={fallback_document_time:.3f}s "
            f"post_process_time={post_process_time:.3f}s "
            f"payload_build_time={payload_build_time:.3f}s "
            f"fallback_built={fallback_built}",
            request_id,
        )

        return result

    # ------------------------------------------------------------------
    # FORCED CHUNK PIPELINE
    # ------------------------------------------------------------------

    def _run_forced_chunk_pipeline(
        self,
        *,
        session,
        question: str,
        chunk_id: int,
        request_id: Optional[str] = None,
        include_payload: bool = True,
        memory_context: Optional[str] = None,
        conversation_id: Optional[str] = None,
        document_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the RAG answer generator against one forced Document chunk.

        In document-scoped mode:
            If the forced chunk does not belong to the selected complete_document_id,
            this method refuses to answer from it.
        """

        normalized_memory_context = (memory_context or "").strip()
        normalized_conversation_id = (
            str(conversation_id).strip()
            if conversation_id is not None and str(conversation_id).strip()
            else None
        )
        normalized_document_scope = self._normalize_document_scope(document_scope)

        debug_id(
            f"[UnifiedSearchService] Forced chunk pipeline start "
            f"chunk_id={chunk_id} include_payload={include_payload} "
            f"document_scope_enabled={bool(normalized_document_scope)}",
            request_id,
        )

        with tracer.span("forced_chunk_lookup", meta={"chunk_id": chunk_id}):
            doc = session.query(Document).filter(Document.id == chunk_id).first()

        if not doc:
            warning_id(
                f"[UnifiedSearchService] Forced chunk not found chunk_id={chunk_id}",
                request_id,
            )
            return self._empty_response(
                strategy="forced_chunk_not_found",
                answer=f"Chunk {chunk_id} not found.",
                payload_status="unavailable",
                conversation_id=normalized_conversation_id,
                memory_context_used=bool(normalized_memory_context),
                memory_context_mode="not_used_forced_chunk_not_found",
                document_scope=normalized_document_scope,
                document_scope_mode=(
                    "forced_chunk_not_found"
                    if normalized_document_scope
                    else "none"
                ),
            )

        complete_doc = None
        complete_document_id = getattr(doc, "complete_document_id", None)

        if normalized_document_scope:
            selected_complete_document_id = normalized_document_scope.get(
                "complete_document_id"
            )

            if self._safe_int(complete_document_id) != self._safe_int(
                selected_complete_document_id
            ):
                warning_id(
                    "[UnifiedSearchService] Forced chunk is outside active "
                    "document_scope. Returning scoped no-answer response. "
                    f"chunk_id={chunk_id} chunk_complete_document_id={complete_document_id} "
                    f"selected_complete_document_id={selected_complete_document_id}",
                    request_id,
                )

                return self._scoped_no_answer_response(
                    question=question,
                    document_scope=normalized_document_scope,
                    payload_status="pending",
                    rag_time=0.0,
                    retriever_top_k=1,
                    query_embedding=[],
                    memory_context_used=bool(normalized_memory_context),
                    memory_context_mode=(
                        "separate_memory_context"
                        if normalized_memory_context
                        else "none"
                    ),
                    document_scope_mode="forced_chunk_outside_scope",
                    conversation_id=normalized_conversation_id,
                    debug_mode=True,
                    debug_chunk_id=chunk_id,
                )

        if complete_document_id:
            complete_doc = (
                session.query(CompleteDocument)
                .filter(CompleteDocument.id == complete_document_id)
                .first()
            )

        complete_document_title = self._complete_document_title(
            complete_doc=complete_doc,
            fallback=f"Forced Chunk #{doc.id}",
        )

        forced_chunks = [
            {
                "document_id": doc.id,
                "chunk_id": doc.id,
                "id": doc.id,
                "content": doc.content or "",
                "text": doc.content or "",
                "complete_document_id": complete_document_id,
                "complete_document_title": complete_document_title,
                "document_title": complete_document_title,
                "title": complete_document_title,
                "file_path": (
                    getattr(complete_doc, "file_path", None)
                    if complete_doc
                    else getattr(doc, "file_path", None)
                ),
                "url": (
                    getattr(complete_doc, "url", None)
                    if complete_doc
                    else getattr(doc, "url", None)
                ),
                "distance": 0.0,
            }
        ]

        answer_start = time.perf_counter()

        answer, generation_mode = self._generate_answer_from_scoped_chunks(
            question=question,
            chunks=forced_chunks,
            document_scope=normalized_document_scope,
            request_id=request_id,
        )

        answer_time = time.perf_counter() - answer_start

        if not include_payload:
            debug_id(
                "[UnifiedSearchService] Forced chunk answer-first mode: "
                "skipping relationship_map and UI payload projection.",
                request_id,
            )

            return {
                "strategy": (
                    "forced_chunk_document_scope"
                    if normalized_document_scope
                    else "forced_chunk"
                ),
                "method": (
                    "forced_chunk_document_scope"
                    if normalized_document_scope
                    else "forced_chunk"
                ),
                "answer": answer,
                "chunks": forced_chunks,
                "used_chunks": forced_chunks,
                "documents": [],
                "drawings": [],
                "images": [],
                "parts": [],
                "relationship_map": {},
                "payload_status": "pending",
                "debug_mode": True,
                "debug_chunk_id": chunk_id,
                "retriever_top_k": 1,
                "query_embedding": [],
                "conversation_id": normalized_conversation_id,
                "memory_enabled": bool(normalized_conversation_id),
                "memory_context_used": bool(normalized_memory_context),
                "memory_context_mode": (
                    "separate_memory_context"
                    if normalized_memory_context
                    else "none"
                ),
                "document_scope": normalized_document_scope,
                "document_scope_enabled": bool(normalized_document_scope),
                "document_scope_mode": generation_mode,
                "payload_performance": {
                    "forced_answer_time": answer_time,
                    "relationship_map_time": 0.0,
                    "projection_time": 0.0,
                    "fallback_document_time": 0.0,
                    "post_process_time": 0.0,
                    "payload_build_time": 0.0,
                    "fallback_documents_built": False,
                },
            }

        return self._build_full_payload_response(
            session=session,
            strategy=(
                "forced_chunk_document_scope"
                if normalized_document_scope
                else "forced_chunk"
            ),
            method=(
                "forced_chunk_document_scope"
                if normalized_document_scope
                else "forced_chunk"
            ),
            answer=answer,
            chunks=forced_chunks,
            used_chunks=forced_chunks,
            fallback_documents=None,
            relationship_map=None,
            retriever_top_k=1,
            query_embedding=[],
            request_id=request_id,
            debug_mode=True,
            debug_chunk_id=chunk_id,
            base_performance={
                "forced_answer_time": answer_time,
            },
            conversation_id=normalized_conversation_id,
            memory_context_used=bool(normalized_memory_context),
            memory_context_mode=(
                "separate_memory_context"
                if normalized_memory_context
                else "none"
            ),
            document_scope=normalized_document_scope,
            document_scope_mode=generation_mode,
        )

    # ------------------------------------------------------------------
    # ANSWER GENERATION SAFETY FALLBACK
    # ------------------------------------------------------------------

    def _generate_answer_from_scoped_chunks(
        self,
        *,
        question: str,
        chunks: List[Dict[str, Any]],
        document_scope: Optional[Dict[str, Any]],
        request_id: Optional[str],
    ) -> Tuple[str, str]:
        """
        Generates an answer from the supplied chunks only.

        Used for:
            - Forced chunk path.
            - Document-scope safety fallback when rag_pipeline.run() did not
              enforce document_scope at retrieval time.

        If document_scope is active, the question is wrapped with strict
        document-mode instructions.
        """

        safe_chunks = chunks if isinstance(chunks, list) else []
        normalized_document_scope = self._normalize_document_scope(document_scope)

        if not safe_chunks:
            if normalized_document_scope:
                return (
                    self._selected_document_not_specified_answer(
                        normalized_document_scope
                    ),
                    "scoped_no_chunks",
                )

            return "", "no_chunks"

        try:
            with tracer.span(
                "document_scope_build_context",
                meta={
                    "chunks": len(safe_chunks),
                    "document_scope_enabled": bool(normalized_document_scope),
                    "complete_document_id": (
                        normalized_document_scope.get("complete_document_id")
                        if normalized_document_scope
                        else None
                    ),
                },
            ):
                ctx = self.rag_pipeline.context_builder.build_context(
                    retrieved_chunks=safe_chunks,
                    request_id=request_id,
                )

            if not isinstance(ctx, dict):
                warning_id(
                    "[UnifiedSearchService] Context builder returned non-dict "
                    "during scoped answer generation; using chunks directly.",
                    request_id,
                )
                context = "\n\n".join(
                    self._chunk_text(ch)
                    for ch in safe_chunks
                    if self._chunk_text(ch)
                )
            else:
                context = ctx.get("context") or "\n\n".join(
                    self._chunk_text(ch)
                    for ch in safe_chunks
                    if self._chunk_text(ch)
                )

            scoped_question = self._build_document_scope_prompt_question(
                question=question,
                document_scope=normalized_document_scope,
            )

            with tracer.span(
                "document_scope_generate_answer",
                meta={
                    "document_scope_enabled": bool(normalized_document_scope),
                    "complete_document_id": (
                        normalized_document_scope.get("complete_document_id")
                        if normalized_document_scope
                        else None
                    ),
                },
            ):
                answer_result = self.rag_pipeline.answer_generator.generate_answer(
                    question=scoped_question,
                    context=context,
                    request_id=request_id,
                )

            answer = self._extract_answer_text(answer_result).strip()

            if not answer and normalized_document_scope:
                answer = self._selected_document_not_specified_answer(
                    normalized_document_scope
                )

            return (
                answer,
                (
                    "scoped_answer_regenerated_from_filtered_chunks"
                    if normalized_document_scope
                    else "answer_generated_from_chunks"
                ),
            )

        except Exception as exc:
            warning_id(
                "[UnifiedSearchService] Scoped answer generation failed. "
                f"error={exc}",
                request_id,
            )

            if normalized_document_scope:
                return (
                    self._selected_document_not_specified_answer(
                        normalized_document_scope
                    ),
                    "scoped_generation_failed",
                )

            return "", "generation_failed"

    @staticmethod
    def _build_document_scope_prompt_question(
        *,
        question: str,
        document_scope: Optional[Dict[str, Any]],
    ) -> str:
        normalized_scope = UnifiedSearchService._normalize_document_scope(
            document_scope
        )

        if not normalized_scope:
            return question.strip()

        document_name = normalized_scope.get("document_name") or "the selected document"
        complete_document_id = normalized_scope.get("complete_document_id")

        return (
            "You are answering in Document Mode.\n"
            f"Selected document: {document_name} "
            f"(complete_document_id={complete_document_id}).\n\n"
            "Rules:\n"
            "1. Answer only from the selected document context provided.\n"
            "2. Do not use information from other documents, prior answers, or general knowledge.\n"
            "3. If the selected document does not specify the answer, say: "
            "\"The selected document does not specify that information.\"\n\n"
            f"User question:\n{question.strip()}"
        )

    @staticmethod
    def _selected_document_not_specified_answer(
        document_scope: Optional[Dict[str, Any]],
    ) -> str:
        normalized_scope = UnifiedSearchService._normalize_document_scope(
            document_scope
        )

        if not normalized_scope:
            return "The selected document does not specify that information."

        document_name = normalized_scope.get("document_name") or "the selected document"

        return (
            f"The selected document, {document_name}, does not specify that information."
        )

    def _scoped_no_answer_response(
        self,
        *,
        question: str,
        document_scope: Dict[str, Any],
        payload_status: str,
        rag_time: float,
        retriever_top_k: Optional[int],
        query_embedding: List[Any],
        memory_context_used: bool,
        memory_context_mode: str,
        document_scope_mode: str,
        conversation_id: Optional[str],
        debug_mode: bool = False,
        debug_chunk_id: Optional[int] = None,
    ) -> Dict[str, Any]:

        normalized_document_scope = self._normalize_document_scope(document_scope)

        return {
            "strategy": "rag_document_scope",
            "method": "rag_document_scope",
            "answer": self._selected_document_not_specified_answer(
                normalized_document_scope
            ),
            "chunks": [],
            "used_chunks": [],
            "documents": [],
            "drawings": [],
            "images": [],
            "parts": [],
            "relationship_map": {},
            "payload_status": payload_status,
            "debug_mode": debug_mode,
            "debug_chunk_id": debug_chunk_id,
            "retriever_top_k": retriever_top_k,
            "query_embedding": query_embedding or [],
            "conversation_id": conversation_id,
            "memory_enabled": bool(conversation_id),
            "memory_context_used": bool(memory_context_used),
            "memory_context_mode": memory_context_mode,
            "document_scope": normalized_document_scope,
            "document_scope_enabled": bool(normalized_document_scope),
            "document_scope_mode": document_scope_mode,
            "payload_performance": {
                "rag_time": rag_time,
                "relationship_map_time": 0.0,
                "projection_time": 0.0,
                "fallback_document_time": 0.0,
                "post_process_time": 0.0,
                "payload_build_time": 0.0,
                "fallback_documents_built": False,
            },
        }

    # ------------------------------------------------------------------
    # FULL PAYLOAD BUILDER
    # ------------------------------------------------------------------

    def _build_full_payload_response(
        self,
        *,
        session,
        strategy: str,
        method: str,
        answer: str,
        chunks: List[Dict[str, Any]],
        used_chunks: List[Dict[str, Any]],
        fallback_documents: Optional[List[Dict[str, Any]]],
        relationship_map: Optional[Dict[str, Any]],
        retriever_top_k: Optional[int],
        query_embedding: List[Any],
        request_id: Optional[str],
        debug_mode: bool = False,
        debug_chunk_id: Optional[int] = None,
        base_performance: Optional[Dict[str, float]] = None,
        conversation_id: Optional[str] = None,
        memory_context_used: bool = False,
        memory_context_mode: str = "none",
        document_scope: Optional[Dict[str, Any]] = None,
        document_scope_mode: str = "none",
    ) -> Dict[str, Any]:

        payload_start = time.perf_counter()
        normalized_document_scope = self._normalize_document_scope(document_scope)

        relationship_map_time = 0.0

        if not isinstance(relationship_map, dict) or not relationship_map:
            relationship_start = time.perf_counter()

            relationship_map = self._build_relationship_map(
                session=session,
                chunks=used_chunks,
                request_id=request_id,
            )

            relationship_map_time = time.perf_counter() - relationship_start

        projection_start = time.perf_counter()

        documents, fallback_document_time, fallback_built = self._project_chunks_for_ui(
            session=session,
            chunks=used_chunks,
            relationship_map=relationship_map,
            fallback_documents=fallback_documents,
            request_id=request_id,
        )

        projection_time = time.perf_counter() - projection_start

        post_process_start = time.perf_counter()
        drawings, images, parts = self._post_process_documents(documents)
        post_process_time = time.perf_counter() - post_process_start

        payload_build_time = time.perf_counter() - payload_start

        payload_performance = {
            "relationship_map_time": relationship_map_time,
            "projection_time": projection_time,
            "fallback_document_time": fallback_document_time,
            "post_process_time": post_process_time,
            "payload_build_time": payload_build_time,
            "fallback_documents_built": fallback_built,
        }

        if base_performance:
            payload_performance.update(base_performance)

        debug_id(
            "[UnifiedSearchService] Full payload built "
            f"strategy={strategy} "
            f"documents={len(documents)} "
            f"images={len(images)} "
            f"parts={len(parts)} "
            f"drawings={len(drawings)} "
            f"relationship_map={'yes' if relationship_map else 'no'} "
            f"document_scope_enabled={bool(normalized_document_scope)} "
            f"complete_document_id="
            f"{normalized_document_scope.get('complete_document_id') if normalized_document_scope else None} "
            f"relationship_map_time={relationship_map_time:.3f}s "
            f"projection_time={projection_time:.3f}s "
            f"fallback_document_time={fallback_document_time:.3f}s "
            f"post_process_time={post_process_time:.3f}s "
            f"payload_build_time={payload_build_time:.3f}s "
            f"fallback_built={fallback_built}",
            request_id,
        )

        return {
            "strategy": strategy,
            "method": method,
            "answer": answer,
            "chunks": chunks,
            "used_chunks": used_chunks,
            "documents": documents,
            "drawings": drawings,
            "images": images,
            "parts": parts,
            "relationship_map": relationship_map,
            "payload_status": "complete",
            "debug_mode": debug_mode,
            "debug_chunk_id": debug_chunk_id,
            "retriever_top_k": retriever_top_k,
            "query_embedding": query_embedding,
            "conversation_id": conversation_id,
            "memory_enabled": bool(conversation_id),
            "memory_context_used": bool(memory_context_used),
            "memory_context_mode": memory_context_mode,
            "document_scope": normalized_document_scope,
            "document_scope_enabled": bool(normalized_document_scope),
            "document_scope_mode": document_scope_mode,
            "payload_performance": payload_performance,
        }

    # ------------------------------------------------------------------
    # RELATIONSHIP HELPERS
    # ------------------------------------------------------------------

    def _build_relationship_map(
        self,
        *,
        session,
        chunks: List[Dict[str, Any]],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds the relationship map used by ChunkRelationshipProjection.

        Expected chunk keys:
            - chunk_id
            - document_id
            - complete_document_id
        """

        if not chunks:
            debug_id(
                "[UnifiedSearchService] No chunks supplied for relationship map.",
                request_id,
            )
            return {}

        chunk_ids, document_ids, complete_document_ids = self._extract_relationship_ids(
            chunks
        )

        if not chunk_ids and not document_ids and not complete_document_ids:
            debug_id(
                "[UnifiedSearchService] Relationship map skipped because no IDs "
                "could be extracted from chunks.",
                request_id,
            )
            return {}

        try:
            from modules.services.chunk_relationship_service import (
                ChunkRelationshipService,
            )

            relationship_service = ChunkRelationshipService()

            relationship_map = relationship_service.resolve(
                session=session,
                chunk_ids=chunk_ids,
                document_ids=document_ids,
                complete_document_ids=complete_document_ids,
                request_id=request_id,
            )

            if not isinstance(relationship_map, dict):
                warning_id(
                    "[UnifiedSearchService] ChunkRelationshipService returned "
                    "non-dict relationship map.",
                    request_id,
                )
                return {}

            summary = relationship_map.get("summary") or {}

            debug_id(
                "[UnifiedSearchService] Relationship map built "
                f"chunk_ids={len(chunk_ids)} "
                f"document_ids={len(document_ids)} "
                f"complete_document_ids={len(complete_document_ids)} "
                f"summary={summary}",
                request_id,
            )

            return relationship_map

        except Exception as e:
            warning_id(
                f"[UnifiedSearchService] Relationship resolution failed: {e}",
                request_id,
            )
            return {}

    def _project_chunks_for_ui(
        self,
        *,
        session,
        chunks: List[Dict[str, Any]],
        relationship_map: Dict[str, Any],
        fallback_documents: Optional[List[Dict[str, Any]]],
        request_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], float, bool]:
        """
        Applies ChunkRelationshipProjection and returns enriched documents.

        Returns:
            documents, fallback_document_time, fallback_documents_built

        If projection is unavailable or fails, returns fallback_documents.

        Performance improvement:
            fallback_documents are built lazily only when needed.
        """

        if not relationship_map:
            debug_id(
                "[UnifiedSearchService] Projection skipped: no relationship_map.",
                request_id,
            )
            documents, fallback_time, fallback_built = self._resolve_fallback_documents(
                fallback_documents=fallback_documents,
                chunks=chunks,
                request_id=request_id,
            )
            return documents, fallback_time, fallback_built

        if not chunks:
            debug_id(
                "[UnifiedSearchService] Projection skipped: no chunks.",
                request_id,
            )
            documents, fallback_time, fallback_built = self._resolve_fallback_documents(
                fallback_documents=fallback_documents,
                chunks=chunks,
                request_id=request_id,
            )
            return documents, fallback_time, fallback_built

        try:
            from modules.ai.search_pathway.rag_core.chunk_relationship_projection import (
                ChunkRelationshipProjection,
            )

            projection = ChunkRelationshipProjection(session=session)

            projected = projection.project_chunks_for_ui(
                chunks=chunks,
                relationship_map=relationship_map,
            )

            if not isinstance(projected, dict):
                warning_id(
                    "[UnifiedSearchService] Projection returned non-dict payload.",
                    request_id,
                )
                documents, fallback_time, fallback_built = self._resolve_fallback_documents(
                    fallback_documents=fallback_documents,
                    chunks=chunks,
                    request_id=request_id,
                )
                return documents, fallback_time, fallback_built

            documents = projected.get("documents-container")

            if not isinstance(documents, list):
                warning_id(
                    "[UnifiedSearchService] Projection did not return "
                    "documents-container list.",
                    request_id,
                )
                documents, fallback_time, fallback_built = self._resolve_fallback_documents(
                    fallback_documents=fallback_documents,
                    chunks=chunks,
                    request_id=request_id,
                )
                return documents, fallback_time, fallback_built

            debug_id(
                "[UnifiedSearchService] Projection applied "
                f"documents={len(documents)}",
                request_id,
            )

            return documents, 0.0, False

        except Exception as e:
            warning_id(
                f"[UnifiedSearchService] Relationship projection failed: {e}",
                request_id,
            )

            documents, fallback_time, fallback_built = self._resolve_fallback_documents(
                fallback_documents=fallback_documents,
                chunks=chunks,
                request_id=request_id,
            )
            return documents, fallback_time, fallback_built

    def _resolve_fallback_documents(
        self,
        *,
        fallback_documents: Optional[List[Dict[str, Any]]],
        chunks: List[Dict[str, Any]],
        request_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], float, bool]:
        """
        Returns fallback documents.

        If fallback_documents were already supplied, use them.
        Otherwise build them from chunks.

        Returns:
            fallback_documents, fallback_document_time, fallback_documents_built
        """

        if isinstance(fallback_documents, list) and fallback_documents:
            return fallback_documents, 0.0, False

        fallback_start = time.perf_counter()

        documents = (
            DocumentUIPayload()
            .aggregate_from_chunks(
                chunks,
                request_id=request_id,
            )
            .build()
        )

        fallback_time = time.perf_counter() - fallback_start

        if not isinstance(documents, list):
            documents = []

        debug_id(
            "[UnifiedSearchService] Fallback documents built "
            f"documents={len(documents)} "
            f"fallback_document_time={fallback_time:.3f}s",
            request_id,
        )

        return documents, fallback_time, True

    # ------------------------------------------------------------------
    # POST PROCESS HELPERS
    # ------------------------------------------------------------------

    def _post_process_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns:
            drawings, images, parts
        """

        safe_documents = documents if isinstance(documents, list) else []

        drawings = self._extract_drawings(safe_documents)
        images = self._promote_images(safe_documents)
        parts = self._promote_parts(safe_documents)

        return drawings, images, parts

    def _extract_drawings(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        drawings: List[Dict[str, Any]] = []
        seen = set()

        def add_drawing(drw: Any) -> None:
            if not isinstance(drw, dict):
                return

            key = self._item_key(
                item=drw,
                preferred_fields=(
                    "id",
                    "drawing_id",
                    "drw_number",
                    "drawing_number",
                ),
            )

            if key in seen:
                return

            seen.add(key)
            drawings.append(drw)

        for doc in documents or []:
            if not isinstance(doc, dict):
                continue

            direct_drawings = doc.get("drawings")
            if isinstance(direct_drawings, list):
                for drw in direct_drawings:
                    add_drawing(drw)

            nav = doc.get("drawing_navigation")
            if not isinstance(nav, dict):
                continue

            for area in nav.get("areas", []) or []:
                if not isinstance(area, dict):
                    continue

                for model in area.get("models", []) or []:
                    if not isinstance(model, dict):
                        continue

                    for asset in model.get("assets", []) or []:
                        if not isinstance(asset, dict):
                            continue

                        for drw in asset.get("drawings", []) or []:
                            add_drawing(drw)

        return drawings

    def _promote_images(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        images: List[Dict[str, Any]] = []
        seen = set()

        def add_image(img: Any) -> None:
            if not isinstance(img, dict):
                return

            key = self._item_key(
                item=img,
                preferred_fields=("id", "image_id", "src", "file_path"),
            )

            if key in seen:
                return

            seen.add(key)
            images.append(img)

        for doc in documents or []:
            if not isinstance(doc, dict):
                continue

            for field_name in ("images", "part_images"):
                nested = doc.get(field_name)

                if isinstance(nested, list):
                    for img in nested:
                        add_image(img)

            parts_panel = doc.get("parts_panel")
            if isinstance(parts_panel, dict):
                panel_images = parts_panel.get("images")
                if isinstance(panel_images, list):
                    for img in panel_images:
                        add_image(img)

        return images

    def _promote_parts(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        parts: List[Dict[str, Any]] = []
        seen = set()

        def add_part(part: Any) -> None:
            if not isinstance(part, dict):
                return

            key = self._item_key(
                item=part,
                preferred_fields=("id", "part_id", "part_number"),
            )

            if key in seen:
                return

            seen.add(key)
            parts.append(part)

        for doc in documents or []:
            if not isinstance(doc, dict):
                continue

            direct_parts = doc.get("parts")
            if isinstance(direct_parts, list):
                for part in direct_parts:
                    add_part(part)

            parts_panel = doc.get("parts_panel")
            if isinstance(parts_panel, dict):
                panel_parts = parts_panel.get("parts")
                if isinstance(panel_parts, list):
                    for part in panel_parts:
                        add_part(part)

        return parts

    # ------------------------------------------------------------------
    # GENERAL HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_seed_chunks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        used_chunks = result.get("used_chunks")

        if isinstance(used_chunks, list) and used_chunks:
            return used_chunks

        chunks = result.get("chunks")

        if isinstance(chunks, list) and chunks:
            return chunks

        return []

    @staticmethod
    def _filter_chunks_by_document_scope(
        *,
        chunks: List[Dict[str, Any]],
        document_scope: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized_scope = UnifiedSearchService._normalize_document_scope(
            document_scope
        )

        if not normalized_scope:
            return chunks if isinstance(chunks, list) else []

        selected_complete_document_id = UnifiedSearchService._safe_int(
            normalized_scope.get("complete_document_id")
        )

        if selected_complete_document_id is None:
            return []

        scoped_chunks: List[Dict[str, Any]] = []

        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue

            chunk_complete_document_id = UnifiedSearchService._extract_chunk_complete_document_id(
                chunk
            )

            if chunk_complete_document_id == selected_complete_document_id:
                scoped_chunks.append(chunk)

        return scoped_chunks

    @staticmethod
    def _extract_chunk_complete_document_id(chunk: Dict[str, Any]) -> Optional[int]:
        if not isinstance(chunk, dict):
            return None

        direct_value = (
            chunk.get("complete_document_id")
            or chunk.get("completed_document_id")
            or chunk.get("completeDocumentId")
            or chunk.get("completeDocumentID")
        )

        direct_int = UnifiedSearchService._safe_int(direct_value)

        if direct_int is not None:
            return direct_int

        document = chunk.get("document")
        if isinstance(document, dict):
            nested_value = (
                document.get("complete_document_id")
                or document.get("completed_document_id")
                or document.get("completeDocumentId")
                or document.get("completeDocumentID")
            )

            nested_int = UnifiedSearchService._safe_int(nested_value)

            if nested_int is not None:
                return nested_int

        complete_document = (
            chunk.get("complete_document")
            or chunk.get("completed_document")
            or chunk.get("completeDocument")
        )

        if isinstance(complete_document, dict):
            complete_document_id = (
                complete_document.get("id")
                or complete_document.get("complete_document_id")
                or complete_document.get("completeDocumentId")
            )

            return UnifiedSearchService._safe_int(complete_document_id)

        return None

    @staticmethod
    def _chunk_text(chunk: Dict[str, Any]) -> str:
        if not isinstance(chunk, dict):
            return ""

        return str(
            chunk.get("content")
            or chunk.get("text")
            or chunk.get("chunk_text")
            or chunk.get("page_content")
            or ""
        ).strip()

    @staticmethod
    def _extract_answer_text(answer_result: Any) -> str:
        """
        Normalizes answer generator output.

        Supports:
            - {"answer": "..."}
            - "..."
            - None
        """

        if isinstance(answer_result, dict):
            return str(answer_result.get("answer", "") or "")

        if isinstance(answer_result, str):
            return answer_result

        return ""

    @classmethod
    def _extract_relationship_ids(
        cls,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Extracts chunk_ids, document_ids, and complete_document_ids from chunks.
        """

        chunk_ids: List[int] = []
        document_ids: List[int] = []
        complete_document_ids: List[int] = []

        for ch in chunks or []:
            if not isinstance(ch, dict):
                continue

            chunk_id = ch.get("chunk_id") or ch.get("id")
            document_id = ch.get("document_id") or ch.get("chunk_document_id")
            complete_document_id = ch.get("complete_document_id")

            normalized_chunk_id = cls._safe_int(chunk_id)
            normalized_document_id = cls._safe_int(document_id)
            normalized_complete_document_id = cls._safe_int(complete_document_id)

            if normalized_chunk_id is not None:
                chunk_ids.append(normalized_chunk_id)

            if normalized_document_id is not None:
                document_ids.append(normalized_document_id)

            if normalized_complete_document_id is not None:
                complete_document_ids.append(normalized_complete_document_id)

        return (
            cls._dedupe_preserve_order(chunk_ids),
            cls._dedupe_preserve_order(document_ids),
            cls._dedupe_preserve_order(complete_document_ids),
        )

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

        complete_document_id = UnifiedSearchService._safe_int(complete_document_id)

        if complete_document_id is None:
            return None

        document_id = (
            document_scope.get("document_id")
            or document_scope.get("documentId")
        )

        document_id = UnifiedSearchService._safe_int(document_id)

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
    def _complete_document_title(
        *,
        complete_doc: Optional[Any],
        fallback: str,
    ) -> str:
        if complete_doc is None:
            return fallback

        for attr_name in (
            "title",
            "name",
            "document_name",
            "file_name",
            "filename",
        ):
            value = getattr(complete_doc, attr_name, None)

            if value:
                return str(value)

        file_path = getattr(complete_doc, "file_path", None)

        if file_path:
            return str(file_path).replace("\\", "/").split("/")[-1] or fallback

        return fallback

    @staticmethod
    def _dedupe_preserve_order(items: List[Any]) -> List[Any]:
        seen = set()
        output = []

        for item in items or []:
            if item in seen:
                continue

            seen.add(item)
            output.append(item)

        return output

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value in (None, "", "None"):
            return None

        if isinstance(value, bool):
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _item_key(
        *,
        item: Dict[str, Any],
        preferred_fields: Tuple[str, ...],
    ) -> Tuple[str, Any]:
        for field_name in preferred_fields:
            value = item.get(field_name)

            if value not in (None, "", "None"):
                return field_name, value

        return "repr", repr(sorted(item.items()))

    def _empty_response(
        self,
        *,
        strategy: str,
        answer: str,
        payload_status: str = "unavailable",
        conversation_id: Optional[str] = None,
        memory_context_used: bool = False,
        memory_context_mode: str = "none",
        document_scope: Optional[Dict[str, Any]] = None,
        document_scope_mode: str = "none",
    ) -> Dict[str, Any]:

        normalized_document_scope = self._normalize_document_scope(document_scope)

        return {
            "strategy": strategy,
            "method": strategy,
            "answer": answer,
            "chunks": [],
            "used_chunks": [],
            "documents": [],
            "drawings": [],
            "images": [],
            "parts": [],
            "relationship_map": {},
            "payload_status": payload_status,
            "retriever_top_k": None,
            "query_embedding": [],
            "conversation_id": conversation_id,
            "memory_enabled": bool(conversation_id),
            "memory_context_used": bool(memory_context_used),
            "memory_context_mode": memory_context_mode,
            "document_scope": normalized_document_scope,
            "document_scope_enabled": bool(normalized_document_scope),
            "document_scope_mode": document_scope_mode,
            "payload_performance": {
                "relationship_map_time": 0.0,
                "projection_time": 0.0,
                "fallback_document_time": 0.0,
                "post_process_time": 0.0,
                "payload_build_time": 0.0,
                "fallback_documents_built": False,
            },
        }