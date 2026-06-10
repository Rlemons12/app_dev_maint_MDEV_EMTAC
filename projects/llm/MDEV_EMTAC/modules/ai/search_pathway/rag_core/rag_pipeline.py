from __future__ import annotations

import inspect
from typing import Dict, Any, Optional, List, Tuple

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id,
    error_id,
    debug_id,
    warning_id,
    with_request_id,
)

from .embedder import DBConfiguredEmbedder, BaseEmbedder
from .retriever import PgVectorRetriever
from .context_builder import ContextBuilder
from .answer_generator import DBConfiguredAnswerGenerator, BaseAnswerGenerator
from .document_ui_payload import DocumentUIPayload

from modules.observability.high_end_tracer import tracer
from modules.services.complete_document_profile_service import (
    CompleteDocumentProfileService,
)


class RAGPipeline:
    DOCUMENT_SCOPE_FALLBACK_TOP_K = 8

    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        embedder: Optional[BaseEmbedder] = None,
        retriever: Optional[PgVectorRetriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        answer_generator: Optional[BaseAnswerGenerator] = None,
    ):
        self.db_config = db_config or DatabaseConfig()
        self.embedder = embedder or DBConfiguredEmbedder()
        self.retriever = retriever or PgVectorRetriever(db_config=self.db_config)
        self.context_builder = context_builder or ContextBuilder()
        self.answer_generator = answer_generator or DBConfiguredAnswerGenerator()
        self.document_profile_service = CompleteDocumentProfileService()

    @with_request_id
    def run(
            self,
            question: str,
            top_k: int = 5,
            request_id: Optional[str] = None,
            memory_context: Optional[str] = None,
            conversation_id: Optional[str] = None,
            document_scope: Optional[Dict[str, Any]] = None,
            **answer_kwargs: Any,
    ) -> Dict[str, Any]:

        normalized_question = (question or "").strip()
        normalized_memory_context = (memory_context or "").strip()
        normalized_conversation_id = (
            str(conversation_id).strip()
            if conversation_id is not None and str(conversation_id).strip()
            else None
        )
        normalized_document_scope = self._normalize_document_scope(document_scope)

        debug_id(
            f"[RAGPipeline] Start RAG: '{normalized_question[:80]}...' "
            f"document_scope_enabled={bool(normalized_document_scope)} "
            f"complete_document_id="
            f"{normalized_document_scope.get('complete_document_id') if normalized_document_scope else None}",
            request_id,
        )

        try:
            if not normalized_question:
                return self._basic_result(
                    answer="Please provide a valid question.",
                    top_k=top_k,
                    conversation_id=normalized_conversation_id,
                    memory_context=normalized_memory_context,
                    document_scope=normalized_document_scope,
                    document_scope_mode=(
                        "not_used_invalid_input"
                        if normalized_document_scope
                        else "none"
                    ),
                )

            document_profile = self._load_document_profile(
                document_scope=normalized_document_scope,
                request_id=request_id,
            )

            if (
                    normalized_document_scope
                    and isinstance(document_profile, dict)
                    and document_profile.get("found")
                    and self.document_profile_service.is_overview_question(
                normalized_question
            )
            ):
                debug_id(
                    "[RAGPipeline] Answering document overview question from document profile "
                    f"complete_document_id={document_profile.get('complete_document_id')} "
                    f"title={document_profile.get('title')!r}",
                    request_id,
                )

                return self._document_profile_overview_result(
                    question=normalized_question,
                    top_k=top_k,
                    memory_context=normalized_memory_context,
                    conversation_id=normalized_conversation_id,
                    document_scope=normalized_document_scope,
                    document_profile=document_profile,
                )

            with tracer.span(
                    "embed_query",
                    meta={
                        "document_scope_enabled": bool(normalized_document_scope),
                        "complete_document_id": (
                                normalized_document_scope.get("complete_document_id")
                                if normalized_document_scope
                                else None
                        ),
                    },
            ):
                query_embedding = self.embedder.embed_query(
                    normalized_question,
                    request_id=request_id,
                )

            if not query_embedding:
                warning_id(
                    "[RAGPipeline] No embedding produced — stopping RAG.",
                    request_id,
                )
                return self._basic_result(
                    answer="Embedding model failed or returned nothing.",
                    top_k=top_k,
                    conversation_id=normalized_conversation_id,
                    memory_context=normalized_memory_context,
                    document_scope=normalized_document_scope,
                    document_scope_mode=(
                        "not_used_embedding_failed"
                        if normalized_document_scope
                        else "none"
                    ),
                )

            retrieval_top_k = self._resolve_retrieval_top_k(
                top_k=top_k,
                document_scope=normalized_document_scope,
            )

            with tracer.span(
                    "retrieve_chunks",
                    meta={
                        "top_k": top_k,
                        "retrieval_top_k": retrieval_top_k,
                        "document_scope_enabled": bool(normalized_document_scope),
                        "complete_document_id": (
                                normalized_document_scope.get("complete_document_id")
                                if normalized_document_scope
                                else None
                        ),
                    },
            ):
                retrieved_chunks, retriever_scope_mode = self._retrieve_chunks(
                    query_embedding=query_embedding,
                    top_k=retrieval_top_k,
                    request_id=request_id,
                    document_scope=normalized_document_scope,
                )

            if not retrieved_chunks:
                warning_id(
                    "[RAGPipeline] No relevant documents retrieved after retrieval quality gate.",
                    request_id,
                )

                return self._no_relevant_context_result(
                    question=normalized_question,
                    top_k=top_k,
                    query_embedding=query_embedding,
                    memory_context=normalized_memory_context,
                    conversation_id=normalized_conversation_id,
                    document_scope=normalized_document_scope,
                    document_scope_mode=(
                        "no_relevant_chunks_after_retrieval"
                        if not normalized_document_scope
                        else "no_relevant_chunks_after_retrieval_document_scope"
                    ),
                )

            document_scope_mode = retriever_scope_mode

            if normalized_document_scope:
                before_filter_count = len(retrieved_chunks or [])

                retrieved_chunks = self._filter_chunks_by_document_scope(
                    chunks=retrieved_chunks,
                    document_scope=normalized_document_scope,
                )

                after_filter_count = len(retrieved_chunks or [])

                if after_filter_count != before_filter_count:
                    warning_id(
                        "[RAGPipeline] Document scope local filter changed retrieved chunks "
                        f"before={before_filter_count} after={after_filter_count} "
                        f"complete_document_id={normalized_document_scope.get('complete_document_id')}",
                        request_id,
                    )

                    if document_scope_mode == "none":
                        document_scope_mode = "local_post_retrieval_filter"
                    elif "local_post_retrieval_filter" not in document_scope_mode:
                        document_scope_mode = (
                            f"{document_scope_mode}+local_post_retrieval_filter"
                        )

                if not retrieved_chunks:
                    return self._document_scope_no_answer_result(
                        question=normalized_question,
                        top_k=top_k,
                        query_embedding=query_embedding,
                        memory_context=normalized_memory_context,
                        conversation_id=normalized_conversation_id,
                        document_scope=normalized_document_scope,
                        document_scope_mode=(
                            document_scope_mode
                            if document_scope_mode != "none"
                            else "no_matching_chunks_after_scope_filter"
                        ),
                    )

            with tracer.span(
                    "build_context",
                    meta={
                        "retrieved_chunks": len(retrieved_chunks or []),
                        "document_scope_enabled": bool(normalized_document_scope),
                    },
            ):
                ctx = self.context_builder.build_context(
                    retrieved_chunks=retrieved_chunks,
                    request_id=request_id,
                    document_scope=normalized_document_scope,
                )

            if not isinstance(ctx, dict):
                warning_id(
                    "[RAGPipeline] Context builder returned non-dict result.",
                    request_id,
                )
                ctx = {}

            context: str = ctx.get("context", "") or self._chunks_to_context(
                retrieved_chunks
            )
            used_chunks: List[Dict[str, Any]] = (
                    ctx.get("used_chunks", []) or retrieved_chunks
            )

            if normalized_document_scope:
                used_chunks = self._filter_chunks_by_document_scope(
                    chunks=used_chunks,
                    document_scope=normalized_document_scope,
                )

                if not used_chunks:
                    return self._document_scope_no_answer_result(
                        question=normalized_question,
                        top_k=top_k,
                        query_embedding=query_embedding,
                        memory_context=normalized_memory_context,
                        conversation_id=normalized_conversation_id,
                        document_scope=normalized_document_scope,
                        document_scope_mode="no_used_chunks_after_scope_filter",
                    )

                context = self._chunks_to_context(used_chunks) or context

            with tracer.span(
                    "build_document_payload",
                    meta={
                        "used_chunks": len(used_chunks or []),
                        "document_scope_enabled": bool(normalized_document_scope),
                    },
            ):
                documents = (
                    DocumentUIPayload()
                    .aggregate_from_chunks(used_chunks)
                    .build()
                )

            if not isinstance(documents, list):
                documents = []

            debug_id(
                f"[RAGPipeline] Built UI payload with {len(documents)} documents "
                f"from {len(used_chunks)} chunks "
                f"document_scope_enabled={bool(normalized_document_scope)}",
                request_id,
            )

            prompt_context = self._build_prompt_context(
                context=context,
                memory_context=normalized_memory_context,
                document_scope=normalized_document_scope,
                document_profile=document_profile,
            )

            answer_kwargs = self._clean_answer_kwargs(answer_kwargs)

            with tracer.span(
                    "generate_answer",
                    meta={
                        "document_scope_enabled": bool(normalized_document_scope),
                        "complete_document_id": (
                                normalized_document_scope.get("complete_document_id")
                                if normalized_document_scope
                                else None
                        ),
                        "memory_context_present": bool(normalized_memory_context),
                        "document_profile_present": bool(
                            isinstance(document_profile, dict)
                            and document_profile.get("found")
                        ),
                    },
            ):
                answer_result = self.answer_generator.generate_answer(
                    question=normalized_question,
                    context=prompt_context,
                    request_id=request_id,
                    **answer_kwargs,
                )

            answer: str = self._extract_answer_text(answer_result)

            answer = self._clean_document_mode_answer_leakage(
                answer=answer,
                document_scope=normalized_document_scope,
            )

            if normalized_document_scope and not answer.strip():
                answer = self._selected_document_not_specified_answer(
                    normalized_document_scope
                )

            return {
                "answer": answer,
                "documents": documents,
                "used_chunks": used_chunks,
                "chunks": used_chunks,
                "query_embedding": query_embedding,
                "retriever_top_k": top_k,
                "conversation_id": normalized_conversation_id,
                "memory_context_used": bool(normalized_memory_context),
                "memory_context_mode": (
                    "separate_memory_context"
                    if normalized_memory_context
                    else "none"
                ),
                "document_scope": normalized_document_scope,
                "document_scope_enabled": bool(normalized_document_scope),
                "document_scope_mode": document_scope_mode,
                "document_profile_used": bool(
                    isinstance(document_profile, dict)
                    and document_profile.get("found")
                ),
            }

        except Exception as e:
            error_id(f"[RAGPipeline] Pipeline failed: {e}", request_id, exc_info=True)
            raise

    def _load_document_profile(
        self,
        *,
        document_scope: Optional[Dict[str, Any]],
        request_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        normalized_scope = self._normalize_document_scope(document_scope)

        if not normalized_scope:
            return None

        complete_document_id = normalized_scope.get("complete_document_id")

        try:
            with self.db_config.main_session() as session:
                profile = self.document_profile_service.get_profile(
                    session=session,
                    complete_document_id=complete_document_id,
                    request_id=request_id,
                )

            if isinstance(profile, dict) and profile.get("found"):
                debug_id(
                    "[RAGPipeline] Loaded document profile "
                    f"complete_document_id={complete_document_id} "
                    f"has_summary={profile.get('has_summary')} "
                    f"has_profile_signals={profile.get('has_profile_signals')}",
                    request_id,
                )
                return profile

            warning_id(
                "[RAGPipeline] Document profile unavailable "
                f"complete_document_id={complete_document_id}",
                request_id,
            )
            return profile if isinstance(profile, dict) else None

        except Exception as exc:
            warning_id(
                "[RAGPipeline] Failed to load document profile "
                f"complete_document_id={complete_document_id}: {exc}",
                request_id,
                exc_info=True,
            )
            return None

    def _document_profile_overview_result(
        self,
        *,
        question: str,
        top_k: int,
        memory_context: str,
        conversation_id: Optional[str],
        document_scope: Dict[str, Any],
        document_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized_scope = self._normalize_document_scope(document_scope)

        if isinstance(document_profile, dict) and document_profile.get("found"):
            answer = self.document_profile_service.build_overview_answer_from_profile(
                document_profile
            )
            document_scope_mode = "document_profile_overview"
        else:
            answer = self._selected_document_not_specified_answer(normalized_scope)
            document_scope_mode = "document_profile_missing"

        return {
            "answer": answer,
            "documents": [],
            "used_chunks": [],
            "chunks": [],
            "query_embedding": [],
            "retriever_top_k": top_k,
            "conversation_id": conversation_id,
            "memory_context_used": bool((memory_context or "").strip()),
            "memory_context_mode": (
                "separate_memory_context"
                if (memory_context or "").strip()
                else "none"
            ),
            "document_scope": normalized_scope,
            "document_scope_enabled": bool(normalized_scope),
            "document_scope_mode": document_scope_mode,
            "document_profile_used": bool(
                isinstance(document_profile, dict)
                and document_profile.get("found")
            ),
        }

    def _retrieve_chunks(
        self,
        *,
        query_embedding: List[float],
        top_k: int,
        request_id: Optional[str],
        document_scope: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:

        normalized_scope = self._normalize_document_scope(document_scope)

        retrieve_kwargs: Dict[str, Any] = {
            "query_embedding": query_embedding,
            "top_k": top_k,
            "request_id": request_id,
        }

        scope_mode = "none"

        if normalized_scope:
            support = self._get_callable_support(self.retriever.retrieve)
            complete_document_id = normalized_scope.get("complete_document_id")

            if support["accepts_kwargs"] or support["supports_document_scope"]:
                retrieve_kwargs["document_scope"] = normalized_scope
                scope_mode = "retriever_document_scope"

                debug_id(
                    "[RAGPipeline] Passing document_scope to retriever "
                    f"complete_document_id={complete_document_id}",
                    request_id,
                )

            elif support["supports_complete_document_id"]:
                retrieve_kwargs["complete_document_id"] = complete_document_id
                scope_mode = "retriever_complete_document_id"

                debug_id(
                    "[RAGPipeline] Passing complete_document_id to retriever "
                    f"complete_document_id={complete_document_id}",
                    request_id,
                )

            else:
                scope_mode = "retriever_missing_scope_support"

                warning_id(
                    "[RAGPipeline] Retriever does not accept document_scope or "
                    "complete_document_id yet. Retrieving expanded candidate set and "
                    "applying local post-retrieval filter. "
                    f"complete_document_id={complete_document_id}",
                    request_id,
                )

        try:
            chunks = self.retriever.retrieve(**retrieve_kwargs)

            if not isinstance(chunks, list):
                warning_id(
                    "[RAGPipeline] Retriever returned non-list chunks.",
                    request_id,
                )
                return [], scope_mode

            return chunks, scope_mode

        except TypeError as type_error:
            type_error_text = str(type_error)

            rejected_scope_kwargs = (
                "unexpected keyword argument" in type_error_text
                and (
                    "document_scope" in type_error_text
                    or "complete_document_id" in type_error_text
                )
            )

            if not rejected_scope_kwargs:
                raise

            warning_id(
                "[RAGPipeline] Retriever rejected document scope kwargs. "
                "Retrying without retriever-level scope support. "
                f"Error: {type_error}",
                request_id,
                exc_info=True,
            )

            retrieve_kwargs.pop("document_scope", None)
            retrieve_kwargs.pop("complete_document_id", None)

            chunks = self.retriever.retrieve(**retrieve_kwargs)

            if not isinstance(chunks, list):
                return [], "retriever_scope_type_error_non_list"

            return chunks, "retriever_scope_type_error_local_filter"

    @staticmethod
    def _get_callable_support(callable_obj: Any) -> Dict[str, bool]:
        support = {
            "accepts_kwargs": False,
            "supports_memory_context": False,
            "supports_conversation_id": False,
            "supports_document_scope": False,
            "supports_complete_document_id": False,
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
        support["supports_complete_document_id"] = "complete_document_id" in parameters

        return support

    @staticmethod
    def _build_prompt_context(
        *,
        context: str,
        memory_context: str,
        document_scope: Optional[Dict[str, Any]],
        document_profile: Optional[Dict[str, Any]] = None,
    ) -> str:

        normalized_scope = RAGPipeline._normalize_document_scope(document_scope)
        memory_context = (memory_context or "").strip()
        context = (context or "").strip()

        sections: List[str] = []

        if normalized_scope:
            document_name = normalized_scope.get("document_name") or "the selected document"
            complete_document_id = normalized_scope.get("complete_document_id")

            sections.append(
                "DOCUMENT MODE ACTIVE\n"
                f"Selected document: {document_name} "
                f"(complete_document_id={complete_document_id}).\n\n"
                "Document Mode rules:\n"
                "- Answer only from the selected document chunk context below.\n"
                "- Do not use other manuals, prior answers, conversation summaries, or general knowledge as evidence.\n"
                "- Conversation memory may only help understand references like this, it, that, those, or same issue.\n"
                "- If the selected document chunk context does not specify the answer, say: "
                "\"The selected document does not specify that information.\"\n"
                "- Do not repeat these Document Mode rules in the final answer."
            )

        elif memory_context:
            sections.append(
                "Conversation memory is provided only to understand context, references, "
                "and follow-up wording. The current user question remains the main request. "
                "Do not let memory override retrieved database/document evidence."
            )

        if memory_context:
            if normalized_scope:
                sections.append(
                    "Conversation memory for reference resolution only:\n"
                    f"{memory_context}"
                )
            else:
                sections.append(f"Conversation memory:\n{memory_context}")

        if normalized_scope:
            sections.append(f"Selected document chunk context:\n{context}")
        else:
            sections.append(f"Retrieved context:\n{context}")

        return "\n\n".join(section for section in sections if section).strip()

    @staticmethod
    def _clean_document_mode_answer_leakage(
        *,
        answer: str,
        document_scope: Optional[Dict[str, Any]],
    ) -> str:

        text = str(answer or "").strip()

        if not text or not RAGPipeline._normalize_document_scope(document_scope):
            return text

        for marker in ["FINAL ANSWER:", "Final answer:", "Answer:"]:
            if marker in text:
                candidate = text.split(marker, 1)[-1].strip()
                if candidate:
                    text = candidate

        leading_markers = [
            "You are answering in Document Mode.",
            "DOCUMENT MODE ACTIVE",
            "Document Mode rules:",
            "Selected document:",
        ]

        if any(text.startswith(marker) for marker in leading_markers):
            lines = text.splitlines()
            cleaned_lines: List[str] = []
            skipping_prompt_block = True

            for line in lines:
                stripped = line.strip()

                if not stripped:
                    continue

                if skipping_prompt_block:
                    promptish = (
                        stripped.startswith("You are answering in Document Mode")
                        or stripped.startswith("DOCUMENT MODE ACTIVE")
                        or stripped.startswith("Selected document:")
                        or stripped.startswith("Document Mode rules:")
                        or stripped.startswith("- Answer only")
                        or stripped.startswith("- Do not use")
                        or stripped.startswith("- The document profile")
                        or stripped.startswith("- Conversation memory")
                        or stripped.startswith("- If the selected document")
                        or stripped.startswith("- Do not repeat")
                        or stripped.startswith("Current user question:")
                        or stripped.startswith("User question:")
                    )

                    if promptish:
                        continue

                    skipping_prompt_block = False
                    cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)

            cleaned = "\n".join(cleaned_lines).strip()

            if cleaned:
                text = cleaned

        for marker in ["Current user question:", "User question:", "QUESTION:"]:
            if text.startswith(marker):
                parts = text.splitlines()
                text = "\n".join(parts[1:]).strip() if len(parts) > 1 else ""

        return text.strip()

    @staticmethod
    def _document_scope_no_answer_result(
        *,
        question: str,
        top_k: int,
        query_embedding: List[Any],
        memory_context: str,
        conversation_id: Optional[str],
        document_scope: Dict[str, Any],
        document_scope_mode: str,
    ) -> Dict[str, Any]:

        normalized_scope = RAGPipeline._normalize_document_scope(document_scope)

        return {
            "answer": RAGPipeline._selected_document_not_specified_answer(
                normalized_scope
            ),
            "documents": [],
            "used_chunks": [],
            "chunks": [],
            "query_embedding": query_embedding or [],
            "retriever_top_k": top_k,
            "conversation_id": conversation_id,
            "memory_context_used": bool((memory_context or "").strip()),
            "memory_context_mode": (
                "separate_memory_context"
                if (memory_context or "").strip()
                else "none"
            ),
            "document_scope": normalized_scope,
            "document_scope_enabled": bool(normalized_scope),
            "document_scope_mode": document_scope_mode,
        }

    @staticmethod
    def _basic_result(
        *,
        answer: str,
        top_k: int,
        conversation_id: Optional[str],
        memory_context: str,
        document_scope: Optional[Dict[str, Any]],
        document_scope_mode: str,
    ) -> Dict[str, Any]:

        normalized_scope = RAGPipeline._normalize_document_scope(document_scope)

        return {
            "answer": answer,
            "documents": [],
            "used_chunks": [],
            "chunks": [],
            "query_embedding": [],
            "retriever_top_k": top_k,
            "conversation_id": conversation_id,
            "memory_context_used": bool((memory_context or "").strip()),
            "memory_context_mode": (
                "separate_memory_context"
                if (memory_context or "").strip()
                else "none"
            ),
            "document_scope": normalized_scope,
            "document_scope_enabled": bool(normalized_scope),
            "document_scope_mode": document_scope_mode,
        }

    @staticmethod
    def _selected_document_not_specified_answer(
        document_scope: Optional[Dict[str, Any]],
    ) -> str:

        normalized_scope = RAGPipeline._normalize_document_scope(document_scope)

        if not normalized_scope:
            return "The selected document does not specify that information."

        document_name = normalized_scope.get("document_name") or "the selected document"

        return (
            f"The selected document, {document_name}, does not specify that information."
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

        complete_document_id = RAGPipeline._safe_int(complete_document_id)

        if complete_document_id is None:
            return None

        document_id = (
            document_scope.get("document_id")
            or document_scope.get("documentId")
        )

        document_id = RAGPipeline._safe_int(document_id)

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
    def _filter_chunks_by_document_scope(
        *,
        chunks: List[Dict[str, Any]],
        document_scope: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized_scope = RAGPipeline._normalize_document_scope(document_scope)

        if not normalized_scope:
            return chunks if isinstance(chunks, list) else []

        selected_complete_document_id = RAGPipeline._safe_int(
            normalized_scope.get("complete_document_id")
        )

        if selected_complete_document_id is None:
            return []

        scoped_chunks: List[Dict[str, Any]] = []

        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue

            chunk_complete_document_id = RAGPipeline._extract_chunk_complete_document_id(
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

        direct_int = RAGPipeline._safe_int(direct_value)

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

            nested_int = RAGPipeline._safe_int(nested_value)

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

            return RAGPipeline._safe_int(complete_document_id)

        return None

    @staticmethod
    def _chunks_to_context(chunks: List[Dict[str, Any]]) -> str:
        texts: List[str] = []

        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue

            text = str(
                chunk.get("content")
                or chunk.get("text")
                or chunk.get("chunk_text")
                or chunk.get("page_content")
                or ""
            ).strip()

            if text:
                texts.append(text)

        return "\n\n".join(texts)

    @staticmethod
    def _resolve_retrieval_top_k(
        *,
        top_k: int,
        document_scope: Optional[Dict[str, Any]],
    ) -> int:
        safe_top_k = RAGPipeline._safe_int(top_k) or 5

        if not RAGPipeline._normalize_document_scope(document_scope):
            return safe_top_k

        return max(safe_top_k, RAGPipeline.DOCUMENT_SCOPE_FALLBACK_TOP_K)

    @staticmethod
    def _clean_answer_kwargs(answer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(answer_kwargs, dict):
            return {}

        cleaned = dict(answer_kwargs)

        for key in (
            "memory_context",
            "conversation_id",
            "document_scope",
            "documentScope",
            "complete_document_id",
            "completeDocumentId",
        ):
            cleaned.pop(key, None)

        return cleaned

    @staticmethod
    def _extract_answer_text(answer_result: Any) -> str:
        if isinstance(answer_result, dict):
            return str(answer_result.get("answer", "") or "")

        if isinstance(answer_result, str):
            return answer_result

        return ""

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

    def _no_relevant_context_result(
        *,
            question: str,
            top_k: int,
            query_embedding: List[Any],
            memory_context: str,
            conversation_id: Optional[str],
            document_scope: Optional[Dict[str, Any]],
            document_scope_mode: str,
    ) -> Dict[str, Any]:

        normalized_scope = RAGPipeline._normalize_document_scope(document_scope)

        if normalized_scope:
            answer = RAGPipeline._selected_document_not_specified_answer(normalized_scope)
        else:
            answer = (
                "I do not have enough relevant document context to answer that from the "
                "stored manuals. The retrieved results were not close enough to trust. "
                "Try rephrasing with the equipment name, station, fault, part, or document name."
            )

        return {
            "answer": answer,
            "documents": [],
            "used_chunks": [],
            "chunks": [],
            "query_embedding": query_embedding or [],
            "retriever_top_k": top_k,
            "conversation_id": conversation_id,
            "memory_context_used": bool((memory_context or "").strip()),
            "memory_context_mode": (
                "separate_memory_context"
                if (memory_context or "").strip()
                else "none"
            ),
            "document_scope": normalized_scope,
            "document_scope_enabled": bool(normalized_scope),
            "document_scope_mode": document_scope_mode,
            "document_profile_used": False,
        }

_default_rag: Optional[RAGPipeline] = None


def get_default_rag() -> RAGPipeline:
    global _default_rag

    if _default_rag is None:
        try:
            db_config = DatabaseConfig()
            _default_rag = RAGPipeline(db_config=db_config)
            info_id("[RAG] Default RAGPipeline initialized")
        except Exception as e:
            error_id(f"[RAG] Failed to construct RAGPipeline: {e}")
            raise

    return _default_rag



__all__ = ["RAGPipeline", "get_default_rag"]