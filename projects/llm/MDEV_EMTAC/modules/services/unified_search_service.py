# modules/services/unified_search_service.py
# Stateless RAG-first unified search domain service.

from __future__ import annotations

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

    Performance notes:
        - Answer-first mode intentionally skips relationship/payload work.
        - Payload fallback document aggregation is lazy.
        - Relationship projection is attempted before fallback aggregation.
        - Expensive stages are timed and returned in payload_performance.
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
        """

        question = (question or "").strip()

        if not question:
            return self._empty_response(
                strategy="invalid_input",
                answer="Please provide a valid question.",
                payload_status="unavailable",
            )

        if not self.rag_pipeline:
            return self._empty_response(
                strategy="rag_unavailable",
                answer="The AI assistant is temporarily unavailable.",
                payload_status="unavailable",
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
            return self._run_forced_chunk_pipeline(
                session=session,
                question=question,
                chunk_id=effective_forced_chunk,
                request_id=request_id,
                include_payload=include_payload,
            )

        rag_start = time.perf_counter()

        with tracer.span("rag_run", meta={"include_payload": include_payload}):
            rag_result = self.rag_pipeline.run(
                question=question,
                request_id=request_id,
            )

        rag_time = time.perf_counter() - rag_start

        if not isinstance(rag_result, dict):
            warning_id(
                "[UnifiedSearchService] RAG pipeline returned non-dict result.",
                request_id,
            )
            return self._empty_response(
                strategy="rag_invalid_result",
                answer="The AI assistant returned an invalid search result.",
                payload_status="unavailable",
            )

        used_chunks = self._resolve_seed_chunks(rag_result)
        answer = rag_result.get("answer", "") or ""

        debug_id(
            "[UnifiedSearchService] Normal RAG result "
            f"include_payload={include_payload} "
            f"used_chunks={len(used_chunks)} "
            f"rag_time={rag_time:.3f}s",
            request_id,
        )

        if not include_payload:
            debug_id(
                "[UnifiedSearchService] Normal RAG answer-first mode: "
                "skipping relationship_map and UI payload projection.",
                request_id,
            )

            return {
                "strategy": "rag",
                "method": "rag",
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
            strategy="rag",
            method="rag",
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
        )

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

        chunks = self._resolve_seed_chunks(result)

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
    ) -> Dict[str, Any]:
        """
        Runs the RAG answer generator against one forced Document chunk.

        include_payload=False:
            Returns answer-first seed only:
                - answer
                - chunks
                - used_chunks
                - debug metadata

            Skips the expensive payload work:
                - DocumentUIPayload aggregation
                - relationship map resolution
                - ChunkRelationshipProjection
                - image/part/drawing promotion

        include_payload=True:
            Legacy/full payload behavior.
        """

        debug_id(
            f"[UnifiedSearchService] Forced chunk pipeline start "
            f"chunk_id={chunk_id} include_payload={include_payload}",
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
            )

        complete_doc = None
        complete_document_id = getattr(doc, "complete_document_id", None)

        if complete_document_id:
            complete_doc = (
                session.query(CompleteDocument)
                .filter(CompleteDocument.id == complete_document_id)
                .first()
            )

        forced_chunks = [
            {
                "document_id": doc.id,
                "chunk_id": doc.id,
                "id": doc.id,
                "content": doc.content or "",
                "complete_document_id": complete_document_id,
                "complete_document_title": (
                    complete_doc.title if complete_doc else None
                ),
                "document_title": (
                    complete_doc.title if complete_doc else None
                ),
                "title": (
                    complete_doc.title
                    if complete_doc
                    else f"Forced Chunk #{doc.id}"
                ),
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

        with tracer.span("forced_chunk_build_context"):
            ctx = self.rag_pipeline.context_builder.build_context(
                retrieved_chunks=forced_chunks,
                request_id=request_id,
            )

        if not isinstance(ctx, dict):
            warning_id(
                "[UnifiedSearchService] Context builder returned non-dict "
                "during forced chunk pipeline; using forced chunk directly.",
                request_id,
            )
            ctx = {
                "used_chunks": forced_chunks,
                "context": forced_chunks[0]["content"],
            }

        used_chunks = ctx.get("used_chunks", []) or forced_chunks
        context = ctx.get("context", "") or forced_chunks[0]["content"]

        answer_start = time.perf_counter()

        with tracer.span("forced_chunk_generate_answer"):
            answer_result = self.rag_pipeline.answer_generator.generate_answer(
                question=question,
                context=context,
                request_id=request_id,
            )

        answer_time = time.perf_counter() - answer_start
        answer = self._extract_answer_text(answer_result)

        if not include_payload:
            debug_id(
                "[UnifiedSearchService] Forced chunk answer-first mode: "
                "skipping relationship_map and UI payload projection.",
                request_id,
            )

            return {
                "strategy": "forced_chunk",
                "method": "forced_chunk",
                "answer": answer,
                "chunks": used_chunks,
                "used_chunks": used_chunks,
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
            strategy="forced_chunk",
            method="forced_chunk",
            answer=answer,
            chunks=used_chunks,
            used_chunks=used_chunks,
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
        )

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
    ) -> Dict[str, Any]:

        payload_start = time.perf_counter()

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
    ) -> Dict[str, Any]:
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
            "payload_performance": {
                "relationship_map_time": 0.0,
                "projection_time": 0.0,
                "fallback_document_time": 0.0,
                "post_process_time": 0.0,
                "payload_build_time": 0.0,
                "fallback_documents_built": False,
            },
        }