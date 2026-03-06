# unified_search_service.py
# Stateless RAG-first unified search domain service.

from __future__ import annotations

from typing import Dict, Any, Optional, List

from modules.configuration.log_config import (
    logger,
    with_request_id,
    debug_id,
    error_id,
)

from modules.emtac_ai.search.rag_core.rag_pipeline import get_default_rag
from modules.configuration import config


class UnifiedSearchService:
    """
    Stateless unified search domain service.

    Responsibilities:
        - Run RAG-first search
        - Normalize RAG output to UI contract
        - Extract drawings from document navigation
        - Promote images
        - NEVER open sessions
        - NEVER commit
        - NEVER persist tracking
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

                logger.info(
                    "[UnifiedSearchService] RAG initialized: %s",
                    type(self.rag_pipeline).__name__,
                )

            except Exception as e:
                logger.error(
                    f"[UnifiedSearchService] RAG init failed: {e}",
                    exc_info=True,
                )
                self.rag_pipeline = None

    # ------------------------------------------------------------------
    # Public Execution
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
    ) -> Dict[str, Any]:

        # --------------------------------------------------
        # Input Validation
        # --------------------------------------------------

        if not question or not question.strip():
            return self._empty_response(
                strategy="invalid_input",
                answer="Please provide a valid question.",
            )

        if not self.rag_pipeline:
            error_id(
                "[UnifiedSearchService] RAG pipeline unavailable",
                request_id,
            )
            return self._empty_response(
                strategy="rag_unavailable",
                answer="The AI assistant is temporarily unavailable.",
            )

        # --------------------------------------------------
        # Forced Chunk (Not Supported by Current RAGPipeline)
        # --------------------------------------------------

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
                "[UnifiedSearchService] Forced chunk not supported by current RAGPipeline",
                request_id,
            )
            return self._empty_response(
                strategy="forced_chunk_unsupported",
                answer="Forced chunk execution is not supported by this RAG pipeline.",
            )

        # --------------------------------------------------
        # Normal RAG Execution
        # --------------------------------------------------

        rag_result = self.rag_pipeline.run(
            question=question.strip(),
            request_id=request_id,
        )

        documents = rag_result.get("documents", []) or []
        used_chunks = rag_result.get("used_chunks", []) or []

        drawings = self._extract_drawings(documents)
        images = self._promote_images(documents)

        return {
            "strategy": "rag",
            "answer": rag_result.get("answer", ""),
            "chunks": used_chunks,
            "documents": documents,
            "drawings": drawings,
            "images": images,
            "parts": [],
        }

    # ------------------------------------------------------------------
    # Drawing Extraction
    # ------------------------------------------------------------------

    def _extract_drawings(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        drawings: List[Dict[str, Any]] = []
        seen = set()

        for doc in documents:
            nav = doc.get("drawing_navigation")
            if not isinstance(nav, dict):
                continue

            for area in nav.get("areas", []):
                for model in area.get("models", []):
                    for asset in model.get("assets", []):
                        for drw in asset.get("drawings", []):
                            drw_id = drw.get("id")
                            key = drw_id or str(sorted(drw.items()))
                            if key in seen:
                                continue
                            seen.add(key)
                            drawings.append(drw)

        return drawings

    # ------------------------------------------------------------------
    # Image Promotion
    # ------------------------------------------------------------------

    def _promote_images(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        images: List[Dict[str, Any]] = []
        seen = set()

        for doc in documents:
            doc_images = doc.get("images", [])
            if not isinstance(doc_images, list):
                continue

            for img in doc_images:
                img_id = img.get("id")
                key = img_id or str(sorted(img.items()))
                if key in seen:
                    continue
                seen.add(key)
                images.append(img)

        return images

    # ------------------------------------------------------------------
    # Empty Response Helper
    # ------------------------------------------------------------------

    def _empty_response(
        self,
        *,
        strategy: str,
        answer: str,
    ) -> Dict[str, Any]:

        return {
            "strategy": strategy,
            "answer": answer,
            "chunks": [],
            "documents": [],
            "drawings": [],
            "images": [],
            "parts": [],
        }