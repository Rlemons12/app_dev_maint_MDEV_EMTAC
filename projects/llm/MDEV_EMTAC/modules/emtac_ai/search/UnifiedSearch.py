# unified_search_service.py
# Clean, intent-first + resolver/expander + RAG-primary unified search hub.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from modules.ai.search_pathway.rag_core.document_ui_payload import DocumentUIPayload
from modules.configuration.log_config import (
    logger,
    with_request_id,
    info_id,
    warning_id,
    error_id,
    debug_id,
    get_request_id,
)
from modules.configuration.config import FORCE_DEBUG_CHUNK, FORCE_DEBUG_CHUNK_ID
from modules.emtac_ai.intent_ner.intent_orchestrator import (
    IntentNEROrchestrator,
)
from modules.services.part_service import PartService
from modules.services.image_service import ImageService

# -------------------------------
# Resolvers
# -------------------------------
from modules.emtac_ai.search.resolvers.parts_resolver import PartsResolver
from modules.emtac_ai.search.resolvers.drawings_resolver import DrawingsResolver
from modules.emtac_ai.search.resolvers.documents_resolver import DocumentResolver

# -------------------------------
# Expanders
# -------------------------------
from modules.emtac_ai.search.expanders.parts_search_expander import PartsSearchExpander
from modules.emtac_ai.search.expanders.drawings_search_expander import DrawingsSearchExpander
from modules.emtac_ai.search.expanders.document_search_expander import DocumentSearchExpander

# -------------------------------
# Optional backends
# -------------------------------
try:
    from modules.emtac_ai import AggregateSearch
except Exception:
    AggregateSearch = None

try:
    from modules.emtacdb.emtacdb_fts import CompleteDocument
except Exception:
    CompleteDocument = None

from modules.ai.search_pathway.rag_core.rag_pipeline import get_default_rag
from modules.emtacdb.emtacdb_fts import Document


# config / feature flag
RAG_ONLY_MODE = True

# ----------------------------------------------------------------------
# Tracking primitives
# ----------------------------------------------------------------------
@dataclass
class SearchEvent:
    query: str
    user_id: Optional[str]
    method: str
    started_at: float
    request_id: Optional[str] = None
    intent: Optional[str] = None
    backend: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    result_count: int = 0
    success: bool = False
    error: Optional[str] = None


class SearchTracker:
    def __init__(self, db_session=None):
        self.db_session = db_session

    def start(
        self,
        query: str,
        user_id: Optional[str],
        method: str,
        request_id: Optional[str],
    ) -> SearchEvent:
        return SearchEvent(
            query=query,
            user_id=user_id,
            method=method,
            started_at=time.time(),
            request_id=request_id,
        )

    def finish(
        self,
        ev: SearchEvent,
        result_count: int,
        success: bool,
        intent: Optional[str],
        backend: Optional[str],
        entities: Dict[str, Any],
        error: Optional[str],
    ) -> Dict[str, Any]:
        ev.result_count = int(result_count or 0)
        ev.success = bool(success)
        ev.intent = intent
        ev.backend = backend
        ev.entities = entities or {}
        ev.error = error

        return {
            "query": ev.query,
            "user_id": ev.user_id or "anonymous",
            "request_id": ev.request_id,
            "method": ev.method,
            "intent": ev.intent,
            "backend": ev.backend,
            "entities": ev.entities,
            "result_count": ev.result_count,
            "success": ev.success,
            "error": ev.error,
            "duration_ms": int((time.time() - ev.started_at) * 1000),
            "timestamp": datetime.utcnow().isoformat(),
        }


# ----------------------------------------------------------------------
# UnifiedSearch
# ----------------------------------------------------------------------
class UnifiedSearch:
    """
    Unified search pipeline:

      1) Intent + Entity extraction (IntentNEROrchestrator)
      2) Resolver (IDs only)
      3) Expander (graph traversal)
      4) RAG (primary narrative answer)
      5) Vector / FTS fallback if nothing found
    """

    def __init__(
            self,
            db_session=None,
            *,
            enable_rag: bool = True,
            enable_vector: bool = True,
            enable_fts: bool = True,
            enable_intent: bool = False,  # 🔒 HARD DEFAULT OFF
    ):
        self.db_session = db_session
        self.tracker = SearchTracker(self.db_session)

        # --------------------------------------------------
        # Core components (explicit, predictable)
        # --------------------------------------------------
        self.intent_orchestrator = None  # 🔒 gated
        self.vector_engine = None
        self.fts_enabled = False
        self.rag_pipeline = None

        logger.info(
            "[UnifiedSearch] Initializing (RAG-first, intent gated)"
        )

        # --------------------------------------------------
        # Enrichment services (REQUIRED for UI enrichment)
        # --------------------------------------------------
        self.image_assoc_service = None
        self.position_service = None
        self.parts_position_image_service = None

        logger.info("[UnifiedSearch] Initializing (RAG-first, intent gated)")

        # --------------------------------------------------
        # INTENT / NER — HARD DISABLED
        # --------------------------------------------------
        if enable_intent:
            logger.warning(
                "[UnifiedSearch] Intent/NER explicitly enabled — this is NOT default"
            )
            from modules.emtac_ai.intent.intent_ner_orchestrator import (
                IntentNEROrchestrator,
            )
            self.intent_orchestrator = IntentNEROrchestrator()
        else:
            logger.info(
                "[UnifiedSearch] Intent/NER disabled (RAG-only mode)"
            )

        # --------------------------------------------------
        # VECTOR BACKEND (feeds RAG, never controls flow)
        # --------------------------------------------------
        if enable_vector:
            try:
                from modules.emtac_ai.search.aggregate_search import AggregateSearch

                self.vector_engine = AggregateSearch(self.db_session)
                logger.info("[UnifiedSearch] Vector backend enabled")
            except Exception as e:
                self.vector_engine = None
                logger.warning(
                    f"[UnifiedSearch] Vector backend unavailable: {e}"
                )

        # --------------------------------------------------
        # FULL TEXT SEARCH (feeds RAG)
        # --------------------------------------------------
        if enable_fts:
            try:
                from modules.emtacdb.models import CompleteDocument

                self.fts_enabled = True
                logger.info("[UnifiedSearch] FTS backend enabled")
            except Exception as e:
                self.fts_enabled = False
                logger.warning(
                    f"[UnifiedSearch] FTS backend unavailable: {e}"
                )

        # --------------------------------------------------
        # RAG PIPELINE — REQUIRED
        # --------------------------------------------------
        if enable_rag:
            try:
                logger.info("[UnifiedSearch] Initializing RAG pipeline")

                self.rag_pipeline = get_default_rag()

                if not self.rag_pipeline:
                    logger.critical("[UnifiedSearch] get_default_rag() returned None/False")
                    raise RuntimeError("RAG pipeline initialization failed")

                logger.info(
                    "[UnifiedSearch] RAG pipeline initialized: %s",
                    type(self.rag_pipeline).__name__,
                )

            except Exception as e:
                self.rag_pipeline = None
                logger.error(
                    f"[UnifiedSearch] RAG initialization FAILED: {e}",
                    exc_info=True,
                )

        # --------------------------------------------------
        # FINAL SAFETY CHECK
        # --------------------------------------------------
        if not self.rag_pipeline:
            logger.critical(
                "[UnifiedSearch] RAG pipeline NOT AVAILABLE — system cannot answer questions"
            )

        logger.info("[UnifiedSearch] Initialization complete")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    @with_request_id
    def execute_unified_search(
            self,question: str,
            user_id: str, request_id: Optional[str] = None,
            session=None,
    ) -> Dict[str, Any]:
        if FORCE_DEBUG_CHUNK and session is None:
            raise RuntimeError("Forced chunk requires an active DB session")

        rid = request_id or get_request_id()

        logger.debug(
            f"[UnifiedSearch] query={question!r}",
            extra={"request_id": rid},
        )

        # -------------------------------------------------
        # Force Debug (CORRECT)
        # -------------------------------------------------
        if FORCE_DEBUG_CHUNK:
            debug_id(
                f"[DEBUG] Forcing chunk id={FORCE_DEBUG_CHUNK_ID}",
                request_id,
            )

            chunk = (
                session.query(Document)
                .filter(Document.id == FORCE_DEBUG_CHUNK_ID)
                .one()
            )

            return self._run_forced_chunk_pipeline(
                forced_chunk=chunk,
                request_id=request_id,
                session=session,
            )

        # -------------------------------------------------
        # 0) Safety: ensure RAG exists
        # -------------------------------------------------
        if not self.rag_pipeline:
            logger.error(
                "[UnifiedSearch] RAG pipeline missing at execution time",
                extra={"request_id": rid},
            )
            return {
                "strategy": "rag_unavailable",
                "answer": "The AI assistant is temporarily unavailable.",
                "context": None,
                "chunks": [],
                "documents": [],
                "entities": {},
                "intent": None,
            }

        # -------------------------------------------------
        # 1) ALWAYS run RAG first
        # -------------------------------------------------
        rag_result = self.rag_pipeline.run(
            question=question,
            request_id=rid,
        )

        # -------------------------------------------------
        # 2) HARD STOP: RAG-only mode
        # -------------------------------------------------
        if RAG_ONLY_MODE:
            logger.info(
                "[UnifiedSearch] RAG_ONLY_MODE enabled — skipping Intent & NER",
                extra={"request_id": rid},
            )
            return {
                "strategy": "rag_only",
                "answer": rag_result.get("answer"),
                "context": rag_result.get("context"),
                "chunks": rag_result.get("chunks", []),
                "documents": rag_result.get("documents", []),
                "entities": {},
                "intent": None,
            }

        # -------------------------------------------------
        # 3) Intent/NER path not implemented yet
        # -------------------------------------------------
        logger.warning(
            "[UnifiedSearch] Intent/NER path not implemented — returning RAG result",
            extra={"request_id": rid},
        )
        return {
            "strategy": "rag_fallback",
            "answer": rag_result.get("answer"),
            "context": rag_result.get("context"),
            "chunks": rag_result.get("chunks", []),
            "documents": rag_result.get("documents", []),
            "entities": {},
            "intent": None,
        }

    def _run_forced_chunk_pipeline(
            self,
            *,
            forced_chunk,
            request_id,
            session,
    ):
        """
        Debug path: bypass retrieval and force a known chunk.

        IMPORTANT:
        - aggregate_from_chunks() is called EXACTLY ONCE
        - all enrichment happens AFTER aggregation
        - NO re-aggregation after enrichment (prevents data loss)
        """

        # --------------------------------------------------
        # Normalize forced chunk into search-like structure
        # --------------------------------------------------
        chunks = [{
            "id": forced_chunk.id,
            "chunk_id": forced_chunk.id,
            "document_id": forced_chunk.id,
            "complete_document_id": forced_chunk.complete_document_id,
            "content": forced_chunk.content,
            "distance": 0.0,
            "document_title": forced_chunk.name,
        }]

        # --------------------------------------------------
        # SINGLE aggregation + FULL enrichment chain
        # --------------------------------------------------
        payload_builder = (
            DocumentUIPayload(session=session)
            .aggregate_from_chunks(chunks, request_id=request_id)
            .enrich_with_images(
                self.image_assoc_service,
                request_id=request_id,
            )
            .enrich_with_positions(
                self.position_service,
                session=session,
                request_id=request_id,
            )
            .enrich_with_parts(
                self.parts_position_image_service,
                session=session,
                request_id=request_id,
            )
            .enrich_with_drawings(
                session=session,
                request_id=request_id,
            )
        )

        # 🔑 FINALIZE ONCE
        documents = payload_builder.build()

        # --------------------------------------------------
        # PROMOTE IMAGES TO TOP-LEVEL (UI CONTRACT)
        # --------------------------------------------------
        images = []
        seen = set()

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            for img in doc.get("images", []):
                if not isinstance(img, dict):
                    continue

                img_id = img.get("id")
                if img_id and img_id not in seen:
                    seen.add(img_id)
                    images.append(img)

        debug_id(
            f"[FORCED CHUNK] docs={len(documents)} imgs={len(images)}",
            request_id,
        )

        # --------------------------------------------------
        # FINAL UI-NORMALIZED RESPONSE
        # --------------------------------------------------
        return {
            "answer": "DEBUG MODE: Forced chunk used.",
            "documents": documents,
            "thumbnails": images,  # UI expects thumbnails/images here
            "parts": [],  # can be promoted later if needed
            "method": "forced_chunk",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _organize_results_by_type(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        buckets = {
            "parts": [],
            "drawings": [],
            "images": [],
            "documents": [],
            "positions": [],
            "other": [],
        }

        for r in results:
            t = (r.get("type") or "").lower()
            if t.startswith("part"):
                buckets["parts"].append(r)
            elif "drawing" in t:
                buckets["drawings"].append(r)
            elif "image" in t:
                buckets["images"].append(r)
            elif "document" in t:
                buckets["documents"].append(r)
            elif "position" in t:
                buckets["positions"].append(r)
            else:
                buckets["other"].append(r)

        return buckets

    def _bad_request_response(self, msg: str) -> Dict[str, Any]:
        return {
            "search_type": "unified",
            "status": "error",
            "message": msg,
            "results_by_type": {},
            "timestamp": datetime.utcnow().isoformat(),
        }
