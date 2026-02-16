"""
E:\emtac\projects\llm\MDEV_EMTAC\modules\emtac_ai\aist_manager.py
aist_manager.py
---------------
Thin orchestrator around UnifiedSearch:
- Initializes orchestrator/vector/fts backends
- Handles user/session tracking via SearchQueryTracker
- Formats responses for chatbot frontends
- Persists interactions (QandA table in emtacdb)
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from modules.configuration.log_config import (
    logger,
    with_request_id,
    debug_id,
    info_id,
    error_id,
    warning_id,
)
from modules.configuration.config_env import DatabaseConfig
from modules.emtac_ai.search.UnifiedSearch import UnifiedSearch
from modules.services.image_service import ImageService
from modules.services.position_service import PositionService
from modules.services.parts_position_image_service import PartsPositionImageService
from modules.services.image_completed_document_association_service import ImageCompletedDocumentAssociationService

# ✅ GPU / unified AI facade ONLY
from modules.services.ai_models_service import AIModelsService

# DB models
from modules.emtacdb.emtacdb_fts import QandA, Document

# NLP / tracking
from modules.emtac_ai.search.nlp import SearchQueryTracker
from modules.emtac_ai.search.nlp.tracker import SearchSessionManager

db_config = DatabaseConfig()


# -------------------------------
# Utility: request id helper
# -------------------------------
@with_request_id
def get_request_id() -> str:
    try:
        from modules.configuration.log_config import get_current_request_id
        return get_current_request_id()
    except Exception:
        return str(uuid.uuid4())[:8]


# ---------------------------------------
# AistManager
# ---------------------------------------
class AistManager:
    def __init__(self, db_session=None):
        self.db_session = db_session
        self.start_time = None

        self.db_config = DatabaseConfig()
        self.performance_history: List[float] = []
        self.current_request_id: Optional[str] = None

        # Tracking
        self.current_user_id: Optional[str] = None
        self.current_session_id: Optional[str] = None
        self.query_tracker = None

        logger.info("=== AIST MANAGER INITIALIZATION (UnifiedSearch hub) ===")

        try:
            self.search_engine = UnifiedSearch(
                db_session=self.db_session,
                enable_intent=False,
                enable_vector=True,
                enable_fts=True,
            )

            # -----------------------------
            # Enrichment services (OWNED HERE)
            # -----------------------------
            self.image_assoc_service = ImageCompletedDocumentAssociationService(self.db_config)
            self.position_service = PositionService(self.db_session)
            self.parts_position_image_service = PartsPositionImageService(self.db_session)

            # -----------------------------
            # Inject services into UnifiedSearch
            # -----------------------------
            self.search_engine.image_assoc_service = self.image_assoc_service
            self.search_engine.position_service = self.position_service
            self.search_engine.parts_position_image_service = self.parts_position_image_service

            logger.info("AistManager: UnifiedSearch + enrichment services initialized.")

        except Exception as e:
            logger.error(f"UnifiedSearch init failed: {e}", exc_info=True)
            self.search_engine = None

        self._init_tracking()
        logger.info("=== AIST MANAGER INITIALIZATION COMPLETE ===")

    # ---------- Tracking ----------
    @with_request_id
    def _init_tracking(self) -> bool:
        if not self.db_session:
            logger.warning("No DB session - tracking disabled")
            return False
        try:
            self.query_tracker = SearchQueryTracker(self.db_session)
            logger.info("Search tracking initialized")
            return True
        except Exception as e:
            logger.warning(f"Tracking unavailable: {e}")
            return False

    # ---------- Session mgmt ----------
    @with_request_id
    def set_current_user(self, user_id: str, context_data: Dict = None) -> bool:
        try:
            self.current_user_id = user_id
            if self.query_tracker:
                manager = SearchSessionManager(self.db_session)
                self.current_session_id = manager.start_session(
                    user_id=user_id,
                    context_data=context_data or {
                        "component": "aist_manager",
                        "session_started_at": datetime.utcnow().isoformat(),
                    },
                )
                logger.info(
                    f"Started session {self.current_session_id} for user {user_id}"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"set_current_user failed: {e}", exc_info=True)
            return False

    # ---------- Main flow ----------
    @with_request_id
    def answer_question(
        self,
        user_id,
        question,
        client_type: str = "web",
        request_id: Optional[str] = None,
    ):
        """
        Main entry point.
        RAG-FIRST, NER-GATED, LLM ALWAYS ANSWERS.

        IMPORTANT:
        - ALL exits go through _format_final_response()
        - Frontend UI contract is enforced here
        """

        self.start_time = time.time()
        self.current_request_id = request_id or get_request_id()

        logger.critical(
            "🚨 RUNNING UPDATED answer_question FROM: %s",
            __file__,
            extra={"request_id": self.current_request_id},
        )

        logger.info(
            f"[AIST] Processing question: '{question}'",
            extra={"request_id": self.current_request_id},
        )

        # --------------------------------------------------
        # Basic input validation (NON-BLOCKING)
        # --------------------------------------------------
        if not question or not question.strip():
            logger.warning(
                "[AIST] Empty or invalid question received",
                extra={"request_id": self.current_request_id},
            )

            # 🔑 ALWAYS return UI-normalized structure
            return self._format_final_response({
                "answer": "Please provide a more detailed question so I can help you better.",
                "documents": [],
                "parts": [],
                "thumbnails": [],
            })

        try:
            # --------------------------------------------------
            # Optional tracking (DOES NOT affect execution)
            # --------------------------------------------------
            if self.query_tracker and not self.current_session_id:
                try:
                    self.set_current_user(
                        user_id or "anonymous",
                        {"client_type": client_type},
                    )
                except Exception as e:
                    logger.warning(
                        f"[AIST] Failed to set tracking session: {e}",
                        extra={"request_id": self.current_request_id},
                    )

            # --------------------------------------------------
            # RAG AVAILABILITY CHECK (CORRECT OWNER)
            # --------------------------------------------------
            if not self.search_engine or not self.search_engine.rag_pipeline:
                logger.error(
                    "[AIST] RAG pipeline not initialized",
                    extra={"request_id": self.current_request_id},
                )

                return self._format_final_response({
                    "answer": "The AI assistant is temporarily unavailable.",
                    "documents": [],
                    "parts": [],
                    "thumbnails": [],
                })

            # --------------------------------------------------
            # UnifiedSearch execution (RAG-FIRST)
            # --------------------------------------------------
            with self.db_config.main_session() as session:
                result = self.search_engine.execute_unified_search(
                    question=question.strip(),
                    user_id=user_id,
                    request_id=self.current_request_id,
                    session=session,  # 🔑 PASS SESSION
                )

            # 🔑 SINGLE EXIT POINT
            return self._format_final_response(result)

        except Exception as e:
            logger.error(
                f"[AIST] Error in answer_question: {e}",
                exc_info=True,
                extra={"request_id": self.current_request_id},
            )

            # 🔑 EVEN ERRORS RETURN VALID UI STRUCTURE
            return self._format_final_response({
                "answer": "An unexpected error occurred while processing your request.",
                "documents": [],
                "parts": [],
                "thumbnails": [],
            })

    # --------------------------------------------------
    # Drawings extraction helpers (DOCUMENT → drawing_navigation → FLAT LIST)
    # --------------------------------------------------
    def _flatten_drawing_navigation(self, drawing_navigation: Any) -> List[Dict[str, Any]]:
        """
        Mirrors your frontend flattening logic, but server-side.
        Expected shape:
        {
          "areas":[
            {"area_name":..., "models":[
              {"model_name":..., "assets":[
                {"asset_name":..., "drawings":[ {...}, ... ]}
              ]}
            ]}
          ]
        }
        """
        results: List[Dict[str, Any]] = []
        if not drawing_navigation or not isinstance(drawing_navigation, dict):
            return results

        areas = drawing_navigation.get("areas")
        if not isinstance(areas, list):
            return results

        for area in areas:
            if not isinstance(area, dict):
                continue
            area_name = area.get("area_name")

            models = area.get("models") or []
            if not isinstance(models, list):
                models = []

            for model in models:
                if not isinstance(model, dict):
                    continue
                model_name = model.get("model_name")

                assets = model.get("assets") or []
                if not isinstance(assets, list):
                    assets = []

                for asset in assets:
                    if not isinstance(asset, dict):
                        continue
                    asset_name = asset.get("asset_name")

                    drawings = asset.get("drawings") or []
                    if not isinstance(drawings, list):
                        drawings = []

                    for drw in drawings:
                        if not isinstance(drw, dict):
                            continue
                        payload = dict(drw)
                        payload["_area"] = area_name
                        payload["_model"] = model_name
                        payload["_asset"] = asset_name
                        results.append(payload)

        return results

    def _extract_drawings_from_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Primary server-side extraction:
        - reads doc["drawing_navigation"] (if present)
        - flattens it
        - dedupes
        """
        if not isinstance(documents, list):
            return []

        all_drawings: List[Dict[str, Any]] = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                continue
            nav = doc.get("drawing_navigation")
            if not nav:
                continue

            flattened = self._flatten_drawing_navigation(nav)
            all_drawings.extend(flattened)

        # Dedupe with stable key preference
        unique: List[Dict[str, Any]] = []
        seen: set = set()

        for d in all_drawings:
            # Prefer id, then drawing number, then url, then full dict
            key = (
                d.get("id")
                or d.get("drw_number")
                or d.get("drawing_number")
                or d.get("url")
                or str(sorted(d.items()))
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(d)

        return unique

    # ---------- Formatting ----------
    def _format_final_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        documents = result.get("documents", []) or []
        parts = result.get("parts", []) or []

        # --------------------------------------------------
        # 🔒 Ensure drawing_navigation ALWAYS exists on docs
        # --------------------------------------------------
        for doc in documents:
            if not isinstance(doc, dict):
                continue

            doc.setdefault(
                "drawing_navigation",
                {
                    "complete_document_id": doc.get("complete_document_id"),
                    "areas": [],
                    "meta": {
                        "area_count": 0,
                        "model_count": 0,
                        "asset_count": 0,
                        "drawing_count": 0,
                    },
                },
            )

        # --------------------------------------------------
        # 📐 Resolve drawings (priority order)
        # 1) explicit top-level drawings
        # 2) drawings extracted from drawing_navigation
        # 3) fallback legacy extraction
        # --------------------------------------------------
        drawings = result.get("drawings")

        if not isinstance(drawings, list) or len(drawings) == 0:
            drawings = self._extract_drawings_from_navigation(documents)

        if not drawings:
            drawings = self._extract_drawings_from_documents(documents)

        # --------------------------------------------------
        # 🖼️ Promote images from documents
        # --------------------------------------------------
        images: List[Any] = []
        for doc in documents:
            doc_images = doc.get("images")
            if isinstance(doc_images, list):
                images.extend(doc_images)

        debug_id(
            f"[FORMAT FINAL] passing docs={len(documents)} "
            f"imgs={len(images)} parts={len(parts)} drawings={len(drawings)}",
            self.current_request_id,
        )

        return {
            "answer": result.get("answer", ""),
            "blocks": {
                "documents-container": documents,
                "parts-container": parts,
                "images-container": images,
                "drawings-container": drawings,
            },
            "request_id": self.current_request_id,
            "total_documents": len(documents),
            "method": result.get("method", "rag"),
            "status": result.get("status", "success"),
            "performance": result.get("performance"),
        }

    def _extract_drawings_from_navigation(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Flatten drawings from doc.drawing_navigation into a unique list.
        """
        seen: set[int] = set()
        drawings: List[Dict[str, Any]] = []

        for doc in documents:
            nav = doc.get("drawing_navigation")
            if not isinstance(nav, dict):
                continue

            for area in nav.get("areas", []):
                for model in area.get("models", []):
                    for asset in model.get("assets", []):
                        for drw in asset.get("drawings", []):
                            drw_id = drw.get("id")
                            if drw_id is None or drw_id in seen:
                                continue
                            seen.add(drw_id)
                            drawings.append(drw)

        return drawings

    # ---------- Persistence ----------
    @with_request_id
    def record_interaction(
        self, user_id: str, question: str, response: Dict[str, Any], request_id: str
    ) -> None:
        try:
            with db_config.main_session() as session:
                QandA.record_interaction(
                    user_id=user_id,
                    question=question,
                    answer=response.get("answer", ""),
                    session=session,
                    request_id=request_id,
                    raw_response=response,
                )
        except Exception as e:
            logger.error("record_interaction failed", exc_info=True)

    # ---------- Error ----------
    def _create_error_response(self, error, question, user_id, request_id):
        return {
            "status": "error",
            "answer": str(error),
            "results": [],
            "intent": "error",
            "method": "error",
            "blocks": {
                "parts-container": [],
                "images-container": [],
                "documents-container": [],
                "drawings-container": [],
            },
            "total_results": 0,
            "request_id": request_id,
        }

    def begin_request(self, request_id: Optional[str] = None) -> None:
        """
        Compatibility shim for legacy endpoints.
        Initializes request timing and request_id.
        """
        self.start_time = time.time()
        if request_id:
            self.current_request_id = request_id
            info_id(f"[AistManager] Request {request_id} started")


# ---------------------------------------
# Global factory
# ---------------------------------------
global_aist_manager: Optional[AistManager] = None


@with_request_id
def get_or_create_aist_manager() -> AistManager:
    global global_aist_manager
    if global_aist_manager is None:
        db_session = DatabaseConfig().get_session()
        global_aist_manager = AistManager(db_session=db_session)
    return global_aist_manager
