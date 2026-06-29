"""
E:\emtac\projects\llm\MDEV_EMTAC\modules\emtac_ai\aist_manager.py
aist_manager.py
---------------
Thin orchestrator around UnifiedSearch:
- Initializes orchestrator/vector/fts backends
- Handles user/session tracking via SearchQueryTracker
- Formats responses for chatbot frontends
- Persists interactions (QandA table in emtacdb)
- Updates QandA question/answer embeddings on the same QandA row
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from modules.configuration.log_config import (
    logger,
    with_request_id,
    debug_id,
    info_id,
    error_id,
    warning_id,
)
from modules.configuration.config_env import get_db_config
from modules.emtac_ai.search.UnifiedSearch import UnifiedSearch
from modules.services.image_service import ImageService
from modules.services.position_service import PositionService
from modules.services.parts_position_image_service import PartsPositionImageService
from modules.services.image_completed_document_association_service import (
    ImageCompletedDocumentAssociationService,
)

# GPU / unified AI facade ONLY
from modules.services.ai_models_service import AIModelsService

# QandA embedding updater
from modules.services.qanda_embedding_service import QandAEmbeddingService

# DB models
from modules.emtacdb.emtacdb_fts import QandA, Document

# NLP / tracking
from modules.emtac_ai.search.nlp import SearchQueryTracker
from modules.emtac_ai.search.nlp.tracker import SearchSessionManager


# Shared singleton DB config only
db_config = get_db_config()


# -------------------------------
# Utility: request id helper
# -------------------------------
@with_request_id
def get_request_id() -> str:
    try:
        from modules.configuration.log_config import get_current_request_id

        current_request_id = get_current_request_id()

        if current_request_id:
            return current_request_id

        return str(uuid.uuid4())[:8]

    except Exception:
        return str(uuid.uuid4())[:8]


# ---------------------------------------
# AistManager
# ---------------------------------------
class AistManager:
    def __init__(self, db_session=None):
        self.db_session = db_session
        self.start_time = None

        # Use shared singleton config instead of new DatabaseConfig()
        self.db_config = get_db_config()
        self.performance_history: List[float] = []
        self.current_request_id: Optional[str] = None

        # Tracking
        self.current_user_id: Optional[str] = None
        self.current_session_id: Optional[str] = None
        self.query_tracker = None

        # QandA embedding service
        # Initialized once so the embedding model/cache can be reused.
        self.qanda_embedding_service = QandAEmbeddingService()

        logger.info("=== AIST MANAGER INITIALIZATION (UnifiedSearch hub) ===")

        try:
            self.search_engine = UnifiedSearch(
                db_session=self.db_session,
                enable_intent=False,
                enable_vector=True,
                enable_fts=True,
            )

            # Enrichment services
            self.image_assoc_service = ImageCompletedDocumentAssociationService()
            self.position_service = PositionService(self.db_session)
            self.parts_position_image_service = PartsPositionImageService(self.db_session)

            # Inject services into UnifiedSearch
            self.search_engine.image_assoc_service = self.image_assoc_service
            self.search_engine.position_service = self.position_service
            self.search_engine.parts_position_image_service = (
                self.parts_position_image_service
            )

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
                    context_data=context_data
                    or {
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
            - Successful answers are persisted through record_interaction()
            - record_interaction() also updates embeddings on the same QandA row
        """

        self.start_time = time.time()
        self.current_request_id = request_id or get_request_id()

        logger.critical(
            "RUNNING UPDATED answer_question FROM: %s",
            __file__,
            extra={"request_id": self.current_request_id},
        )

        logger.info(
            f"[AIST] Processing question: '{question}'",
            extra={"request_id": self.current_request_id},
        )

        cleaned_question = question.strip() if isinstance(question, str) else ""

        if not cleaned_question:
            logger.warning(
                "[AIST] Empty or invalid question received",
                extra={"request_id": self.current_request_id},
            )

            return self._format_final_response(
                {
                    "answer": "Please provide a more detailed question so I can help you better.",
                    "documents": [],
                    "parts": [],
                    "thumbnails": [],
                    "drawings": [],
                    "images": [],
                    "status": "success",
                    "method": "invalid_input",
                }
            )

        try:
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

            if not self.search_engine or not getattr(
                self.search_engine,
                "rag_pipeline",
                None,
            ):
                logger.error(
                    "[AIST] RAG pipeline not initialized",
                    extra={"request_id": self.current_request_id},
                )

                return self._format_final_response(
                    {
                        "answer": "The AI assistant is temporarily unavailable.",
                        "documents": [],
                        "parts": [],
                        "thumbnails": [],
                        "drawings": [],
                        "images": [],
                        "status": "error",
                        "method": "rag_unavailable",
                    }
                )

            with self.db_config.get_main_session() as session:
                result = self.search_engine.execute_unified_search(
                    question=cleaned_question,
                    user_id=user_id,
                    request_id=self.current_request_id,
                    session=session,
                )

            formatted_response = self._format_final_response(result)

            # Persist QandA row and update embeddings.
            # This method is duplicate-safe by request_id.
            self.record_interaction(
                user_id=str(user_id),
                question=cleaned_question,
                response=formatted_response,
                request_id=self.current_request_id,
            )

            return formatted_response

        except Exception as e:
            logger.error(
                f"[AIST] Error in answer_question: {e}",
                exc_info=True,
                extra={"request_id": self.current_request_id},
            )

            return self._format_final_response(
                {
                    "answer": "An unexpected error occurred while processing your request.",
                    "documents": [],
                    "parts": [],
                    "thumbnails": [],
                    "drawings": [],
                    "images": [],
                    "status": "error",
                    "method": "exception",
                }
            )

    # --------------------------------------------------
    # Drawings extraction helpers
    # --------------------------------------------------
    def _flatten_drawing_navigation(
        self,
        drawing_navigation: Any,
    ) -> List[Dict[str, Any]]:
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

    def _extract_drawings_from_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not isinstance(documents, list):
            return []

        all_drawings: List[Dict[str, Any]] = []

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            nav = doc.get("drawing_navigation")

            if not nav:
                continue

            flattened = self._flatten_drawing_navigation(nav)
            all_drawings.extend(flattened)

        unique: List[Dict[str, Any]] = []
        seen: set = set()

        for drawing in all_drawings:
            key = (
                drawing.get("id")
                or drawing.get("drw_number")
                or drawing.get("drawing_number")
                or drawing.get("url")
                or str(sorted(drawing.items()))
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(drawing)

        return unique

    def _extract_drawings_from_navigation(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen: set[int] = set()
        drawings: List[Dict[str, Any]] = []

        if not isinstance(documents, list):
            return drawings

        for doc in documents:
            if not isinstance(doc, dict):
                continue

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
                            if not isinstance(drw, dict):
                                continue

                            drw_id = drw.get("id")

                            if drw_id is None or drw_id in seen:
                                continue

                            seen.add(drw_id)
                            drawings.append(drw)

        return drawings

    # ---------- Formatting ----------
    def _format_final_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            result = {}

        documents = result.get("documents", []) or []
        parts = result.get("parts", []) or []

        if not isinstance(documents, list):
            documents = []

        if not isinstance(parts, list):
            parts = []

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

        drawings = result.get("drawings")

        if not isinstance(drawings, list) or len(drawings) == 0:
            drawings = self._extract_drawings_from_navigation(documents)

        if not drawings:
            drawings = self._extract_drawings_from_documents(documents)

        images: List[Any] = []

        result_images = result.get("images")

        if isinstance(result_images, list):
            images.extend(result_images)

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            doc_images = doc.get("images")

            if isinstance(doc_images, list):
                images.extend(doc_images)

        images = self._dedupe_payload_items(images)
        drawings = self._dedupe_payload_items(drawings)
        parts = self._dedupe_payload_items(parts)

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

    @staticmethod
    def _dedupe_payload_items(items: Any) -> List[Any]:
        """
        Deduplicate payload items while preserving order.

        Supports dictionaries and non-dictionary values.
        """

        if not isinstance(items, list):
            return []

        unique: List[Any] = []
        seen: set = set()

        for item in items:
            if isinstance(item, dict):
                key = (
                    item.get("id")
                    or item.get("part_id")
                    or item.get("image_id")
                    or item.get("drawing_id")
                    or item.get("drw_number")
                    or item.get("drawing_number")
                    or item.get("src")
                    or item.get("url")
                    or str(sorted(item.items()))
                )
            else:
                key = str(item)

            if key in seen:
                continue

            seen.add(key)
            unique.append(item)

        return unique

    # ---------- Persistence ----------
    @with_request_id
    def record_interaction(
        self,
        user_id: str,
        question: str,
        response: Dict[str, Any],
        request_id: str,
    ) -> None:
        """
        Persist the Q&A interaction, then update the same QandA row with embeddings.

        Important:
            - Creates one QandA row per request_id.
            - If called twice for the same request_id, it updates/reuses the same row.
            - Feedback/rating/comment updates should still happen elsewhere.
            - Embedding failure should not prevent the Q&A row from being saved.
        """

        try:
            with self.db_config.get_main_session() as session:
                answer = response.get("answer", "") if isinstance(response, dict) else ""

                qa_record = self._find_existing_qanda_by_request_id(
                    session=session,
                    user_id=user_id,
                    request_id=request_id,
                )

                if qa_record:
                    logger.info(
                        "[record_interaction] Existing QandA row found. "
                        "qanda_id=%s user_id=%s request_id=%s",
                        qa_record.id,
                        user_id,
                        request_id,
                    )

                    self._update_existing_qanda_interaction(
                        qa_record=qa_record,
                        question=question,
                        answer=answer,
                        raw_response=response,
                    )

                    try:
                        session.commit()
                    except Exception:
                        session.rollback()
                        raise

                else:
                    qa_record = QandA.record_interaction(
                        user_id=user_id,
                        question=question,
                        answer=answer,
                        session=session,
                        request_id=request_id,
                        raw_response=response,
                    )

                if not qa_record:
                    logger.warning(
                        "[record_interaction] No QandA row available after persistence. "
                        "user_id=%s request_id=%s",
                        user_id,
                        request_id,
                    )
                    return

                self._update_qanda_embeddings(
                    session=session,
                    qa_record=qa_record,
                    user_id=user_id,
                    question=question,
                    answer=answer,
                    request_id=request_id,
                )

        except Exception:
            logger.error(
                "[record_interaction] Failed to persist QandA interaction "
                "user_id=%s request_id=%s",
                user_id,
                request_id,
                exc_info=True,
            )

    def _find_existing_qanda_by_request_id(
        self,
        *,
        session,
        user_id: str,
        request_id: str,
    ) -> Optional[QandA]:
        """
        Finds an existing QandA row for the request_id.

        This prevents duplicate rows if:
            - answer_question() records the interaction
            - another route/coordinator also calls record_interaction()
        """

        if not request_id:
            return None

        try:
            query = session.query(QandA).filter(QandA.request_id == request_id)

            if user_id is not None:
                query = query.filter(QandA.user_id == str(user_id))

            return query.order_by(QandA.timestamp.desc()).first()

        except Exception:
            logger.error(
                "[record_interaction] Failed to search existing QandA row "
                "user_id=%s request_id=%s",
                user_id,
                request_id,
                exc_info=True,
            )
            return None

    @staticmethod
    def _update_existing_qanda_interaction(
        *,
        qa_record: QandA,
        question: str,
        answer: str,
        raw_response: Dict[str, Any],
    ) -> None:
        """
        Updates safe QandA fields on an existing row.

        Does not touch:
            - rating
            - comment
            - question_embedding
            - answer_embedding
        """

        if question and not getattr(qa_record, "question", None):
            qa_record.question = question

        if answer and not getattr(qa_record, "answer", None):
            qa_record.answer = answer

        if hasattr(qa_record, "raw_response") and raw_response:
            qa_record.raw_response = raw_response

        if hasattr(qa_record, "question_length"):
            qa_record.question_length = len(qa_record.question or "")

        if hasattr(qa_record, "answer_length"):
            qa_record.answer_length = len(qa_record.answer or "")

    def _update_qanda_embeddings(
        self,
        *,
        session,
        qa_record: QandA,
        user_id: str,
        question: str,
        answer: str,
        request_id: str,
    ) -> None:
        """
        Best-effort embedding update for the same QandA row.

        This method should never break the chatbot response if embedding fails.
        """

        try:
            embedded = self.qanda_embedding_service.embed_existing_qanda(
                session=session,
                qa_id=qa_record.id,
                question=question,
                answer=answer,
                embed_question=True,
                embed_answer=True,
                request_id=request_id,
                skip_existing=True,
                commit=True,
            )

            logger.info(
                "[record_interaction] QandA embedding update finished "
                "qanda_id=%s user_id=%s request_id=%s embedded=%s",
                qa_record.id,
                user_id,
                request_id,
                embedded,
            )

        except Exception:
            logger.error(
                "[record_interaction] QandA embedding update failed "
                "qanda_id=%s user_id=%s request_id=%s",
                qa_record.id,
                user_id,
                request_id,
                exc_info=True,
            )

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
        # Use shared singleton config, do not instantiate DatabaseConfig()
        db_session = get_db_config().get_main_session()
        global_aist_manager = AistManager(db_session=db_session)

    return global_aist_manager