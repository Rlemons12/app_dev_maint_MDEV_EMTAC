"""
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

from modules.configuration.log_config import logger, with_request_id, debug_id, info_id, error_id,warning_id
from modules.configuration.config_env import DatabaseConfig
from modules.emtac_ai.search.UnifiedSearch import UnifiedSearch
from plugins.ai_modules.ai_models import ModelsConfig
from modules.emtac_ai.response_formatter import ResponseFormatter

# DB models used here
from modules.emtacdb.emtacdb_fts import QandA, Document

# Import from new NLP package
from modules.emtac_ai.search.nlp import SearchQueryTracker
from modules.emtac_ai.search.nlp.tracker import SearchSessionManager
from modules.emtac_ai.search.db_search_repo.aggregate_search import AggregateSearch
from plugins.ai_modules.ai_models import ModelsConfig

db_config = DatabaseConfig()


# -------------------------------
# Utility: request id helper
# -------------------------------
@with_request_id
def get_request_id() -> str:
    """Helper function to get request ID from context or generate one"""
    try:
        from modules.configuration.log_config import get_current_request_id
        return get_current_request_id()
    except Exception:
        return str(uuid.uuid4())[:8]


# ---------------------------------------
# AistManager: thin wrapper over UnifiedSearch
# ---------------------------------------
class AistManager(UnifiedSearch):
    """
    Orchestrator that delegates *all* search to the UnifiedSearch hub:
      - initialize hub + tracking
      - answer_question → execute_unified_search
      - format + persist interaction
    """

    def __init__(self, ai_model=None, db_session=None):
        self.ai_model = ai_model
        self.db_session = db_session
        self.start_time = None
        self.db_config = DatabaseConfig()
        self.performance_history: List[float] = []
        self.current_request_id: Optional[str] = None

        # Tracking state
        self.tracked_search = None
        self.current_user_id: Optional[str] = None
        self.current_session_id: Optional[str] = None
        self.query_tracker = None

        logger.info("=== AIST MANAGER INITIALIZATION (UnifiedSearch hub) ===")

        # Initialize the unified search hub
        try:
            UnifiedSearch.__init__(
                self,
                db_session=self.db_session,
                enable_orchestrator=True,
                enable_vector=True,
                enable_fts=True
            )
            logger.info("UnifiedSearch hub initialized.")
        except Exception as e:
            logger.error(f"UnifiedSearch initialization failed: {e}")

        # Initialize tracking
        self._init_tracking()
        logger.info("=== AIST MANAGER INITIALIZATION COMPLETE ===")

    def begin_request(self, request_id: Optional[str] = None) -> None:
        """Compatibility shim: start request lifecycle (kept for endpoint)."""
        self.start_time = time.time()
        if request_id:
            self.current_request_id = request_id
            info_id(f"[AistManager] Request {request_id} started")

    # ---------- Tracking ----------
    @with_request_id
    def _init_tracking(self) -> bool:
        """Initialize search tracking components (DB-backed)."""
        if not self.db_session:
            logger.warning("No database session - tracking disabled")
            return False
        try:
            self.query_tracker = SearchQueryTracker(self.db_session)
            logger.info("Search tracking initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Tracking not available: {e}")
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
                        'component': 'aist_manager',
                        'session_started_at': datetime.utcnow().isoformat()
                    }
                )
                logger.info(f"Started tracking session {self.current_session_id} for user {user_id}")
                return True
            logger.debug(f"Set current user {user_id} (tracking disabled)")
            return False
        except Exception as e:
            logger.error(f"Failed to set current user {user_id}: {e}")
            return False

    # ---------- Main flow ----------
    @with_request_id
    def answer_question(
        self,
        user_id: str,
        question: str,
        client_type: str = "web",
        request_id: Optional[str] = None,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entry: delegate to the unified hub and format output."""
        self.start_time = time.time()
        rid = request_id or get_request_id()
        self.current_request_id = rid

        logger.info(f"Processing question: {question}", extra={"request_id": rid})

        try:
            # Ensure tracking session exists if enabled
            if self.query_tracker and self.tracked_search and not self.current_session_id:
                try:
                    self.set_current_user(
                        user_id or "anonymous",
                        {
                            "client_type": client_type,
                            "rating": rating,
                            "comment": bool(comment),
                            "session_started_at": datetime.utcnow().isoformat(),
                        },
                    )
                except Exception as e:
                    logger.warning(f"Failed to set user session: {e}")

            # Delegate to hub
            raw_results = self.execute_unified_search(
                question=question,
                user_id=user_id,
                request_id=rid,
            )

            # ✅ Correct call order
            formatted = self._format_final_response(user_id, question, raw_results, rid)

            # Persist
            self.record_interaction(user_id, question, formatted, rid)

            return formatted

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            return self._create_error_response(e, question, user_id, rid)

    @with_request_id
    def _format_final_response(
            self,
            user_id: str,
            question: str,
            results: dict,
            request_id: Optional[str] = None
    ) -> dict:
        rid = request_id or get_request_id()
        debug_id("[AistManager] _format_final_response called", rid)

        # Handle naming schemes (orchestrator vs unified payloads)
        intent = results.get("intent") or results.get("detected_intent") or "unknown"
        method = results.get("method") or results.get("search_method") or "orchestrator"

        items = results.get("results", []) or results.get("results_by_type", {}).get(intent, [])
        total = results.get("total_results", len(items))

        # ✅ Always pre-populate blocks with empty arrays
        blocks: dict[str, list] = {
            "parts-container": [],
            "images-container": [],
            "documents-container": [],
            "drawings-container": [],
        }

        answer_text = ""

        try:
            if intent == "parts":
                blocks["parts-container"] = [
                    {
                        "id": r.get("id"),
                        "title": r.get("title") or r.get("name"),
                        "description": r.get("description", ""),
                        "source": r.get("source", "parts_fts"),
                        "score": r.get("score", 0.0),
                    }
                    for r in items
                ]

                # Optional AI summarization
                try:
                    from plugins.ai_modules.ai_models import ModelsConfig
                    ai_model = ModelsConfig.load_ai_model()
                    prompt = (
                            f"Summarize the following {total} parts:\n\n" +
                            "\n".join([f"- {r.get('title') or r.get('name', '')}" for r in items[:10]])
                    )
                    ai_summary = ai_model.get_response(prompt)
                    if ai_summary and "AI disabled" not in ai_summary:
                        answer_text = ai_summary
                        info_id(f"[AistManager] Summarized with {ai_model.__class__.__name__}", rid)
                    else:
                        answer_text = f"Found {total} parts."
                except Exception as e:
                    warning_id(f"[AistManager] Summarization skipped: {e}", rid)
                    answer_text = f"Found {total} parts."

            elif intent == "images":
                blocks["images-container"] = [
                    {
                        "id": r.get("id"),
                        "title": r.get("title") or "Untitled Image",
                        "src": r.get("file_path") or r.get("src"),
                        "thumbnail_src": r.get("thumbnail_src") or r.get("file_path"),
                    }
                    for r in items
                ]
                answer_text = f"Found {total} images."

            elif intent == "documents":
                blocks["documents-container"] = [
                    {
                        "id": r.get("id"),
                        "title": r.get("title") or "Untitled Document",
                        "url": r.get("file_path") or r.get("url"),
                    }
                    for r in items
                ]
                answer_text = f"Found {total} documents."

            elif intent == "drawings":
                blocks["drawings-container"] = [
                    {
                        "id": r.get("id"),
                        "title": r.get("drw_name") or r.get("title") or "Untitled Drawing",
                        "url": r.get("file_path"),
                        "drw_equipment_name": r.get("drw_equipment_name") or "Unknown Equipment",
                        "drw_name": r.get("drw_name") or "Untitled",
                        "drw_number": r.get("drw_number") or "N/A",
                        "drw_spare_part_number": r.get("drw_spare_part_number") or "N/A",
                    }
                    for r in items
                ]
                answer_text = f"Found {total} drawings."

            else:
                answer_text = (
                    f"Found {total} results via {method}:{intent}."
                    if total > 0 else f"No results found for: {question}"
                )

            response = {
                "status": "success" if total > 0 else "no_results",
                "answer": answer_text,
                "results": items,
                "intent": intent,
                "method": method,
                "blocks": blocks,  # ✅ Always present with all containers
                "total_results": total,
            }

            info_id(
                f"[AistManager] Response built: status={response['status']}, "
                f"intent={intent}, total={total}", rid
            )
            return response

        except Exception as e:
            error_id(f"[AistManager] Error in _format_final_response: {e}", rid, exc_info=True)
            return {
                "status": "error",
                "answer": f"Error formatting response: {str(e)}",
                "results": [],
                "intent": intent,
                "method": method,
                "blocks": {
                    "parts-container": [],
                    "images-container": [],
                    "documents-container": [],
                    "drawings-container": [],
                },  # ✅ Always included, even on error
                "total_results": 0,
            }

    # ---------- Persistence ----------
    @with_request_id
    def record_interaction(
        self, user_id: str, question: str, response: Dict[str, Any], request_id: str
    ) -> None:
        """Persist interaction into QandA table."""
        try:
            with db_config.main_session() as session:
                QandA.record_interaction(
                    user_id=user_id,
                    question=question,
                    answer=response.get("answer", ""),
                    session=session,
                    request_id=request_id,
                    raw_response=response.get("raw_results", {}),
                )
            logger.info(f"Successfully recorded interaction for request {request_id}")
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}", exc_info=True)

    @with_request_id
    def _create_error_response(self, error, question, user_id=None, request_id=None):
        """Create error response and persist it."""
        error_msg = f"I encountered an error while processing your question: {str(error)}"

        # ✅ Always pre-populate all 4 containers
        empty_blocks = {
            "parts-container": [],
            "images-container": [],
            "documents-container": [],
            "drawings-container": [],
        }

        try:
            # Best-effort persistence
            self.record_interaction(
                user_id or "anonymous",
                question,
                {
                    "status": "error",
                    "answer": error_msg,
                    "results": [],
                    "intent": "error",
                    "method": "error_fallback",
                    "blocks": empty_blocks,  # ✅ always consistent
                    "request_id": request_id or self.current_request_id,
                    "total_results": 0,
                },
                request_id or self.current_request_id,
            )
        except Exception:
            pass

        return {
            "status": "error",
            "answer": error_msg,
            "message": str(error),
            "results": [],
            "intent": "error",
            "method": "error_fallback",
            "blocks": empty_blocks,  # ✅ always consistent
            "total_results": 0,
            "request_id": request_id or self.current_request_id,
        }


# ---------------------------------------
# Global instance factory
# ---------------------------------------
global_aist_manager: Optional[AistManager] = None

@with_request_id
def get_or_create_aist_manager() -> AistManager:
    """Get or create a global AistManager instance with database session for tracking."""
    global global_aist_manager
    if global_aist_manager is None:
        try:
            logger.info("Creating AistManager with tracking support...")
            db_config = DatabaseConfig()
            db_session = db_config.get_session()
            if not db_session:
                logger.error("Could not get database session")
                global_aist_manager = AistManager()
            else:
                logger.info("Database session obtained")
                ai_model = ModelsConfig.load_ai_model()
                global_aist_manager = AistManager(ai_model=ai_model, db_session=db_session)
            logger.info("Global AistManager created successfully")
        except Exception as e:
            logger.error(f"Failed to create AistManager with tracking: {e}")
            try:
                ai_model = ModelsConfig.load_ai_model()
                global_aist_manager = AistManager(ai_model=ai_model)
                logger.info("Created fallback AistManager without tracking")
            except Exception as fallback_error:
                logger.error(f"Fallback AistManager creation failed: {fallback_error}")
                global_aist_manager = AistManager()
    return global_aist_manager

