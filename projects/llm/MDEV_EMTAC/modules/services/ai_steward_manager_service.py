# modules/services/ai_steward_manager_service.py

from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    with_request_id,
    error_id,
)

from modules.services.unified_search_service import UnifiedSearchService


class AIStewardManagerService:
    """
    Stateless AI execution service.

    Responsibilities:
        - Call UnifiedSearchService
        - Return raw domain result
        - No persistence
        - No transaction ownership
        - No session creation
        - No UI formatting
    """

    def __init__(self):
        # Heavy components can be cached safely
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
    ) -> Dict[str, Any]:

        if not question or not question.strip():
            return {
                "strategy": "invalid_input",
                "answer": "Please provide a more detailed question.",
                "chunks": [],
                "documents": [],
            }

        try:
            result = self.search_service.execute(
                session=session,
                user_id=user_id,
                question=question.strip(),
                request_id=request_id,
                rag_only=rag_only,
                forced_chunk_id=forced_chunk_id,
            )

            return result or {}

        except Exception as e:
            error_id(
                f"AIStewardManagerService failure: {e}",
                request_id,
                exc_info=True,
            )

            return {
                "strategy": "error",
                "answer": "AI processing error.",
                "chunks": [],
                "documents": [],
            }

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
    ) -> Dict[str, Any]:

        if not question or not question.strip():
            return {
                "strategy": "invalid_input",
                "answer": "Please provide a more detailed question.",
                "chunks": [],
                "documents": [],
                "images": [],
                "drawings": [],
                "parts": [],
                "model_name": None,
            }

        try:
            result = self.search_service.execute(
                session=session,
                user_id=user_id,
                question=question.strip(),
                request_id=request_id,
                rag_only=rag_only,
                forced_chunk_id=forced_chunk_id,
            )

            # --------------------------------------------------
            # Defensive model_name injection
            # --------------------------------------------------
            model_name = None

            try:
                rag_pipeline = getattr(self.search_service, "rag_pipeline", None)

                if rag_pipeline:
                    answer_generator = getattr(rag_pipeline, "answer_generator", None)

                    # Most common pattern
                    model_name = getattr(answer_generator, "model_name", None)

                    # Fallback: if generator exposes config object
                    if not model_name:
                        model_config = getattr(answer_generator, "model_config", None)
                        if model_config:
                            model_name = getattr(model_config, "model_name", None)

            except Exception:
                # Never allow telemetry extraction to break AI flow
                model_name = None

            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {}

            result.setdefault("strategy", "rag")
            result.setdefault("documents", [])
            result.setdefault("images", [])
            result.setdefault("drawings", [])
            result.setdefault("parts", [])
            result.setdefault("answer", "")

            # Inject model_name safely
            result["model_name"] = model_name

            return result

        except Exception as e:
            error_id(
                f"AIStewardManagerService failure: {e}",
                request_id,
                exc_info=True,
            )

            return {
                "strategy": "error",
                "answer": "AI processing error.",
                "chunks": [],
                "documents": [],
                "images": [],
                "drawings": [],
                "parts": [],
                "model_name": None,
            }