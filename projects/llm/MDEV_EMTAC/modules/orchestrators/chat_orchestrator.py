import time
from typing import Dict, Any

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.services.ai_steward_manager_service import AIStewardManagerService
from modules.services.search_tracking_service import SearchTrackingService
from modules.services.qanda_service import QandAService
from modules.services.chat_service import ChatService
from modules.configuration.config import (
    FORCE_DEBUG_CHUNK,
    FORCE_DEBUG_CHUNK_ID,
)
from modules.configuration.log_config import (
    with_request_id,
    info_id,
    error_id,
    debug_id,
)


class ChatOrchestrator(BaseOrchestrator):

    def __init__(self):
        super().__init__()
        self.ai_service = AIStewardManagerService()
        self.tracking_service = SearchTrackingService()
        self.qanda_service = QandAService()
        self.chat_service = ChatService()

    @with_request_id
    def handle_question(
        self,
        *,
        user_id: str,
        question: str,
        client_type: str,
        request_id: str = None,
    ) -> Dict[str, Any]:

        request_start = time.time()

        try:
            with self.transaction() as session:

                # --------------------------------------------------
                # DEBUG MODE RESOLUTION (Orchestrator owns this)
                # --------------------------------------------------
                forced_chunk_id = None

                if FORCE_DEBUG_CHUNK:
                    forced_chunk_id = FORCE_DEBUG_CHUNK_ID
                    debug_id(
                        f"[ChatOrchestrator] DEBUG MODE ENABLED → chunk={forced_chunk_id}",
                        request_id,
                    )

                # --------------------------------------------------
                # 1️⃣ AI Execution
                # --------------------------------------------------
                ai_start = time.time()

                ai_result = self.ai_service.execute(
                    session=session,
                    user_id=user_id,
                    question=question,
                    request_id=request_id,
                    forced_chunk_id=forced_chunk_id,   # <-- explicit injection
                )

                ai_time = time.time() - ai_start

                if not isinstance(ai_result, dict):
                    raise ValueError("AI service returned invalid response format")

                # --------------------------------------------------
                # 2️⃣ Persist Q&A
                # --------------------------------------------------
                persist_start = time.time()

                answer = ai_result.get("answer", "")

                self.qanda_service.create_interaction(
                    session=session,
                    user_id=user_id,
                    question=question,
                    answer=answer,
                    request_id=request_id,
                    processing_time_ms=int(ai_time * 1000),
                    raw_response=ai_result,
                )

                persist_time = time.time() - persist_start

                # --------------------------------------------------
                # 3️⃣ Analytics (non-blocking)
                # --------------------------------------------------
                try:
                    self.analytics_tracker.record_query(
                        session=session,
                        session_id=None,
                        query=question,
                        user_id=user_id,
                        result_count=len(ai_result.get("documents", [])),
                        success=ai_result.get("strategy") != "error",
                        method=ai_result.get("strategy", "rag"),
                        duration_ms=int((time.time() - request_start) * 1000),
                        request_id=request_id,
                    )
                except Exception:
                    pass

                # --------------------------------------------------
                # 4️⃣ Format Response
                # --------------------------------------------------
                response = self.chat_service.format_response(ai_result)

            # -------------------------------
            # Transaction commits here
            # -------------------------------

            total_time = time.time() - request_start

            # --------------------------------------------------
            # Attach Standard Fields
            # --------------------------------------------------
            response["response_time"] = total_time
            response["request_id"] = request_id
            response["status"] = response.get("status", "success")
            response.setdefault("method", ai_result.get("method"))
            response.setdefault("strategy", ai_result.get("strategy"))

            # --------------------------------------------------
            # Explicit Debug Metadata
            # --------------------------------------------------
            response["debug_mode"] = ai_result.get("debug_mode", False)
            response["debug_chunk_id"] = ai_result.get("debug_chunk_id")

            # --------------------------------------------------
            # Performance Payload
            # --------------------------------------------------
            if client_type == "debug":
                response["performance"] = {
                    "total_time": total_time,
                    "ai_time": ai_time,
                    "persist_time": persist_time,
                    "method": response.get("method"),
                    "debug_mode": response["debug_mode"],
                }
            else:
                response["performance"] = {
                    "total_time": total_time,
                    "method": response.get("method"),
                }

            info_id(
                f"Chat processed in {total_time:.3f}s "
                f"(AI: {ai_time:.3f}s | Persist: {persist_time:.3f}s)",
                request_id,
            )

            return response

        except Exception as e:
            error_id(
                f"ChatOrchestrator failure: {e}",
                request_id,
                exc_info=True,
            )

            total_time = time.time() - request_start

            return {
                "status": "error",
                "answer": "An unexpected error occurred while processing your request.",
                "method": "error",
                "strategy": "error",
                "debug_mode": False,
                "request_id": request_id,
                "response_time": total_time,
                "blocks": {
                    "documents-container": [],
                    "parts-container": [],
                    "images-container": [],
                    "drawings-container": [],
                },
                "performance": {
                    "total_time": total_time,
                    "method": "error",
                },
            }