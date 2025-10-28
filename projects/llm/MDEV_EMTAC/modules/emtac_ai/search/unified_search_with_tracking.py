from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, text
from typing import Dict, Optional, Any
from sqlalchemy.orm import relationship
from datetime import datetime
from modules.configuration.base import Base  # Adjust path as needed
import time
from modules.configuration.log_config import (
    with_request_id, get_request_id,
    info_id, debug_id, warning_id, error_id
)

class UnifiedSearchWithTracking:
    """
    Wrapper around UnifiedSearch that ensures ALL questions are tracked.
    - Session lifecycle
    - Query tracking
    - Analytics recording
    - Request-ID integrated logging
    """

    def __init__(self, unified_search_mixin):
        self.unified_search = unified_search_mixin
        self.query_tracker = None
        self.current_session_id = None
        self.db_session = getattr(unified_search_mixin, 'db_session', None)

        if self.db_session:
            try:
                from modules.search.nlp_search import SearchQueryTracker
                from modules.search.aggregate_search import AggregateSearch
                self.query_tracker = SearchQueryTracker(self.db_session)
                self.aggregate_search = AggregateSearch(session=self.db_session)
                info_id("QueryTracker + AggregateSearch initialized with DB session")
            except Exception as e:
                error_id(f"Failed to init tracker or aggregate search: {e}")
                self.query_tracker = None
                self.aggregate_search = None
        else:
            warning_id("No DB session available - tracking limited")
            try:
                from modules.search.aggregate_search import AggregateSearch
                self.aggregate_search = AggregateSearch()
                info_id("AggregateSearch initialized (no session)")
            except Exception as e:
                error_id(f"Failed to init AggregateSearch: {e}")
                self.aggregate_search = None

    # -------------------------
    # Session Management
    # -------------------------
    @with_request_id
    def start_user_session(self, user_id: str, context_data: Dict = None, request_id=None) -> Optional[int]:
        if self.query_tracker:
            self.current_session_id = self.query_tracker.start_search_session(user_id, context_data)
            info_id(f"Started search session {self.current_session_id} for {user_id}", request_id)
            return self.current_session_id
        return None

    @with_request_id
    def end_session(self, request_id=None) -> bool:
        if self.current_session_id and self.query_tracker:
            success = self.query_tracker.end_search_session(self.current_session_id)
            if success:
                info_id(f"Ended search session {self.current_session_id}", request_id)
                self.current_session_id = None
            return success
        return False

    # -------------------------
    # Main entry: tracked search
    # -------------------------
    @with_request_id
    def execute_unified_search_with_tracking(
        self, question: str, user_id: str = None, request_id: str = None
    ) -> Dict[str, Any]:
        search_start = time.time()
        user_id = user_id or "anonymous"
        req_id = request_id or get_request_id()

        info_id(f"[TRACKED SEARCH] Question: {question}", req_id)

        if not self.current_session_id and self.query_tracker:
            try:
                self.current_session_id = self.query_tracker.start_search_session(user_id)
                debug_id(f"New search session {self.current_session_id} started", req_id)
            except Exception as e:
                warning_id(f"Failed to start search session: {e}", req_id)

        # TODO: Replace with ML-based intent classifier
        detected_intent_id = None
        intent_confidence = 0.0
        intent_classification = None

        result = None
        search_method = "direct_search"
        try:
            if hasattr(self.unified_search, 'execute_unified_search'):
                result = self.unified_search.execute_unified_search(
                    question, user_id=user_id, request_id=req_id
                )
                search_method = result.get("search_method", "unified")
                debug_id(f"Search executed via {search_method}", req_id)
            else:
                result = {"status": "error", "message": "UnifiedSearch not available"}
                search_method = "no_unified_search"
                warning_id("UnifiedSearch not available", req_id)
        except Exception as e:
            error_id(f"Search execution failed: {e}", req_id)
            result = {"status": "error", "message": str(e)}
            search_method = "search_error"

        # -------------------------
        # Tracking + Analytics
        # -------------------------
        execution_time = int((time.time() - search_start) * 1000)
        result_count = result.get("total_results", 0) if isinstance(result, dict) else 0

        query_id = None
        if self.query_tracker and self.current_session_id:
            try:
                query_id = self.query_tracker.track_search_query(
                    session_id=self.current_session_id,
                    query_text=question,
                    detected_intent_id=detected_intent_id,
                    intent_confidence=intent_confidence,
                    search_method=search_method,
                    result_count=result_count,
                    execution_time_ms=execution_time,
                    extracted_entities={},  # TODO: add ML entity extraction later
                    normalized_query=question.lower().strip(),
                )
                info_id(f"Query {query_id} tracked (method={search_method}, results={result_count})", req_id)

                # Insert into analytics table
                try:
                    from modules.search.models.search_models import SearchAnalytics
                    analytics_entry = SearchAnalytics(
                        user_id=user_id,
                        session_id=self.current_session_id,
                        query_text=question,
                        detected_intent=None,  # TODO: add when ML classifier is in place
                        intent_confidence=intent_confidence,
                        search_method=search_method,
                        execution_time_ms=execution_time,
                        result_count=result_count,
                        success=result.get("status") != "error" if isinstance(result, dict) else False,
                        error_message=result.get("message")
                        if isinstance(result, dict) and result.get("status") == "error"
                        else None,
                        user_agent=getattr(self, "request_meta", {}).get("user_agent") if hasattr(self, "request_meta") else None,
                        ip_address=getattr(self, "request_meta", {}).get("ip_address") if hasattr(self, "request_meta") else None,
                    )
                    if self.db_session:
                        self.db_session.add(analytics_entry)
                        self.db_session.commit()
                        debug_id("SearchAnalytics committed", req_id)
                except Exception as e:
                    warning_id(f"Failed to insert SearchAnalytics: {e}", req_id)

            except Exception as e:
                warning_id(f"Failed to track query: {e}", req_id)

        # -------------------------
        # Attach tracking info to result
        # -------------------------
        if isinstance(result, dict):
            result.update({
                "tracking_info": {
                    "query_id": query_id,
                    "session_id": self.current_session_id,
                    "detected_intent_id": detected_intent_id,
                    "intent_confidence": intent_confidence,
                    "execution_time_ms": execution_time,
                    "search_method": search_method,
                    "intent_classification_used": intent_classification is not None,
                    "request_id": req_id,
                }
            })

        info_id(f"Search completed in {execution_time}ms, results={result_count}", req_id)
        return result

    # -------------------------
    # User feedback tracking
    # -------------------------
    @with_request_id
    def record_satisfaction(self, query_id: int, satisfaction_score: int, request_id=None) -> bool:
        if self.query_tracker:
            info_id(f"Recording satisfaction={satisfaction_score} for query={query_id}", request_id)
            return self.query_tracker.record_user_satisfaction(query_id, satisfaction_score)
        return False

    @with_request_id
    def track_result_click(
        self, query_id: int, result_type: str, result_id: int,
        click_position: int, action_taken: str = "view", request_id=None
    ) -> bool:
        if self.query_tracker:
            info_id(
                f"Result clicked: q={query_id}, type={result_type}, id={result_id}, pos={click_position}, action={action_taken}",
                request_id
            )
            return self.query_tracker.track_result_click(
                query_id, result_type, result_id, click_position, action_taken
            )
        return False

    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        if self.query_tracker:
            return self.query_tracker.get_search_performance_report(days)
        return {"error": "Query tracker not available"}



