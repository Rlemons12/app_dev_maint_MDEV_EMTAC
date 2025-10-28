# modules/ai/ai_steward.py
# Updated to use UnifiedSearch hub (formerly UnifiedSearchMixin)
# AistManager is now a thin orchestrator: tracking, persistence, formatting.

import re
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from modules.configuration.log_config import logger, with_request_id
from modules.configuration.config_env import DatabaseConfig
from modules.emtac_ai.query_expansion import UnifiedSearch
from plugins.ai_modules.ai_models import ModelsConfig




# DB models used here (for recording interactions / optional FTS helpers inside formatters)
from modules.emtacdb.emtacdb_fts import (
    QandA, Document
)


# -------------------------------
# Utility: request id helper
# -------------------------------
def get_request_id():
    """Helper function to get request ID from context or generate one"""
    try:
        from modules.configuration.log_config import get_current_request_id
        return get_current_request_id()
    except Exception:
        import uuid
        return str(uuid.uuid4())[:8]


# -----------------------------------
# Response formatting (unchanged API)
# -----------------------------------
class ResponseFormatter:
    """Utility class for formatting search responses."""

    @staticmethod
    def format_search_results(result):
        """Format search results into a user-friendly response."""
        try:
            if not result or not isinstance(result, dict):
                return "I couldn't find relevant information for your query."

            if result.get('status') != 'success':
                error_msg = result.get('message', 'Search failed')
                return f"Search error: {error_msg}"

            # Handle AI-generated knowledge query responses (if unified hub adds these)
            if 'answer' in result and result.get('method') in [
                'ai_knowledge_synthesis_with_chunks',
                'ai_knowledge_synthesis_direct'
            ]:
                ai_answer = result['answer']
                if 'source_info' in result:
                    source_info = result['source_info']
                    if source_info.get('document_source') and source_info.get('chunk_similarity'):
                        similarity = source_info['chunk_similarity']
                        doc_source = source_info['document_source']
                        ai_answer += f"\n\n*Source: {doc_source} (Similarity: {similarity:.1%})*"
                    elif source_info.get('source_type') == 'ai_general_knowledge':
                        ai_answer += f"\n\n*Source: AI General Knowledge*"
                return ai_answer

            total_results = result.get('total_results', 0)
            if total_results == 0:
                return "No results found for your query."

            # Prefer unified result buckets when present
            if 'organized_results' in result:
                return ResponseFormatter._format_organized_results(result['organized_results'], total_results)
            elif 'results_by_type' in result:
                return ResponseFormatter._format_results_by_type(result['results_by_type'], total_results)
            elif 'results' in result and isinstance(result['results'], list):
                return ResponseFormatter._format_direct_results(result['results'], total_results)
            elif total_results > 0:
                return ResponseFormatter._format_main_result_structure(result, total_results)

            return f"Found {total_results} results for your query."

        except Exception as e:
            logger.error(f"Error formatting search results: {e}", exc_info=True)
            return "Found some results, but had trouble formatting them."

    @staticmethod
    def _format_organized_results(organized_results, total_results):
        """Format organized results structure."""
        parts = []

        if 'parts' in organized_results and organized_results['parts']:
            parts_list = organized_results['parts'][:10]
            parts.append(f"Found {len(parts_list)} Banner sensor{'s' if len(parts_list) != 1 else ''}:")
            for i, part in enumerate(parts_list, 1):
                part_info = f"{i}. {part.get('part_number', 'Unknown')}"
                if part.get('name'):
                    part_info += f" - {part.get('name')}"
                if part.get('oem_mfg'):
                    part_info += f" (Manufacturer: {part['oem_mfg']})"
                parts.append(part_info)

        if 'images' in organized_results and organized_results['images']:
            image_count = len(organized_results['images'])
            parts.append(f"\nFound {image_count} related image{'s' if image_count != 1 else ''}.")

        if 'positions' in organized_results and organized_results['positions']:
            position_count = len(organized_results['positions'])
            parts.append(f"\nFound {position_count} installation location{'s' if position_count != 1 else ''}.")

        return "\n".join(parts) if parts else f"Found {total_results} results for your query."

    @staticmethod
    def _format_results_by_type(results_by_type, total_results):
        """Format results_by_type structure."""
        response_parts = []

        # Handle parts
        if 'parts' in results_by_type and results_by_type['parts']:
            parts_list = results_by_type['parts'][:10]
            response_parts.append(f"Found {len(parts_list)} part{'s' if len(parts_list) != 1 else ''}:")
            for i, part in enumerate(parts_list, 1):
                part_info = f"{i}. {part.get('part_number', 'Unknown')}"
                if part.get('name'):
                    part_info += f" - {part.get('name')}"
                if part.get('oem_mfg'):
                    part_info += f" (Manufacturer: {part['oem_mfg']})"
                response_parts.append(part_info)

        # Handle other types
        for result_type, results in results_by_type.items():
            if result_type == 'parts' or not results:
                continue
            if response_parts:
                response_parts.append("")
            type_name = result_type.replace('_', ' ').title()
            response_parts.append(f"Found {len(results)} {type_name}:")
            for i, item in enumerate(results[:5], 1):
                item_info = f"{i}. {item.get('title', item.get('name', 'Unknown'))}"
                response_parts.append(item_info)

        return "\n".join(response_parts) if response_parts else f"Found {total_results} results."

    @staticmethod
    def _format_direct_results(results, total_results):
        """Format direct results list structure - handles both dicts and SQLAlchemy objects."""
        if not results or not isinstance(results, list):
            return f"Found {total_results} results for your query."
        if len(results) == 0:
            return "No results found for your query."

        response_parts = []
        results_list = results[:10]  # Limit to first 10 results
        response_parts.append(f"Found {len(results_list)} result{'s' if len(results_list) != 1 else ''}:")
        for i, item in enumerate(results_list, 1):
            # Handle SQLAlchemy Part-like objects
            if hasattr(item, 'part_number'):
                part_info = f"{i}. {item.part_number or 'Unknown Part'}"
                if getattr(item, 'name', None):
                    part_info += f" - {item.name}"
                if getattr(item, 'oem_mfg', None):
                    part_info += f" (Manufacturer: {item.oem_mfg})"
                if getattr(item, 'model', None) and item.model != getattr(item, 'name', None):
                    part_info += f" [Model: {item.model}]"
                response_parts.append(part_info)

            elif isinstance(item, dict):
                if 'part_number' in item:
                    part_info = f"{i}. {item.get('part_number', 'Unknown')}"
                    if item.get('name'):
                        part_info += f" - {item.get('name')}"
                    if item.get('oem_mfg'):
                        part_info += f" (Manufacturer: {item['oem_mfg']})"
                    response_parts.append(part_info)
                else:
                    name = (item.get('name') or item.get('title') or item.get('id') or 'Unknown')
                    description = (item.get('description') or item.get('notes') or item.get('model') or item.get('oem_mfg') or '')
                    if description:
                        response_parts.append(f"{i}. {name} - {description}")
                    else:
                        response_parts.append(f"{i}. {name}")
            else:
                response_parts.append(f"{i}. {str(item)}")

        return "\n".join(response_parts)

    @staticmethod
    def _format_main_result_structure(result, total_results):
        """Handle main result structure."""
        response_parts = []
        if 'summary' in result:
            response_parts.append(result['summary'])

        found_results = False
        for key in ['parts', 'results', 'data', 'items']:
            if key in result and isinstance(result[key], list) and result[key]:
                found_results = True
                items_list = result[key][:10]
                if not response_parts:
                    response_parts.append(f"Found {len(items_list)} result{'s' if len(items_list) != 1 else ''}:")
                for i, item in enumerate(items_list, 1):
                    if isinstance(item, dict):
                        if 'part_number' in item:
                            part_info = f"{i}. {item.get('part_number', 'Unknown')}"
                            if item.get('name'):
                                part_info += f" - {item.get('name')}"
                            response_parts.append(part_info)
                        else:
                            name = item.get('name', item.get('title', item.get('id', 'Unknown')))
                            response_parts.append(f"{i}. {name}")
                    else:
                        response_parts.append(f"{i}. {str(item)}")
                break

        if not found_results:
            response_parts.append(f"Found {total_results} results for your query.")
        return "\n".join(response_parts) if response_parts else f"Found {total_results} results."


# ---------------------------------------
# AistManager: thin wrapper over hub
# ---------------------------------------
class AistManager(UnifiedSearch):
    """
    Thin orchestrator that delegates *all* search to the UnifiedSearch hub:
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
                enable_orchestrator=True,  # Orchestrator (NER + intent + adapters)
                enable_vector=True,        # Vector / hybrid (AggregateSearch)
                enable_fts=True,           # TSVECTOR FTS
                enable_regex=False         # Legacy regex fallback disabled by default
            )
            logger.info("UnifiedSearch hub initialized.")
        except Exception as e:
            logger.error(f"UnifiedSearch initialization failed: {e}")

        # Initialize tracking (optional, DB-backed)
        self._init_tracking()
        logger.info("=== AIST MANAGER INITIALIZATION COMPLETE ===")

    # ---------- Tracking ----------
    @with_request_id
    def _init_tracking(self) -> bool:
        """Initialize search tracking components (DB-backed)."""
        if not self.db_session:
            logger.warning("No database session - tracking disabled")
            return False
        try:
            from modules.search.nlp_search import SearchQueryTracker
            from modules.search.models.search_models import UnifiedSearchWithTracking
            self.query_tracker = SearchQueryTracker(self.db_session)
            self.tracked_search = UnifiedSearchWithTracking(self)  # wraps hub method
            self.tracked_search.query_tracker = self.query_tracker
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
            if self.tracked_search:
                self.current_session_id = self.tracked_search.start_user_session(
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

    @with_request_id
    def end_user_session(self) -> bool:
        try:
            if self.tracked_search and self.current_session_id:
                success = self.tracked_search.end_session()
                if success:
                    logger.info(f"Ended tracking session {self.current_session_id}")
                    self.current_session_id = None
                    self.current_user_id = None
                return success
            return False
        except Exception as e:
            logger.error(f"Failed to end user session: {e}")
            return False

    # ---------- Main Q&A ----------
    def answer_question(self, user_id, question, client_type='web', request_id=None):
        """Main entry: delegate to the unified hub and format output."""
        self.start_time = time.time()
        self.current_request_id = request_id or get_request_id()
        logger.info(f"Processing question: {question}")

        try:
            # Ensure tracking session exists if enabled
            if self.query_tracker and self.tracked_search and not self.current_session_id:
                try:
                    self.set_current_user(user_id or "anonymous", {'client_type': client_type})
                except Exception as e:
                    logger.warning(f"Failed to set user session: {e}")

            # Delegate to hub
            result = self.execute_unified_search(
                question=question,
                user_id=user_id,
                request_id=self.current_request_id
            )
            return self._format_final_response(result, question, user_id)

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            return self._create_error_response(e, question, user_id)

    # ---------- Formatting / persistence ----------
    @with_request_id
    def _format_final_response(self, result, question, user_id):
        """Format the final response using ResponseFormatter; record interaction."""
        if result and result.get('status') == 'success':
            answer = ResponseFormatter.format_search_results(result)
        elif result and result.get('message'):
            answer = result.get('message', 'Search failed - no specific error message')
        elif result and result.get('answer'):
            answer = result.get('answer')
        else:
            answer = f"I couldn't find specific information about '{question}'."

        # Persist Q&A
        try:
            interaction = self.record_interaction(user_id or "anonymous", question, answer)
            if interaction:
                logger.info(f"Recorded interaction: {interaction.id}")
        except Exception as e:
            logger.warning(f"Failed to record interaction: {e}")

        status = 'success'
        if result:
            if result.get('status') in ['error', 'failed']:
                status = 'error'
            elif result.get('status') in ['success', 'no_results']:
                status = 'success'
            else:
                status = result.get('status', 'success')

        return {
            'status': status,
            'answer': answer,
            'method': result.get('search_method', 'unified') if result else 'unified',
            'total_results': result.get('total_results', 0) if result else 0,
            'request_id': self.current_request_id
        }

    @with_request_id
    def _create_error_response(self, error, question, user_id):
        """Create error response and persist it."""
        error_msg = f"I encountered an error while processing your question: {str(error)}"
        try:
            self.record_interaction(user_id or "anonymous", question, error_msg)
        except Exception:
            pass
        return {
            'status': 'error',
            'answer': error_msg,
            'message': str(error),
            'method': 'error_fallback',
            'total_results': 0,
            'request_id': self.current_request_id
        }

    @with_request_id
    def record_interaction(self, user_id, question, answer):
        """Record interaction with timing."""
        try:
            local_session = None
            session = self.db_session or self.db_config.get_main_session()
            if not self.db_session:
                local_session = session

            try:
                processing_time = None
                if hasattr(self, 'start_time') and self.start_time:
                    processing_time = int((time.time() - self.start_time) * 1000)

                interaction = QandA.record_interaction(
                    user_id=user_id,
                    question=question,
                    answer=answer,
                    session=session,
                    processing_time_ms=processing_time
                )
                if interaction:
                    logger.info(f"Successfully recorded interaction {interaction.id} for user {user_id}")
                    return interaction
                logger.error("QandA.record_interaction returned None")
                return None
            finally:
                if local_session:
                    local_session.close()
        except Exception as e:
            logger.error(f"Error in record_interaction: {e}", exc_info=True)
            return None

    # ---------- Analytics & health ----------
    @with_request_id
    def execute_search_with_analytics(self, question: str, request_id: str = None) -> Dict[str, Any]:
        """Search with tracking analytics where available."""
        search_start = time.time()
        self.current_request_id = request_id or get_request_id()
        logger.info(f"Execute search with analytics: '{question}' (Request: {self.current_request_id})")
        try:
            if self.tracked_search:
                result = self.tracked_search.execute_unified_search_with_tracking(
                    question=question,
                    user_id=self.current_user_id or "anonymous",
                    request_id=self.current_request_id
                )
                result.update({
                    'aist_manager_info': {
                        'request_id': self.current_request_id,
                        'user_id': self.current_user_id,
                        'session_id': self.current_session_id,
                        'vector_backend_available': 'vector' in getattr(self, 'backends', {}),
                        'tracking_enabled': True
                    }
                })
                return result
            else:
                result = self.execute_unified_search(
                    question=question,
                    user_id=self.current_user_id,
                    request_id=self.current_request_id
                )
                result.update({
                    'aist_manager_info': {
                        'request_id': self.current_request_id,
                        'user_id': self.current_user_id,
                        'vector_backend_available': 'vector' in getattr(self, 'backends', {}),
                        'tracking_enabled': False,
                        'fallback_reason': 'tracking_unavailable'
                    }
                })
                return result
        except Exception as e:
            search_time = time.time() - search_start
            logger.error(f"Search failed after {search_time:.3f}s: {e}")
            return {
                'status': 'error',
                'message': f"Search failed: {str(e)}",
                'search_type': 'aist_manager_error',
                'aist_manager_info': {
                    'request_id': self.current_request_id,
                    'error_type': type(e).__name__,
                    'execution_time_ms': int(search_time * 1000)
                }
            }

    @with_request_id
    def get_search_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get search analytics and performance metrics."""
        try:
            if self.tracked_search:
                analytics = self.tracked_search.get_performance_report(days)
                analytics.update({
                    'aist_manager_metrics': self._get_aist_manager_metrics(),
                    'system_health': self._get_system_health_status(),
                    'generated_at': datetime.utcnow().isoformat(),
                    'report_type': 'comprehensive_search_analytics'
                })
                return analytics
            else:
                return {
                    'status': 'limited',
                    'message': 'Search tracking not available',
                    'aist_manager_metrics': self._get_aist_manager_metrics(),
                    'system_health': self._get_system_health_status()
                }
        except Exception as e:
            logger.error(f"Failed to generate analytics: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'error_type': 'analytics_generation_failed'
            }

    @with_request_id
    def _get_aist_manager_metrics(self) -> Dict[str, Any]:
        """Basic AistManager metrics."""
        return {
            'database_available': self.db_session is not None,
            'performance_history_count': len(self.performance_history),
            'current_session_active': self.current_session_id is not None,
            'current_user': self.current_user_id,
            'tracking_enabled': self.tracked_search is not None
        }

    @with_request_id
    def _get_system_health_status(self) -> Dict[str, Any]:
        """Overall system health based on unified hub and DB/AI availability."""
        components = {
            'unified_search': bool(getattr(self, 'backends', {})),  # hub registers backends here
            'vector_backend': 'vector' in getattr(self, 'backends', {}),
            'database': self.db_session is not None,
            'search_tracking': self.tracked_search is not None,
            'ai_model': self.ai_model is not None
        }
        warnings: List[str] = []
        if not components['search_tracking']:
            warnings.append('Search tracking disabled - analytics limited')
        if not components['vector_backend']:
            warnings.append('Vector backend unavailable')
        if not components['database']:
            warnings.append('Database unavailable - persistent search disabled')

        overall = 'healthy'
        critical = ['unified_search', 'database']
        if not all(components[c] for c in critical):
            overall = 'degraded'
        elif warnings:
            overall = 'healthy_with_warnings'

        return {
            'overall_status': overall,
            'components': components,
            'warnings': warnings
        }

    # ---------- Misc utilities ----------
    @with_request_id
    def begin_request(self, request_id=None):
        """Start timing a new request."""
        self.start_time = time.time()
        if request_id:
            self.current_request_id = request_id
            logger.debug(f"Request {request_id} started with performance tracking")
        else:
            self.current_request_id = None

    @with_request_id
    def get_response_time(self):
        """Response time so far."""
        if self.start_time:
            return time.time() - self.start_time
        return 0

    @with_request_id
    def format_response(self, answer, client_type=None, results=None):
        """Format HTML-ish answer and include performance info."""
        formatted_answer = answer.strip()
        if hasattr(self, 'start_time') and self.start_time:
            response_time = time.time() - self.start_time
            if response_time > 2.0 or client_type == 'debug':
                if '<div class="performance-note">' not in formatted_answer:
                    formatted_answer += f"<div class='performance-note'><small>Response time: {response_time:.2f}s</small></div>"

        if '<a href=' not in formatted_answer and ('http://' in formatted_answer or 'https://' in formatted_answer):
            formatted_answer = re.sub(
                r'(https?://[^\s]+)',
                r'<a href="\1" target="_blank">\1</a>',
                formatted_answer
            )
        return formatted_answer

    @with_request_id
    def find_most_relevant_document(self, question, session=None, request_id=None):
        """
        Quick helper that tries vector backend (through DB session) to fetch a single doc.
        Note: This is a lightweight utility and NOT part of the main search path anymore.
        """
        search_start = time.time()
        logger.debug(f"Finding most relevant document for question: {question[:50]}...")

        local_session = None
        if not session:
            if self.db_session:
                session = self.db_session
            else:
                local_session = self.db_config.get_main_session()
                session = local_session

        try:
            # Respect current embedding model setting
            embedding_model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')
            if embedding_model_name == "NoEmbeddingModel":
                logger.info("Embeddings are disabled. Returning None for document search.")
                return None

            def search_operation():
                # Use AggregateSearch via UnifiedSearch vector backend if available
                if 'vector' in getattr(self, 'backends', {}):
                    # The unified hub already does ranked retrieval; here we do a minimal doc fallback
                    from sqlalchemy import text
                    # This is intentionally simple: grab a recent doc as last resort
                    recent = session.query(Document).order_by(Document.id.desc()).limit(1).all()
                    return recent[0] if recent else None

                logger.debug("Vector backend not available; returning a recent doc as fallback")
                recent_docs = session.query(Document).limit(1).all()
                return recent_docs[0] if recent_docs else None

            # very short timeout
            result = search_operation()
            return result

        except Exception as e:
            search_time = time.time() - search_start
            logger.error(f"Error in document search after {search_time:.3f}s: {e}", exc_info=True)
            return None
        finally:
            if local_session:
                local_session.close()

    @with_request_id
    def cleanup_session(self):
        """Clean up the current session and resources."""
        try:
            if self.current_session_id:
                self.end_user_session()
            self.current_request_id = None
            self.start_time = None
            logger.debug("Session cleanup completed")
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_session()


# ---------------------------------------
# Global instance factory
# ---------------------------------------
global_aist_manager: Optional[AistManager] = None

@with_request_id
def get_or_create_aist_manager():
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
