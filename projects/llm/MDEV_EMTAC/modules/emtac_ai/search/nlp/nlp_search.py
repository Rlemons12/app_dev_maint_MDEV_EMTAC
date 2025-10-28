# modules/search/nlp_search.py
"""
COMPLETE Enhanced NLP-enhanced search using spaCy for intelligent natural language understanding.
Integrates with the AggregateSearch system for comprehensive search capabilities.
Enhanced to support intent hierarchies, context-aware detection, pattern templates,
entity synonyms, and DATABASE PATTERN INTEGRATION.

This is a complete replacement that includes all original functionality plus database integration.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
import json
from decimal import Decimal
import pickle
import itertools
from uuid import uuid4
from modules.search.models import SearchQuery, SearchSession,SearchResultClick
# SQLAlchemy imports
from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, JSON, ARRAY, \
    UniqueConstraint, select, func, text
from sqlalchemy.orm import Session, relationship
from modules.configuration.log_config import logger,with_request_id
# Import core search functionality with error handling
try:
    from modules.search.aggregate_search import AggregateSearch

    AGGREGATE_SEARCH_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("AggregateSearch imported successfully")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"AggregateSearch not available: {e}")
    AGGREGATE_SEARCH_AVAILABLE = False

# Force enable MODELS_AVAILABLE for database models
MODELS_AVAILABLE = True

def safe_numeric_multiply(a, b):
    """Safely multiply numeric values, handling Decimal and float types."""
    try:
        val_a = float(a) if a is not None else 0.0
        val_b = float(b) if b is not None else 0.0
        return val_a * val_b
    except (ValueError, TypeError):
        logging.warning(f"Failed to multiply {a} and {b}, returning 0.0")
        return 0.0

try:
    from modules.search.pattern_manager import SearchPatternManager

    PATTERN_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SearchPatternManager not available: {e}")
    PATTERN_MANAGER_AVAILABLE = False

try:
    from modules.search.utils import (
        validate_search_parameters, extract_numeric_ids, extract_area_identifiers,
        extract_search_terms, log_search_performance, extract_part_numbers,
        normalize_part_number
    )

    UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Search utils not available: {e}")
    UTILS_AVAILABLE = False

try:
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher
    from spacy.tokens import Span

    HAS_SPACY = True
except ImportError:
    logger.warning("spaCy not available - NLP search will use fallback methods")
    HAS_SPACY = False

# Import database models
from modules.configuration.base import Base

# Import existing search models
try:
    from modules.search.models.search_models import SearchIntent
except ImportError:
    logger.warning("Could not import SearchIntent - using placeholder")


    class SearchIntent:
        pass



class SearchSession(Base):
    """Track user search sessions for analytics."""
    __tablename__ = 'search_session'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)
    session_token = Column(String(255), unique=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    total_queries = Column(Integer, default=0)
    successful_queries = Column(Integer, default=0)
    context_data = Column(JSON)
    is_active = Column(Boolean, default=True)

    # Relationships
    queries = relationship("SearchQuery", back_populates="session",
                           cascade="all, delete-orphan", passive_deletes=True)


class SearchQuery(Base):
    """Track individual search queries with comprehensive metadata."""
    __tablename__ = 'search_query'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('search_session.id'))
    parent_query_id = Column(Integer, ForeignKey('search_query.id'))

    # Query details
    query_text = Column(Text, nullable=False)
    normalized_query = Column(Text)
    detected_intent_id = Column(Integer, ForeignKey('search_intent.id'))
    intent_confidence = Column(Float)
    extracted_entities = Column(JSON)

    # Execution details
    search_method = Column(String(100))
    execution_time_ms = Column(Integer)
    result_count = Column(Integer)
    was_successful = Column(Boolean, default=False)

    # User feedback
    user_satisfaction_score = Column(Integer)  # 1-5 rating
    was_refined = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("SearchSession", back_populates="queries")
    child_queries = relationship("SearchQuery",
                                 backref="parent_query",
                                 remote_side=[id])


class SearchQueryTracker:
    """
    Comprehensive search query tracking and analytics system.
    Integrates with your existing search infrastructure to provide detailed insights.
    """

    def __init__(self, session: Session):
        self.session = session
        self._active_sessions = {}  # Cache for active sessions

    @with_request_id
    def _try_load_ml_model(self):
        """Try to load trained ML model"""
        try:
            active_model = self._session.query(MLModel).filter_by(
                model_type='intent_classifier',
                is_active=True
            ).first()

            if active_model and active_model.model_path:
                self.ml_classifier.load_model(active_model.model_path)
                logger.info("Loaded ML intent classifier")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")


    def start_search_session(self, user_id: str, context_data: Dict = None) -> Optional[int]:
        """Start a new search session for a user."""
        try:
            import uuid
            session_token = str(uuid.uuid4())

            search_session = SearchSession(
                user_id=user_id,
                session_token=session_token,
                context_data=context_data or {},
                started_at=datetime.utcnow(),
                is_active=True
            )

            self.session.add(search_session)
            self.session.commit()

            # Cache the session
            self._active_sessions[user_id] = search_session.id

            logger.info(f" Started search session {search_session.id} for user {user_id}")
            return search_session.id

        except Exception as e:
            logger.error(f" Failed to start search session for {user_id}: {e}")
            if self.session:
                try:
                    self.session.rollback()
                except:
                    pass
            return None


    def track_search_query(self, session_id: Optional[int], query_text: str,
                           detected_intent_id: Optional[int] = None,
                           intent_confidence: float = 0.0,
                           search_method: str = "unknown",
                           result_count: int = 0,
                           execution_time_ms: int = 0,
                           extracted_entities: Dict = None,
                           normalized_query: str = None,
                           parent_query_id: Optional[int] = None) -> Optional[int]:
        """Track a search query with comprehensive metadata."""
        try:
            was_successful = result_count > 0

            search_query = SearchQuery(
                session_id=session_id,
                parent_query_id=parent_query_id,
                query_text=query_text,
                normalized_query=normalized_query or query_text.lower().strip(),
                detected_intent_id=detected_intent_id,
                intent_confidence=intent_confidence,
                extracted_entities=extracted_entities or {},
                search_method=search_method,
                execution_time_ms=execution_time_ms,
                result_count=result_count,
                was_successful=was_successful,
                created_at=datetime.utcnow()
            )

            self.session.add(search_query)
            self.session.commit()

            # Update session statistics
            if session_id:
                self._update_session_stats(session_id, was_successful)

            logger.debug(f" Tracked query {search_query.id}: '{query_text}' -> {result_count} results")
            return search_query.id

        except Exception as e:
            logger.error(f" Failed to track search query '{query_text}': {e}")
            if self.session:
                try:
                    self.session.rollback()
                except:
                    pass
            return None


    def record_user_satisfaction(self, query_id: int, satisfaction_score: int) -> bool:
        """Record user satisfaction rating for a query (1-5 scale)."""
        try:
            if not (1 <= satisfaction_score <= 5):
                logger.warning(f"Invalid satisfaction score: {satisfaction_score}")
                return False

            query = self.session.query(SearchQuery).get(query_id)
            if query:
                query.user_satisfaction_score = satisfaction_score
                self.session.commit()

                logger.debug(f" Recorded satisfaction {satisfaction_score}/5 for query {query_id}")
                return True
            else:
                logger.warning(f"Query {query_id} not found for satisfaction recording")
                return False

        except Exception as e:
            logger.error(f" Failed to record satisfaction for query {query_id}: {e}")
            if self.session:
                try:
                    self.session.rollback()
                except:
                    pass
            return False


    def track_result_click(self, query_id: int, result_type: str, result_id: int,
                           click_position: int, action_taken: str = "view") -> bool:
        """Track when a user clicks on a search result."""
        try:
            click_record = SearchResultClick(
                query_id=query_id,
                result_type=result_type,
                result_id=result_id,
                click_position=click_position,
                action_taken=action_taken,
                created_at=datetime.utcnow()
            )

            self.session.add(click_record)
            self.session.commit()

            logger.debug(f" Tracked click on {result_type} {result_id} at position {click_position}")
            return True

        except Exception as e:
            logger.error(f" Failed to track result click: {e}")
            if self.session:
                try:
                    self.session.rollback()
                except:
                    pass
            return False


    def get_intent_id(self, intent_name: str) -> Optional[int]:
        """Get intent ID from intent name (integrates with your search_intent table)."""
        try:
            # Import your SearchIntent model
            from modules.search.models.search_models import SearchIntent

            intent = self.session.query(SearchIntent).filter_by(name=intent_name).first()
            if intent:
                return intent.id
            else:
                logger.debug(f"Intent '{intent_name}' not found in database")
                return None

        except Exception as e:
            logger.warning(f"Failed to get intent ID for '{intent_name}': {e}")
            return None


    def _update_session_stats(self, session_id: int, was_successful: bool):
        """Update session statistics."""
        try:
            session = self.session.query(SearchSession).get(session_id)
            if session:
                session.total_queries += 1
                if was_successful:
                    session.successful_queries += 1
                self.session.commit()

        except Exception as e:
            logger.warning(f"Failed to update session stats: {e}")


    def end_search_session(self, session_id: int) -> bool:
        """End a search session."""
        try:
            session = self.session.query(SearchSession).get(session_id)
            if session:
                session.ended_at = datetime.utcnow()
                session.is_active = False
                self.session.commit()

                # Remove from cache
                for user_id, cached_session_id in list(self._active_sessions.items()):
                    if cached_session_id == session_id:
                        del self._active_sessions[user_id]
                        break

                logger.info(f" Ended search session {session_id}")
                return True
            return False

        except Exception as e:
            logger.error(f" Failed to end search session {session_id}: {e}")
            return False


    def get_search_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive search performance report."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Basic query statistics
            total_queries = self.session.query(SearchQuery).filter(
                SearchQuery.created_at >= cutoff_date
            ).count()

            successful_queries = self.session.query(SearchQuery).filter(
                SearchQuery.created_at >= cutoff_date,
                SearchQuery.was_successful == True
            ).count()

            # Average execution time
            avg_execution_time = self.session.execute(text("""
                SELECT AVG(execution_time_ms) as avg_time
                FROM search_query 
                WHERE created_at >= :cutoff_date
                AND execution_time_ms > 0
            """), {'cutoff_date': cutoff_date}).scalar() or 0

            # Top search methods
            top_methods = self.session.execute(text("""
                SELECT search_method, COUNT(*) as count
                FROM search_query 
                WHERE created_at >= :cutoff_date
                GROUP BY search_method
                ORDER BY count DESC
                LIMIT 10
            """), {'cutoff_date': cutoff_date}).fetchall()

            # User satisfaction
            avg_satisfaction = self.session.execute(text("""
                SELECT AVG(user_satisfaction_score) as avg_satisfaction
                FROM search_query 
                WHERE created_at >= :cutoff_date
                AND user_satisfaction_score IS NOT NULL
            """), {'cutoff_date': cutoff_date}).scalar() or 0

            # Most common queries
            top_queries = self.session.execute(text("""
                SELECT normalized_query, COUNT(*) as frequency
                FROM search_query 
                WHERE created_at >= :cutoff_date
                GROUP BY normalized_query
                ORDER BY frequency DESC
                LIMIT 10
            """), {'cutoff_date': cutoff_date}).fetchall()

            # Intent breakdown
            intent_breakdown = self.session.execute(text("""
                SELECT si.name as intent_name, COUNT(sq.id) as count
                FROM search_query sq
                LEFT JOIN search_intent si ON sq.detected_intent_id = si.id
                WHERE sq.created_at >= :cutoff_date
                GROUP BY si.name
                ORDER BY count DESC
            """), {'cutoff_date': cutoff_date}).fetchall()

            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0

            report = {
                'period_days': days,
                'report_generated_at': datetime.utcnow().isoformat(),
                'summary': {
                    'total_queries': total_queries,
                    'successful_queries': successful_queries,
                    'success_rate_percent': round(success_rate, 2),
                    'average_execution_time_ms': round(avg_execution_time, 2),
                    'average_satisfaction_score': round(avg_satisfaction, 2)
                },
                'top_search_methods': [
                    {'method': row[0], 'count': row[1]} for row in top_methods
                ],
                'most_common_queries': [
                    {'query': row[0], 'frequency': row[1]} for row in top_queries
                ],
                'intent_breakdown': [
                    {'intent': row[0] or 'UNKNOWN', 'count': row[1]} for row in intent_breakdown
                ]
            }

            logger.info(f" Generated performance report: {success_rate:.1f}% success rate over {days} days")
            return report

        except Exception as e:
            logger.error(f" Failed to generate performance report: {e}")
            return {
                'error': str(e),
                'period_days': days,
                'report_generated_at': datetime.utcnow().isoformat()
            }  # <-- FIXED: Added missing closing brace here


class MLModel(Base):
    __tablename__ = 'ml_model'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    model_type = Column(String(50))  # 'intent_classifier', 'entity_extractor', 'similarity'
    version = Column(String(20))
    model_path = Column(Text)
    training_data_hash = Column(String(64))
    accuracy_score = Column(Float)
    is_active = Column(Boolean, default=False)
    deployed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserFeedback(Base):
    __tablename__ = 'user_feedback'
    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey('search_query.id'))
    user_id = Column(String(100))
    feedback_type = Column(String(50))  # 'relevance', 'intent_correct', 'result_useful'
    feedback_value = Column(String(500))
    rating = Column(Integer)  # 1-5
    created_at = Column(DateTime, default=datetime.utcnow)

class SearchSessionManager:
    """Track user search sessions for learning"""

    def __init__(self, session):
        self.session = session


    def start_session(self, user_id: str, context: Dict) -> str:
        session_token = str(uuid4())

        search_session = SearchSession(
            user_id=user_id,
            session_token=session_token,
            context_data=context,
            started_at=datetime.utcnow()
        )
        self.session.add(search_session)
        self.session.commit()
        return session_token


    def log_query(self, session_token: str, query: str, result: Dict):
        """Log individual queries for learning"""
        search_session = self.session.query(SearchSession).filter_by(
            session_token=session_token
        ).first()

        if search_session:
            query_record = SearchQuery(
                session_id=search_session.id,
                query_text=query,
                detected_intent_id=result.get('intent_id'),
                intent_confidence=result.get('confidence'),
                result_count=result.get('count', 0),
                execution_time_ms=result.get('execution_time_ms'),
                created_at=datetime.utcnow()
            )
            self.session.add(query_record)
            self.session.commit()


class IntentClassifierML:
    """Machine learning intent classifier"""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.vectorizer = None
        if model_path:
            self.load_model(model_path)


    def load_model(self, model_path: str):
        """Load trained model from file"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.vectorizer = model_data.get('vectorizer')
            logger.info(f"Loaded ML model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")


    def save_model(self, model_path: str):
        """Save trained model to file"""
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Saved ML model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {e}")


    def train_from_database(self, session):
        """Train model from SearchQuery historical data"""
        # Get training data
        queries = session.query(SearchQuery).filter(
            SearchQuery.intent_confidence > 0.8  # High confidence samples
        ).all()

        if len(queries) < 100:
            logger.warning("Insufficient training data")
            return False

        # Prepare training data
        X = [q.query_text for q in queries]
        y = [q.detected_intent_id for q in queries]

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline

            # Create and train pipeline
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', MultinomialNB())
            ])

            self.model.fit(X, y)
            logger.info(f"Trained model on {len(queries)} samples")
            return True
        except ImportError:
            logger.error("scikit-learn not available for ML training")
            return False


    def predict_intent(self, query: str) -> Tuple[str, float]:
        """Predict intent with confidence"""
        if not self.model:
            return "UNKNOWN", 0.0

        try:
            prediction = self.model.predict([query])[0]
            probabilities = self.model.predict_proba([query])[0]
            confidence = max(probabilities)
            return prediction, confidence
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return "UNKNOWN", 0.0


class FeedbackLearner:
    """Learn from user feedback to improve search"""

    def __init__(self, session):
        self.session = session

    def record_feedback(self, query_id: int, feedback_type: str, rating: int):
        """Record user feedback"""
        feedback = UserFeedback(
            query_id=query_id,
            feedback_type=feedback_type,
            rating=rating,
            created_at=datetime.utcnow()
        )
        self.session.add(feedback)
        self.session.commit()

    def learn_from_clicks(self, query_id: int, clicked_results: List[Dict]):
        """Learn from which results users actually clicked"""
        for i, result in enumerate(clicked_results):
            click_record = SearchResultClick(
                query_id=query_id,
                result_type=result['type'],
                result_id=result['id'],
                click_position=i,
                created_at=datetime.utcnow()
            )
            self.session.add(click_record)
        self.session.commit()

    def get_popular_patterns(self, days: int = 30) -> List[Dict]:
        """Get popular search patterns for pattern template generation"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        popular_queries = self.session.query(
            SearchQuery.query_text,
            func.count(SearchQuery.id).label('frequency'),
            func.avg(SearchQuery.user_satisfaction_score).label('avg_satisfaction')
        ).filter(
            SearchQuery.created_at >= cutoff_date
        ).group_by(
            SearchQuery.query_text
        ).having(
            func.count(SearchQuery.id) > 5  # At least 5 occurrences
        ).order_by(
            func.count(SearchQuery.id).desc()
        ).limit(50).all()

        return [
            {
                "query": q.query_text,
                "frequency": q.frequency,
                "satisfaction": q.avg_satisfaction or 0
            }
            for q in popular_queries
        ]


# ===== DATABASE PATTERN INTEGRATION MIXIN =====

# ===== MAIN ENHANCED SPACY SEARCH CLASS =====
class SpaCyEnhancedAggregateSearch():
    """
    COMPLETE Enhanced NLP-enhanced aggregate search with database pattern integration.

    This replaces the original SpaCyEnhancedAggregateSearch with all functionality plus:
    - Database pattern integration (your 155+ patterns)
    - Automatic learning from pattern success rates
    - Enhanced parameter extraction using proven regex patterns
    - Fixed synonym loading and transaction handling
    - Performance monitoring and optimization suggestions
    - Backward compatible with existing functionality
    """

    def __init__(self, session: Optional[Session] = None, nlp_instance=None, user_context: Optional[Dict] = None):
        # Initialize database pattern integration mixin first
        DatabasePatternIntegrationMixin.__init__(self)

        self._session = session
        self.user_context = user_context or {}

        # Initialize parent functionality
        if AGGREGATE_SEARCH_AVAILABLE:
            try:
                self._aggregate_search = AggregateSearch(session=session)
                logger.info("AggregateSearch backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AggregateSearch: {e}")
                self._aggregate_search = None
        else:
            self._aggregate_search = None

        # Initialize spaCy
        self.nlp = None
        self.has_spacy = HAS_SPACY
        if HAS_SPACY:
            if nlp_instance:
                self.nlp = nlp_instance
                logger.info("Using provided spaCy instance")
            else:
                try:
                    self.nlp = spacy.load('en_core_web_sm')
                    logger.info("Loaded spaCy model: en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model 'en_core_web_sm' not found - using fallback methods")
                    self.has_spacy = False

        # Initialize pattern manager
        if PATTERN_MANAGER_AVAILABLE:
            try:
                self.pattern_manager = SearchPatternManager(session=session)
                logger.debug("Initialized SearchPatternManager")
            except Exception as e:
                logger.warning(f"Could not initialize SearchPatternManager: {e}")
                self.pattern_manager = None
        else:
            self.pattern_manager = None

        # Initialize NLP components
        if self.has_spacy and self.nlp:
            self.setup_domain_patterns()
            self.setup_intent_classification()
        else:
            logger.info("Operating in fallback mode without spaCy")

        # Analysis cache
        self._analysis_cache = {}
        self._cache_max_size = 100

        # Load entity synonyms with fixed method
        self.entity_synonyms = self._load_entity_synonyms_fixed()

        logger.info(f" Enhanced SpaCy search initialized with database integration")

    def _load_entity_synonyms(self) -> Dict[str, Dict[str, str]]:
        """ENHANCED: Load entity synonyms from the database (calls fixed version)."""
        return self._load_entity_synonyms_fixed()

    @with_request_id
    def setup_domain_patterns(self):
        """Set up domain-specific patterns, enhanced with database integration."""
        if not self.nlp:
            return

        try:
            # Create entity ruler
            if "entity_ruler" not in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            else:
                ruler = self.nlp.get_pipe("entity_ruler")

            # Load entity types and their regex patterns from database
            patterns = []
            if self._session:
                try:
                    entity_types = self._session.execute(select(EntityType)).scalars().all()
                    for et in entity_types:
                        if et.validation_regex:
                            patterns.append({
                                "label": et.name,
                                "pattern": [{"TEXT": {"REGEX": et.validation_regex}}]
                            })
                except Exception as e:
                    logger.warning(f"Error loading entity types: {e}")

            # Existing patterns
            patterns.extend([
                {"label": "PART_NUMBER", "pattern": [{"TEXT": {"REGEX": r"^[A-Z0-9]{2,}[-\.][A-Z0-9]+$"}}]},
                {"label": "PART_NUMBER", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,}\d{3,}$"}}]},
                {"label": "PART_NUMBER", "pattern": [{"TEXT": {"REGEX": r"^\d{4,}[-\.][A-Z0-9]+$"}}]},
                {"label": "PART_NUMBER", "pattern": [{"TEXT": {"REGEX": r"^\d{5,}$"}}]},
                {"label": "AREA_ID", "pattern": [{"LOWER": "area"}, {"TEXT": {"REGEX": r"^[A-Z0-9]{1,3}$"}}]},
                {"label": "EQUIPMENT_TYPE", "pattern": [
                    {"LOWER": {"IN": ["centrifugal", "rotary", "electric", "hydraulic"]}, "OP": "?"},
                    {"LOWER": {"IN": ["pump", "motor", "compressor", "valve", "bearing", "filter", "sensor"]}}
                ]},
                {"label": "MAINTENANCE_ACTION", "pattern": [
                    {"LOWER": {"IN": ["repair", "replace", "maintain", "fix", "install", "remove", "clean", "inspect"]}}
                ]},
                {"label": "PROBLEM_TYPE", "pattern": [
                    {"LOWER": {"IN": ["leak", "noise", "vibration", "failure", "malfunction", "broken", "damaged"]}}
                ]}
            ])

            ruler.add_patterns(patterns)

            # Set up phrase matcher with synonyms
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            equipment_phrases = {
                "PUMP_TYPES": ["centrifugal pump", "diaphragm pump"],
                "MOTOR_TYPES": ["electric motor", "servo motor"],
                "VALVE_TYPES": ["ball valve", "gate valve"],
                "BEARING_TYPES": ["ball bearing", "roller bearing"],
                "COMMON_PROBLEMS": ["high vibration", "oil leak"]
            }

            # Add synonyms to phrase matcher
            for entity_type, synonym_dict in self.entity_synonyms.items():
                if entity_type in equipment_phrases:
                    for synonym in synonym_dict:
                        equipment_phrases[entity_type].append(synonym)

            for label, phrases in equipment_phrases.items():
                patterns = [self.nlp(phrase) for phrase in phrases]
                self.phrase_matcher.add(label, patterns)

            logger.debug("Enhanced domain patterns and phrase matchers set up successfully")
        except Exception as e:
            logger.error(f"Error setting up enhanced domain patterns: {e}")

    @with_request_id
    def setup_intent_classification(self):
        """ENHANCED: Set up intent classification using your database patterns."""
        if not self.nlp:
            return

        try:
            self.intent_matcher = Matcher(self.nlp.vocab)

            # ENHANCED: Load patterns from your database instead of hardcoded ones
            if self._session:
                try:
                    database_patterns = self._load_database_patterns()

                    spacy_pattern_count = 0
                    for intent_name, intent_data in database_patterns.items():
                        spacy_patterns = []
                        for pattern_info in intent_data['patterns']:
                            # Convert simple patterns to spaCy format (complex ones handled separately)
                            spacy_pattern = self._regex_to_spacy_pattern(pattern_info['pattern'])
                            if spacy_pattern:
                                spacy_patterns.append(spacy_pattern)

                        if spacy_patterns:
                            self.intent_matcher.add(intent_name, spacy_patterns)
                            spacy_pattern_count += len(spacy_patterns)

                    logger.info(
                        f" Loaded {spacy_pattern_count} spaCy patterns from database for {len(database_patterns)} intents")

                except Exception as e:
                    logger.warning(f"Could not load database patterns for spaCy: {e}")
                    self._setup_fallback_intent_patterns()
            else:
                self._setup_fallback_intent_patterns()

        except Exception as e:
            logger.error(f"Error setting up enhanced intent classification: {e}")

    @with_request_id
    def _regex_to_spacy_pattern(self, regex_pattern: str) -> Optional[List[Dict]]:
        """Convert simple regex patterns to spaCy patterns (basic conversion)."""
        try:
            # Handle simple word-based patterns
            if regex_pattern.count('.*') <= 2 and not any(
                    char in regex_pattern for char in ['(', ')', '[', ']', '{', '}']):
                # Pattern like "find.*part" or "show.*images"
                words = re.sub(r'\.\*', ' ', regex_pattern).strip().split()
                if len(words) <= 4:  # Keep it simple
                    return [{"LOWER": word} for word in words if word and len(word) > 1]

            # For complex regex patterns, return None - we'll handle them in database classification
            return None

        except Exception as e:
            logger.warning(f"Could not convert regex pattern '{regex_pattern}': {e}")
            return None

    def _setup_fallback_intent_patterns(self):
        """Enhanced fallback intent patterns."""
        if not self.nlp:
            return

        fallback_patterns = {
            "FIND_PART": [
                [{"LOWER": {"IN": ["find", "search", "show", "get"]}},
                 {"LOWER": {"IN": ["part", "component", "spare"]}}],
                [{"LOWER": "part"}, {"TEXT": {"REGEX": r"^\d+$"}}],
                [{"LOWER": {"IN": ["where", "locate"]}}, {"LOWER": "part"}],
                [{"TEXT": {"REGEX": r"^#?\d{4,}$"}}]
            ],
            "SHOW_IMAGES": [
                [{"LOWER": {"IN": ["show", "display", "find"]}},
                 {"LOWER": {"IN": ["image", "images", "picture", "pictures", "photo", "photos"]}}],
                [{"LOWER": {"IN": ["image", "picture", "photo"]}}, {"LOWER": "of"}]
            ],
            "LOCATION_SEARCH": [
                [{"TEXT": {"REGEX": r"what'?s"}}, {"LOWER": "in"}],
                [{"LOWER": "equipment"}, {"LOWER": {"IN": ["in", "at"]}}]
            ],
            "MAINTENANCE_PROCEDURE": [
                [{"LOWER": {"IN": ["how", "procedure", "steps"]}},
                 {"LOWER": {"IN": ["maintenance", "repair", "replace"]}}]
            ]
        }

        for intent, patterns in fallback_patterns.items():
            self.intent_matcher.add(intent, patterns)

        logger.debug("Enhanced fallback intent patterns set up")

    def execute_nlp_aggregated_search(self, user_input: str) -> Dict[str, Any]:
        """ENHANCED: Execute aggregated search with database pattern integration."""
        start_time = datetime.utcnow()

        try:
            # Analyze input with enhanced NLP
            analysis = self.analyze_user_input(user_input)

            if analysis["confidence_score"] < 0.3:
                return {
                    "status": "low_confidence",
                    "message": "I'm not sure what you're looking for. Could you be more specific?",
                    "analysis": analysis,
                    "suggestions": self._get_suggestion_examples(),
                    "input": user_input
                }

            # Resolve hierarchical intent
            final_intent = self._resolve_intent_hierarchy(analysis["intent"]["intent"])
            analysis["intent"]["resolved_intent"] = final_intent

            # Apply context boosts
            context_boost = self._apply_context_boost(final_intent)
            analysis["intent"]["confidence"] *= context_boost

            # If AggregateSearch is available, use it
            if self._aggregate_search:
                intent_to_method = {
                    "FIND_PART": "comprehensive_part_search",
                    "FIND_PART_BY_NUMBER": "comprehensive_part_search",
                    "SHOW_IMAGES": "comprehensive_image_search",
                    "LOCATION_SEARCH": "comprehensive_position_search",
                    "MAINTENANCE_PROCEDURE": "comprehensive_task_search"
                }

                search_method_name = intent_to_method.get(final_intent)
                if search_method_name:
                    search_method = getattr(self._aggregate_search, search_method_name, None)
                    if search_method:
                        search_params = validate_search_parameters(
                            analysis["search_parameters"]) if UTILS_AVAILABLE else analysis["search_parameters"]
                        search_params.setdefault("limit", 10)
                        result = search_method(search_params)

                        # ENHANCED: Update pattern statistics
                        self.update_pattern_usage(analysis["intent"], result)
                    else:
                        result = {
                            "status": "error",
                            "message": f"Search method {search_method_name} not available"
                        }
                else:
                    result = {
                        "status": "error",
                        "message": f"Don't know how to handle intent: {final_intent}"
                    }
            else:
                result = self._fallback_search(analysis)

            # Log performance
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            if UTILS_AVAILABLE:
                log_search_performance(
                    search_type="enhanced_nlp_aggregated",
                    execution_time_ms=int(execution_time),
                    result_count=result.get("count", 0),
                    user_input=user_input,
                    success=result.get("status") == "success"
                )

            # Add enhanced metadata
            result.update({
                "nlp_analysis": {
                    "detected_intent": analysis["intent"]["intent"],
                    "resolved_intent": final_intent,
                    "intent_confidence": analysis["intent"]["confidence"],
                    "overall_confidence": analysis["confidence_score"],
                    "extracted_entities": analysis["entities"],
                    "processing_method": analysis["processing_method"],
                    "database_patterns_used": analysis["intent"].get("method") == "database_patterns",
                    "extraction_method": analysis["search_parameters"].get("extraction_method"),
                    "pattern_details": {
                        "matched_pattern": analysis["intent"].get("matched_pattern"),
                        "priority": analysis["intent"].get("priority"),
                        "success_rate": analysis["intent"].get("success_rate"),
                        "usage_count": analysis["intent"].get("usage_count")
                    } if analysis["intent"].get("method") == "database_patterns" else None
                },
                "natural_language_input": user_input,
                "search_type": "enhanced_nlp_aggregated",
                "execution_time_ms": int(execution_time),
                "spacy_available": self.has_spacy,
                "aggregate_search_available": self._aggregate_search is not None,
                "database_patterns_count": len(self._database_patterns),
                "enhancement_level": "database_integrated"
            })

            return result
        except Exception as e:
            logger.error(f"Error in enhanced NLP aggregated search: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error processing request: {str(e)}",
                "input": user_input,
                "error_type": "enhanced_nlp_search_error"
            }

    def _resolve_intent_hierarchy(self, detected_intent: str) -> str:
        """Resolve intent hierarchy to find the most specific intent."""
        if not self._session:
            return detected_intent

        try:
            # Find child intents
            query = select(SearchIntentHierarchy).filter(
                SearchIntentHierarchy.parent_intent_id.in_(
                    select(SearchIntent.id).filter(SearchIntent.name == detected_intent)
                )
            )
            hierarchies = self._session.execute(query).scalars().all()

            if not hierarchies:
                return detected_intent

            # Select the child intent with highest priority modifier
            best_child = max(hierarchies, key=lambda x: x.priority_modifier, default=None)
            if best_child:
                child_intent = self._session.get(SearchIntent, best_child.child_intent_id)
                return child_intent.name if child_intent else detected_intent

            return detected_intent
        except Exception as e:
            logger.warning(f"Error resolving intent hierarchy: {e}")
            return detected_intent

    def _apply_context_boost(self, intent: str) -> float:
        """Apply context-based confidence boost."""
        if not self._session or not self.user_context:
            return 1.0

        try:
            query = select(IntentContext).join(SearchIntent).filter(SearchIntent.name == intent)
            contexts = self._session.execute(query).scalars().all()

            boost = 1.0
            for context in contexts:
                if context.context_type in self.user_context:
                    if str(self.user_context[context.context_type]).lower() == context.context_value.lower():
                        boost *= context.boost_factor
                        if context.is_required:
                            boost *= 1.5  # Extra boost for required contexts
            return boost
        except Exception as e:
            logger.warning(f"Error applying context boost: {e}")
            return 1.0

    def _classify_intent(self, doc):
        """ENHANCED: Intent classification using database patterns first."""

        # ENHANCED: First try database regex patterns (more accurate)
        database_result = self._classify_intent_with_database(doc.text)
        if database_result['confidence'] > 0.6:  # Lower threshold for database patterns
            logger.debug(
                f" Database pattern matched: {database_result['intent']} (confidence: {database_result['confidence']:.2f})")
            return database_result

        # Fall back to spaCy pattern matching
        if hasattr(self, 'intent_matcher'):
            spacy_result = self._classify_intent_with_spacy(doc)
            if spacy_result['confidence'] > database_result['confidence']:
                logger.debug(
                    f" SpaCy pattern matched: {spacy_result['intent']} (confidence: {spacy_result['confidence']:.2f})")
                return spacy_result

        # Final fallback
        if database_result['confidence'] > 0:
            return database_result
        else:
            return self._fallback_intent_classification(doc)

    def _classify_intent_with_spacy(self, doc) -> Dict[str, Any]:
        """Original spaCy-based classification as fallback."""
        if not hasattr(self, 'intent_matcher'):
            return {"intent": "UNKNOWN", "confidence": 0.0, "method": "no_spacy_matcher"}

        matches = self.intent_matcher(doc)
        if not matches:
            return {"intent": "UNKNOWN", "confidence": 0.0, "method": "no_spacy_match"}

        intent_scores = {}
        for match_id, start, end in matches:
            intent_name = self.nlp.vocab.strings[match_id]
            span_length = end - start
            score = span_length / len(doc)
            intent_scores[intent_name] = {
                "score": score,
                "matched_span": doc[start:end].text,
                "start": start,
                "end": end
            }

        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1]["score"])
            intent_name, intent_data = best_intent
            return {
                "intent": intent_name,
                "confidence": min(intent_data["score"] * 1.5, 1.0),  # Slight boost
                "matched_text": intent_data["matched_span"],
                "method": "spacy_pattern_matching",
                "all_matches": list(intent_scores.keys())
            }

        return {"intent": "UNKNOWN", "confidence": 0.0, "method": "no_spacy_match"}

    def _fallback_intent_classification(self, doc):
        """Fallback intent classification with keyword similarity."""
        text_lower = doc.text.lower()
        intent_keywords = {
            "FIND_PART": ["part", "component", "spare", "find", "search", "#"],
            "SHOW_IMAGES": ["image", "picture", "photo", "show", "display"],
            "LOCATION_SEARCH": ["where", "location", "area", "in", "at"],
            "MAINTENANCE_PROCEDURE": ["how", "procedure", "steps", "maintenance"]
        }

        best_intent = "UNKNOWN"
        best_score = 0.0
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
            if score > best_score:
                best_score = score
                best_intent = intent

        return {
            "intent": best_intent,
            "confidence": best_score,
            "method": "keyword_fallback"
        }

    def analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """Comprehensive NLP analysis with enhanced database integration."""
        cache_key = user_input.lower().strip()
        if cache_key in self._analysis_cache:
            cached_result = self._analysis_cache[cache_key]
            cached_result["from_cache"] = True
            return cached_result

        if not self.has_spacy or not self.nlp:
            return self._fallback_analysis(user_input)

        try:
            doc = self.nlp(user_input)
            analysis = {
                "original_text": user_input,
                "entities": self._extract_all_entities(doc),
                "intent": self._classify_intent(doc),
                "semantic_info": self._extract_semantic_info(doc),
                "search_parameters": {},
                "confidence_score": 0.0,
                "processing_method": "enhanced_spacy_nlp",
                "timestamp": datetime.utcnow().isoformat()
            }

            analysis["search_parameters"] = self._build_search_parameters(analysis, doc)
            analysis["confidence_score"] = self._calculate_confidence(analysis)
            self._add_to_analysis_cache(cache_key, analysis)
            return analysis
        except Exception as e:
            logger.error(f"Error in enhanced NLP analysis: {e}")
            return self._fallback_analysis(user_input)

    def _build_search_parameters(self, analysis, doc):
        """FIXED: Enhanced parameter building with proper part_candidate handling."""
        params = {"raw_input": doc.text}
        entities = analysis["entities"]

        # ENHANCED: Use database pattern extraction if available
        intent_info = analysis.get("intent", {})
        if intent_info.get("method") == "database_patterns":
            # Extract parameters using the matched database pattern
            extracted_params = self._extract_params_from_database_pattern(
                doc.text,
                intent_info.get("intent"),
                intent_info.get("matched_pattern"),
                intent_info.get("extracted_groups", ())
            )
            params.update(extracted_params)

            if extracted_params:
                logger.debug(f" Database pattern extracted: {extracted_params}")
                return params

        # Fall back to enhanced extraction logic
        text = doc.text.lower()

        # FIXED: Enhanced patterns that properly handle descriptions vs part numbers
        part_patterns = [
            # DESCRIPTION PATTERNS (highest priority)
            r'(?:i\s+)?need\s+(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',  # "I need part number for BEARING"
            r'what\s+(?:is\s+)?(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',
            # "what is part number for VALVE..."
            r'part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',  # "part number for VALVE BYPASS..."

            # DIRECT PART NUMBER PATTERNS (lower priority)
            r'(?:find|search|show|get)\s+part\s+number\s+([A-Za-z0-9\-\.]+)(?:\s|$)',  # "find part number A115957"
            r'(?:find|search|show|get)\s+part\s+([A-Za-z0-9\-\.]+)(?:\s|$)',  # "find part A115957"
            r'part\s+number\s+([A-Za-z0-9\-\.]+)(?:\s|$)',  # "part number A115957"
            r'part\s+([A-Za-z0-9\-\.]+)(?:\s|$)',  # "part A115957"
            r'\b([A-Za-z]\d{5,})\b',  # "A115957" standalone
            r'\b(\d{5,})\b'  # "115957" standalone
        ]

        for i, pattern in enumerate(part_patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()

                if i <= 2:  # Description patterns (0, 1, 2)
                    # This is a description request
                    if len(extracted.strip()) > 0:
                        # Clean up the description
                        description = re.sub(r'\s+', ' ', extracted.strip())
                        params.update({
                            "search_text": description,
                            "entity_type": "part",
                            "fields": ['name', 'part_number', 'oem_mfg', 'model', 'notes'],
                            "extraction_method": "enhanced_regex_description"
                        })
                        logger.debug(f" Extracted description search: '{description}'")
                        break

                elif i <= 7:  # Direct part number patterns (3-7)
                    # This should be a direct part number
                    if (re.match(r'^[A-Za-z0-9\-\.]+$', extracted) and
                            extracted.upper() not in ['FOR', 'THE', 'A', 'AN', 'OF', 'IN', 'ON', 'AT', 'TO', 'NUMBER',
                                                      'PART'] and
                            len(extracted) >= 3):
                        params.update({
                            "part_number": extracted.upper(),
                            "entity_type": "part",
                            "extraction_method": "enhanced_regex_direct"
                        })
                        logger.debug(f" Extracted part number: {extracted.upper()}")
                        break

        # Extract other entities (areas, equipment, numbers) - unchanged
        if "part_number" not in params and "search_text" not in params and entities["part_numbers"]:
            canonical_text = entities["part_numbers"][0].get("canonical_text")
            entity_text = entities["part_numbers"][0]["text"]
            part_candidate = canonical_text if canonical_text else entity_text

            if (part_candidate and
                    part_candidate.upper() not in ['NUMBER', 'PART', 'FOR', 'THE'] and
                    len(part_candidate) >= 3):
                params["part_number"] = part_candidate.upper()
                params["extraction_method"] = "entity_extraction"

        if entities["areas"]:
            area_text = entities["areas"][0]["text"]
            area_match = re.search(r'([A-Z0-9]+)', area_text.upper())
            if area_match:
                params["area"] = area_match.group(1)

        if entities["equipment"]:
            equipment_text = entities["equipment"][0].get(
                "canonical_text", entities["equipment"][0]["text"]
            ).lower()
            params["equipment"] = equipment_text

        if "part_number" not in params and "search_text" not in params and entities["numbers"]:
            for num_entity in entities["numbers"]:
                if num_entity.get("numeric_value") and num_entity["numeric_value"] > 10:
                    if num_entity["numeric_value"] >= 10000:
                        params.update({
                            "part_number": str(num_entity["numeric_value"]),
                            "entity_type": "part",
                            "extraction_method": "numeric_entity"
                        })
                    else:
                        params["extracted_id"] = num_entity["numeric_value"]
                    break

        if "entity_type" not in params:
            params["entity_type"] = "part"

        logger.debug(f" Final search parameters: {params}")
        return params

    def _fallback_search(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced fallback search when AggregateSearch is not available."""
        try:
            entities = analysis.get("entities", {})
            search_params = analysis.get("search_parameters", {})

            # Try to extract searchable value
            search_value = None
            if "part_number" in search_params:
                search_value = search_params["part_number"]
            elif "search_text" in search_params:
                search_value = search_params["search_text"]
            elif entities.get("part_numbers"):
                search_value = entities["part_numbers"][0]["text"]
            elif entities.get("numbers"):
                search_value = str(entities["numbers"][0].get("numeric_value", entities["numbers"][0]["text"]))

            if search_value:
                try:
                    from modules.emtacdb.emtacdb_fts import Part
                    from modules.configuration.config_env import DatabaseConfig

                    db_config = DatabaseConfig()
                    session = db_config.get_main_session()

                    # Try different search strategies
                    parts = []
                    if "part_number" in search_params:
                        parts = Part.search(part_number=search_value, session=session)
                    elif "search_text" in search_params:
                        parts = Part.search(search_text=search_value, session=session)

                    if not parts and search_value.isdigit():
                        parts = Part.search(part_id=int(search_value), session=session)

                    if parts:
                        part = parts[0]
                        return {
                            "status": "success",
                            "count": 1,
                            "results": [{
                                "id": part.id,
                                "part_number": part.part_number,
                                "name": part.name,
                                "oem_mfg": part.oem_mfg,
                                "model": part.model,
                                "type": "part"
                            }],
                            "entity_type": "part",
                            "search_method": "enhanced_fallback_direct_part_search"
                        }
                    else:
                        return {
                            "status": "no_results",
                            "message": f"No part found with identifier: {search_value}",
                            "searched_for": search_value
                        }
                except Exception as e:
                    logger.error(f"Enhanced fallback part search failed: {e}")
                    return {
                        "status": "error",
                        "message": f"Enhanced fallback search failed: {str(e)}"
                    }

            return {
                "status": "no_results",
                "message": "Could not extract searchable information from your query",
                "suggestion": "Try asking about a specific part number like 'A115957' or 'part number for bearing assembly'"
            }
        except Exception as e:
            logger.error(f"Error in enhanced fallback search: {e}")
            return {
                "status": "error",
                "message": f"Enhanced fallback search error: {str(e)}"
            }

    def _fallback_analysis(self, user_input: str) -> Dict[str, Any]:
        """Enhanced fallback analysis with database pattern support."""

        # First try database patterns even without spaCy
        if self._session:
            database_result = self._classify_intent_with_database(user_input)
            if database_result['confidence'] > 0.5:
                # Extract parameters using database pattern
                extracted_params = self._extract_params_from_database_pattern(
                    user_input,
                    database_result.get("intent"),
                    database_result.get("matched_pattern"),
                    database_result.get("extracted_groups", ())
                )

                if extracted_params:
                    logger.debug(" Database fallback extraction successful")
                    return {
                        "original_text": user_input,
                        "entities": {"extracted_from_db": True},
                        "intent": database_result,
                        "semantic_info": {"database_extraction": True},
                        "search_parameters": {**extracted_params, "raw_input": user_input},
                        "confidence_score": database_result['confidence'],
                        "processing_method": "database_fallback",
                        "timestamp": datetime.utcnow().isoformat()
                    }

        # Original fallback logic
        if UTILS_AVAILABLE:
            numeric_ids = extract_numeric_ids(user_input)
            area_ids = extract_area_identifiers(user_input)
            search_terms = extract_search_terms(user_input)
        else:
            numeric_ids = [int(m) for m in re.findall(r'\b(\d{4,})\b', user_input)]
            area_ids = re.findall(r'\b(?:area|zone)\s+([A-Z0-9]+)\b', user_input, re.IGNORECASE)
            search_terms = re.findall(r'\b[a-zA-Z]{3,}\b', user_input.lower())

        # Enhanced intent classification
        text_lower = user_input.lower()
        intent = "UNKNOWN"
        confidence = 0.0

        if any(word in text_lower for word in ["part", "component", "#"]) or numeric_ids:
            intent = "FIND_PART"
            confidence = 0.8
        elif any(word in text_lower for word in ["image", "picture", "photo", "show"]):
            intent = "SHOW_IMAGES"
            confidence = 0.7
        elif any(word in text_lower for word in ["where", "what's in", "location"]):
            intent = "LOCATION_SEARCH"
            confidence = 0.6
        elif any(word in text_lower for word in ["similar", "like", "compare"]):
            intent = "SIMILARITY_SEARCH"
            confidence = 0.6

        # Enhanced entity extraction
        entities = {
            "part_numbers": [],
            "areas": [{"text": area, "confidence": 0.8} for area in area_ids],
            "equipment": [],
            "image_refs": [],
            "numbers": [{"text": str(num), "numeric_value": num, "confidence": 0.7} for num in numeric_ids],
            "general": []
        }

        # Enhanced part number patterns
        part_patterns = [
            r'\b([A-Z0-9]{2,}[-\.][A-Z0-9]+)\b',
            r'\b([A-Z]{2,}\d{3,})\b',
            r'\b(\d{5,})\b'
        ]
        for pattern in part_patterns:
            matches = re.findall(pattern, user_input.upper())
            for match in matches:
                entities["part_numbers"].append({
                    "text": match,
                    "confidence": 0.8,
                    "source": "fallback_regex"
                })

        # Enhanced parameter building
        params = {"raw_input": user_input}
        if entities["part_numbers"]:
            params["part_number"] = entities["part_numbers"][0]["text"]
        if entities["areas"]:
            params["area"] = entities["areas"][0]["text"]
        if numeric_ids:
            params["extracted_id"] = numeric_ids[0]

        return {
            "original_text": user_input,
            "entities": entities,
            "intent": {"intent": intent, "confidence": confidence, "method": "enhanced_fallback_regex"},
            "semantic_info": {"search_terms": search_terms},
            "search_parameters": params,
            "confidence_score": confidence,
            "processing_method": "enhanced_fallback_regex",
            "timestamp": datetime.utcnow().isoformat()
        }

    def _extract_all_entities(self, doc):
        """Extract entities with enhanced synonym normalization."""
        entities = {
            "part_numbers": [],
            "areas": [],
            "equipment": [],
            "image_refs": [],
            "doc_refs": [],
            "actions": [],
            "problems": [],
            "numbers": [],
            "general": []
        }

        for ent in doc.ents:
            entity_data = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 0.9
            }

            # Enhanced synonym normalization
            if ent.label_ in self.entity_synonyms:
                text_lower = ent.text.lower()
                if text_lower in self.entity_synonyms[ent.label_]:
                    synonym_data = self.entity_synonyms[ent.label_][text_lower]
                    entity_data["canonical_text"] = synonym_data["canonical"]
                    entity_data["confidence"] *= synonym_data["confidence"]

            # Categorize entities
            if ent.label_ == "PART_NUMBER":
                entities["part_numbers"].append(entity_data)
            elif ent.label_ == "AREA_ID":
                entities["areas"].append(entity_data)
            elif ent.label_ == "EQUIPMENT_TYPE":
                entities["equipment"].append(entity_data)
            elif ent.label_ == "IMAGE_REF":
                entities["image_refs"].append(entity_data)
            elif ent.label_ == "DOC_REF":
                entities["doc_refs"].append(entity_data)
            elif ent.label_ == "MAINTENANCE_ACTION":
                entities["actions"].append(entity_data)
            elif ent.label_ == "PROBLEM_TYPE":
                entities["problems"].append(entity_data)
            elif ent.label_ in ["CARDINAL", "ORDINAL"]:
                entities["numbers"].append(entity_data)
            else:
                entities["general"].append(entity_data)

        # Extract standalone numbers
        for token in doc:
            if token.like_num and len(token.text) >= 3:
                is_part_of_entity = any(
                    token.idx >= ent["start"] and token.idx < ent["end"]
                    for ent_list in entities.values()
                    for ent in ent_list
                )
                if not is_part_of_entity:
                    entities["numbers"].append({
                        "text": token.text,
                        "label": "NUMERIC_ID",
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                        "confidence": 0.7,
                        "numeric_value": int(token.text) if token.text.isdigit() else None
                    })

        return entities

    def _extract_semantic_info(self, doc):
        """Extract enhanced semantic information."""
        return {
            "lemmas": [token.lemma_ for token in doc if not token.is_stop and not token.is_punct],
            "pos_tags": [(token.text, token.pos_) for token in doc],
            "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
            "verbs": [token.lemma_ for token in doc if token.pos_ == "VERB"],
            "adjectives": [token.lemma_ for token in doc if token.pos_ == "ADJ"]
        }

    def _calculate_confidence(self, analysis):
        """Calculate enhanced confidence score with Decimal/Float safety."""
        from decimal import Decimal

        def safe_float(value):
            """Safely convert any numeric value to float."""
            if value is None:
                return 0.0
            if isinstance(value, Decimal):
                return float(value)
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0

        scores = []

        # Intent confidence (enhanced weight for database patterns)
        intent_confidence = safe_float(analysis["intent"]["confidence"])
        if analysis["intent"].get("method") == "database_patterns":
            intent_confidence *= 1.2  # Boost database pattern confidence
        scores.append(intent_confidence)

        # Entity confidence
        total_entities = sum(len(ent_list) for ent_list in analysis["entities"].values())
        if total_entities > 0:
            entity_confidence = min(total_entities * 0.2, 1.0)
            scores.append(entity_confidence)

        # Parameter confidence
        param_count = len([k for k in analysis["search_parameters"].keys() if k != "raw_input"])
        if param_count > 0:
            param_confidence = min(param_count * 0.3, 1.0)
            scores.append(param_confidence)

        return sum(scores) / len(scores) if scores else 0.0

    # Alternative: Add type safety to the entire analysis object
    def safe_analysis_preprocessing(self, analysis):
        """Preprocess analysis object to ensure all numeric values are floats."""
        from decimal import Decimal

        def convert_decimals_to_floats(obj):
            """Recursively convert Decimal values to floats in nested structures."""
            if isinstance(obj, dict):
                return {k: convert_decimals_to_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals_to_floats(item) for item in obj]
            elif isinstance(obj, Decimal):
                return float(obj)
            else:
                return obj

        return convert_decimals_to_floats(analysis)

    # Usage: Call this before _calculate_confidence
    def enhanced_calculate_confidence(self, analysis):
        """Enhanced version that preprocesses the analysis object."""
        # First, ensure all Decimals are converted to floats
        safe_analysis = self.safe_analysis_preprocessing(analysis)

        # Then proceed with normal calculation
        return self._calculate_confidence(safe_analysis)

    # Quick debugging version to identify the exact source
    def debug_calculate_confidence(self, analysis):
        """Debug version to identify where Decimals are coming from."""
        import logging
        from decimal import Decimal

        # Check for Decimals in the analysis object
        def find_decimals(obj, path=""):
            """Find all Decimal values in the analysis object."""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    find_decimals(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_decimals(item, f"{path}[{i}]")
            elif isinstance(obj, Decimal):
                logging.warning(f"Found Decimal at {path}: {obj}")

        # Debug the analysis object
        find_decimals(analysis, "analysis")

        scores = []

        # Intent confidence (with debugging)
        intent_confidence = analysis["intent"]["confidence"]
        if isinstance(intent_confidence, Decimal):
            logging.warning(f"Intent confidence is Decimal: {intent_confidence}")
            intent_confidence = float(intent_confidence)

        if analysis["intent"].get("method") == "database_patterns":
            intent_confidence *= 1.2
        scores.append(intent_confidence)

        # Rest of the method remains the same...
        total_entities = sum(len(ent_list) for ent_list in analysis["entities"].values())
        if total_entities > 0:
            entity_confidence = min(total_entities * 0.2, 1.0)
            scores.append(entity_confidence)

        param_count = len([k for k in analysis["search_parameters"].keys() if k != "raw_input"])
        if param_count > 0:
            param_confidence = min(param_count * 0.3, 1.0)
            scores.append(param_confidence)

        return sum(scores) / len(scores) if scores else 0.0

    @with_request_id
    def _add_to_analysis_cache(self, key: str, analysis: Dict[str, Any]):
        """Add analysis to enhanced cache with LRU behavior."""
        if len(self._analysis_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._analysis_cache))
            del self._analysis_cache[oldest_key]
        self._analysis_cache[key] = analysis

    @with_request_id
    def _get_suggestion_examples(self):
        """Get enhanced example phrases."""
        return [
            "find part A115957",
            "part number for bearing assembly",
            "show images of pump maintenance",
            "what's in room 2312",
            "find motor repair documentation"
        ]

    @with_request_id
    def get_nlp_statistics(self):
        """Get enhanced NLP processing statistics."""
        stats = {
            "spacy_available": self.has_spacy,
            "spacy_model": "en_core_web_sm" if self.nlp else None,
            "nlp_instance_active": self.nlp is not None,
            "custom_entities": list(self.entity_synonyms.keys()),
            "supported_intents": ["FIND_PART", "SHOW_IMAGES", "LOCATION_SEARCH", "MAINTENANCE_PROCEDURE"],
            "cache_size": len(self._analysis_cache),
            "cache_max_size": self._cache_max_size,
            "pattern_manager_available": self.pattern_manager is not None,
            "aggregate_search_available": self._aggregate_search is not None,
            "fallback_mode": not self.has_spacy,
            "synonym_count": sum(len(synonyms) for synonyms in self.entity_synonyms.values()),
            "database_patterns_loaded": len(self._database_patterns),
            "enhancement_level": "database_integrated"
        }

        if self.has_spacy and self.nlp:
            stats.update({
                "phrase_matchers": len(self.phrase_matcher) if hasattr(self, 'phrase_matcher') else 0,
                "intent_matchers": len(self.intent_matcher) if hasattr(self, 'intent_matcher') else 0,
                "pipeline_components": [comp for comp in self.nlp.pipe_names]
            })

        # Add database pattern statistics
        if self._database_patterns:
            pattern_stats = {
                "intents_with_patterns": list(self._database_patterns.keys()),
                "total_patterns": sum(len(intent_data['patterns']) for intent_data in self._database_patterns.values()),
                "high_success_patterns": sum(
                    1 for intent_data in self._database_patterns.values()
                    for pattern in intent_data['patterns']
                    if pattern['success_rate'] > 0.8
                ),
                "frequently_used_patterns": sum(
                    1 for intent_data in self._database_patterns.values()
                    for pattern in intent_data['patterns']
                    if pattern['usage_count'] > 10
                )
            }
            stats.update(pattern_stats)

        return stats

    def validate_nlp_setup(self):
        """Validate enhanced NLP setup with database integration."""
        diagnostics = {
            "timestamp": datetime.utcnow().isoformat(),
            "enhancement_level": "database_integrated",
            "spacy_installation": HAS_SPACY,
            "spacy_model_loaded": self.nlp is not None,
            "pattern_manager_available": self.pattern_manager is not None,
            "aggregate_search_available": self._aggregate_search is not None,
            "database_connectivity": self._session is not None,
            "models_available": MODELS_AVAILABLE,
            "utils_available": UTILS_AVAILABLE,
            "components_initialized": {},
            "warnings": [],
            "errors": [],
            "synonym_support": len(self.entity_synonyms) > 0,
            "database_patterns_loaded": len(self._database_patterns),
            "pattern_integration_status": "active" if self._database_patterns else "inactive"
        }

        # Test database pattern loading
        try:
            if self._session:
                test_patterns = self._load_database_patterns()
                diagnostics["database_pattern_test"] = "passed"
                diagnostics["patterns_available"] = len(test_patterns)
            else:
                diagnostics["database_pattern_test"] = "skipped"
                diagnostics["warnings"].append("No database session available")
        except Exception as e:
            diagnostics["database_pattern_test"] = "failed"
            diagnostics["errors"].append(f"Database pattern test failed: {str(e)}")

        # Test enhanced analysis
        try:
            test_analysis = self.analyze_user_input("find part A115957")
            diagnostics["enhanced_analysis_test"] = "passed"
            diagnostics["test_analysis_method"] = test_analysis.get("processing_method")
            diagnostics["test_extraction_method"] = test_analysis.get("search_parameters", {}).get("extraction_method")
        except Exception as e:
            diagnostics["enhanced_analysis_test"] = "failed"
            diagnostics["errors"].append(f"Enhanced analysis test failed: {str(e)}")

        # Test enhanced search
        try:
            test_search = self.execute_nlp_aggregated_search("find part 115982")
            diagnostics["enhanced_search_test"] = "passed"
            diagnostics["test_search_status"] = test_search.get("status")
            diagnostics["database_patterns_used"] = test_search.get("nlp_analysis", {}).get("database_patterns_used",
                                                                                            False)
        except Exception as e:
            diagnostics["enhanced_search_test"] = "failed"
            diagnostics["errors"].append(f"Enhanced search test failed: {str(e)}")

        return diagnostics

    def get_pattern_performance_report(self) -> Dict[str, Any]:
        """Get a comprehensive report on database pattern performance."""
        if not self._database_patterns:
            return {"status": "no_patterns", "message": "No database patterns loaded"}

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_intents": len(self._database_patterns),
            "total_patterns": 0,
            "intent_breakdown": {},
            "performance_summary": {
                "high_performers": [],  # >80% success rate
                "frequent_patterns": [],  # >50 uses
                "new_patterns": [],  # <5 uses
                "problematic_patterns": []  # <20% success rate and >10 uses
            }
        }

        for intent_name, intent_data in self._database_patterns.items():
            patterns = intent_data['patterns']
            report["total_patterns"] += len(patterns)

            intent_stats = {
                "pattern_count": len(patterns),
                "avg_success_rate": sum(p['success_rate'] for p in patterns) / len(patterns) if patterns else 0,
                "total_usage": sum(p['usage_count'] for p in patterns),
                "best_pattern": max(patterns, key=lambda p: p['success_rate']) if patterns else None
            }
            report["intent_breakdown"][intent_name] = intent_stats

            # Categorize patterns by performance
            for pattern in patterns:
                pattern_info = {
                    "intent": intent_name,
                    "pattern": pattern['pattern'],
                    "success_rate": pattern['success_rate'],
                    "usage_count": pattern['usage_count']
                }

                if pattern['success_rate'] > 0.8 and pattern['usage_count'] > 5:
                    report["performance_summary"]["high_performers"].append(pattern_info)
                elif pattern['usage_count'] > 50:
                    report["performance_summary"]["frequent_patterns"].append(pattern_info)
                elif pattern['usage_count'] < 5:
                    report["performance_summary"]["new_patterns"].append(pattern_info)
                elif pattern['success_rate'] < 0.2 and pattern['usage_count'] > 10:
                    report["performance_summary"]["problematic_patterns"].append(pattern_info)

        return report

    def optimize_patterns(self) -> Dict[str, Any]:
        """Suggest optimizations for database patterns based on performance."""
        performance_report = self.get_pattern_performance_report()

        if performance_report.get("status") == "no_patterns":
            return performance_report

        suggestions = {
            "timestamp": datetime.utcnow().isoformat(),
            "optimization_suggestions": [],
            "pattern_recommendations": {
                "disable_suggestions": [],
                "priority_boosts": [],
                "new_pattern_ideas": []
            }
        }

        # Analyze problematic patterns
        for pattern_info in performance_report["performance_summary"]["problematic_patterns"]:
            suggestions["pattern_recommendations"]["disable_suggestions"].append({
                "pattern": pattern_info["pattern"],
                "intent": pattern_info["intent"],
                "reason": f"Low success rate ({pattern_info['success_rate']:.1%}) with high usage ({pattern_info['usage_count']} uses)",
                "action": "Consider disabling or revising this pattern"
            })

        # Suggest priority boosts for high performers
        for pattern_info in performance_report["performance_summary"]["high_performers"]:
            if pattern_info["usage_count"] > 20:  # Only boost frequently used high performers
                suggestions["pattern_recommendations"]["priority_boosts"].append({
                    "pattern": pattern_info["pattern"],
                    "intent": pattern_info["intent"],
                    "reason": f"High success rate ({pattern_info['success_rate']:.1%}) with good usage ({pattern_info['usage_count']} uses)",
                    "action": "Consider increasing priority for this pattern"
                })

        return suggestions

    @with_request_id
    def refresh_patterns(self) -> Dict[str, Any]:
        """Manually refresh database patterns cache."""
        try:
            self._pattern_cache_timestamp = None  # Invalidate cache
            new_patterns = self._load_database_patterns()

            return {
                "status": "success",
                "message": "Database patterns refreshed successfully",
                "patterns_loaded": sum(len(intent_data['patterns']) for intent_data in new_patterns.values()),
                "intents_loaded": len(new_patterns),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error refreshing patterns: {e}")
            return {
                "status": "error",
                "message": f"Failed to refresh patterns: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    @with_request_id
    def reset_cache(self):
        """Reset enhanced analysis cache."""
        cache_size = len(self._analysis_cache)
        self._analysis_cache.clear()
        return {
            "status": "success",
            "message": f"Cleared {cache_size} cached analyses",
            "cache_size_before": cache_size,
            "cache_size_after": 0
        }

    @with_request_id
    def close_session(self):
        """Close database session."""
        if hasattr(self, '_session') and self._session:
            try:
                self._session.close()
                self._session = None
            except Exception as e:
                logger.error(f"Error closing enhanced NLP search session: {e}")

        if self._aggregate_search and hasattr(self._aggregate_search, 'close_session'):
            try:
                self._aggregate_search.close_session()
            except Exception as e:
                logger.error(f"Error closing aggregate search session: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()


# ===== ENHANCED SPACY AGGREGATE SEARCH WITH ML =====
class EnhancedSpaCyAggregateSearch(SpaCyEnhancedAggregateSearch):
    """
    Enhanced version with all new features including ML and session tracking.
    This includes everything from the original plus advanced ML capabilities.
    """

    def __init__(self, session=None, nlp_instance=None, user_context=None):
        super().__init__(session, nlp_instance, user_context)

        # Initialize new ML components
        self.session_manager = SearchSessionManager(session) if session else None
        self.pattern_generator = PatternTemplateGenerator()
        self.feedback_learner = FeedbackLearner(session) if session else None
        self.ml_classifier = IntentClassifierML()

        # Try to load ML model
        if session:
            self._try_load_ml_model()

    @with_request_id
    def _try_load_ml_model(self):
        """Try to load trained ML model"""
        try:
            active_model = self._session.query(MLModel).filter_by(
                model_type='intent_classifier',
                is_active=True
            ).first()

            if active_model and active_model.model_path:
                self.ml_classifier.load_model(active_model.model_path)
                logger.info("Loaded ML intent classifier")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")

    def execute_nlp_aggregated_search_enhanced(self, user_input: str,
                                               session_token: str = None) -> Dict[str, Any]:
        """Enhanced search with session tracking and ML"""

        # Start session if needed
        request_id = get_request_id()
        debug_id("Starting operation with key params", request_id)
        if not session_token and self.session_manager:
            session_token = self.session_manager.start_session(
                user_id=self.user_context.get('user_id', 'anonymous'),
                context=self.user_context
            )

        # Get ML prediction first
        ml_intent, ml_confidence = self.ml_classifier.predict_intent(user_input)

        # Execute original search
        result = self.execute_nlp_aggregated_search(user_input)

        # Combine ML and rule-based results
        if ml_confidence > 0.8 and result.get('nlp_analysis', {}).get('overall_confidence', 0) < 0.7:
            # Trust ML more than rules for this query
            result['nlp_analysis']['ml_intent'] = ml_intent
            result['nlp_analysis']['ml_confidence'] = ml_confidence
            result['nlp_analysis']['used_ml_prediction'] = True

        # Log the query
        if self.session_manager and session_token:
            self.session_manager.log_query(session_token, user_input, result)

        # Add session info to result
        result['session_token'] = session_token

        return result

    @with_request_id
    def record_user_feedback(self, query_id: int, feedback_data: Dict):
        """Record user feedback for learning"""
        if self.feedback_learner:
            self.feedback_learner.record_feedback(
                query_id=query_id,
                feedback_type=feedback_data.get('type', 'general'),
                rating=feedback_data.get('rating', 3)
            )

            # Record click data if available
            if 'clicked_results' in feedback_data:
                self.feedback_learner.learn_from_clicks(
                    query_id, feedback_data['clicked_results']
                )

    @with_request_id
    def auto_generate_patterns(self) -> List[str]:
        """Auto-generate new patterns from popular queries"""
        if not self.feedback_learner:
            return []

        popular_patterns = self.feedback_learner.get_popular_patterns()
        new_patterns = []

        for pattern_data in popular_patterns:
            # Use simple regex to extract pattern structure
            query = pattern_data['query']

            # Convert specific values to placeholders
            generalized = query

            # Replace part numbers with placeholder
            generalized = re.sub(r'\b[A-Z0-9]{4,}\b', '{part_number}', generalized)

            # Replace area identifiers
            generalized = re.sub(r'\barea\s+[A-Z0-9]+\b', 'area {area}', generalized, flags=re.IGNORECASE)

            # Replace equipment types (would need more sophisticated logic)
            equipment_types = ['pump', 'motor', 'valve', 'bearing', 'filter']
            for eq_type in equipment_types:
                generalized = re.sub(rf'\b{eq_type}\b', '{equipment}', generalized, flags=re.IGNORECASE)

            if generalized != query and generalized not in new_patterns:
                new_patterns.append(generalized)

        return new_patterns

    def retrain_ml_model(self) -> bool:
        """Retrain ML model with new data"""
        if not self.ml_classifier or not self._session:
            return False

        success = self.ml_classifier.train_from_database(self._session)
        if success:
            # Save model info to database
            model_record = MLModel(
                name=f"intent_classifier_v{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                model_type='intent_classifier',
                version='1.0',
                deployed_at=datetime.utcnow(),
                is_active=True
            )

            # Deactivate old models
            self._session.query(MLModel).filter_by(
                model_type='intent_classifier',
                is_active=True
            ).update({'is_active': False})

            self._session.add(model_record)
            self._session.commit()

            logger.info("Successfully retrained and deployed ML model")

        return success


# ===== FACTORY FUNCTIONS AND UTILITIES =====
def create_enhanced_search_system(session, user_context=None, nlp_instance=None):
    """
    Factory function to create enhanced search system with database integration.

    Args:
        session: Database session
        user_context: Optional user context dict
        nlp_instance: Optional pre-loaded spaCy instance

    Returns:
        SpaCyEnhancedAggregateSearch instance with database integration
    """
    enhanced_search = SpaCyEnhancedAggregateSearch(
        session=session,
        nlp_instance=nlp_instance,
        user_context=user_context or {}
    )

    logger.info(" Enhanced search system created with database integration")
    return enhanced_search


def create_ml_enhanced_search_system(session, user_context=None):
    """Factory function to create ML-enhanced search system"""
    enhanced_search = EnhancedSpaCyAggregateSearch(
        session=session,
        user_context=user_context or {}
    )

    # Initial ML training if no model exists
    if enhanced_search._session:
        try:
            existing_model = enhanced_search._session.query(MLModel).filter_by(
                model_type='intent_classifier', is_active=True
            ).first()

            if not existing_model:
                logger.info("No ML model found, training initial model...")
                enhanced_search.retrain_ml_model()
        except Exception as e:
            logger.warning(f"Could not check for existing ML models: {e}")

    return enhanced_search


# For backward compatibility, keep the original class name available
# Users can import either name
SpaCyEnhancedAggregateSearchLegacy = SpaCyEnhancedAggregateSearch

# todo remove
'''
#=========TO BE REMOVED==================
            
class DatabasePatternIntegrationMixin:
    """
    NEW: Mixin to integrate database patterns into SpaCy NLP search.
    Provides methods to load, cache, and use your database patterns.
    """

    def __init__(self):
        self._database_patterns = {}
        self._pattern_cache_timestamp = None
        self._pattern_cache_ttl = 300  # 5 minutes
        self._pattern_statistics = {}

    def _load_database_patterns(self) -> Dict[str, List[Dict]]:
        """FIXED: Load your regex patterns from the database with proper error handling."""
        from decimal import Decimal

        current_time = datetime.utcnow()

        # Check cache validity
        if (self._pattern_cache_timestamp and
                (current_time - self._pattern_cache_timestamp).seconds < self._pattern_cache_ttl and
                self._database_patterns):
            logger.debug("Using cached database patterns")
            return self._database_patterns

        # FIXED: Check if session is available
        if not self._session:
            logger.warning("No database session available for loading patterns")
            return {}

        def safe_float(value):
            """Safely convert Decimal or any numeric value to float."""
            if value is None:
                return 0.0
            if isinstance(value, Decimal):
                return float(value)
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0

        try:
            # FIXED: Clean transaction state first
            if self._session.in_transaction():
                self._session.rollback()

            # FIXED: Simpler query without ORDER BY issues
            query = text("""
                SELECT ip.intent_id, ip.pattern_text, ip.priority, ip.success_rate,
                       ip.usage_count, ip.pattern_type, si.name as intent_name,
                       si.description, si.search_method
                FROM intent_pattern ip
                JOIN search_intent si ON si.id = ip.intent_id
                WHERE ip.is_active = true AND si.is_active = true
                LIMIT 200
            """)

            results = self._session.execute(query).fetchall()

            # Organize by intent name
            patterns_by_intent = {}
            total_patterns = 0

            for row in results:
                intent_name = row[6]  # intent name
                pattern_text = row[1]  # pattern_text
                priority = safe_float(row[2]) or 1.0  # priority - FIXED
                success_rate = safe_float(row[3]) or 0.0  # success_rate - FIXED
                usage_count = row[4] or 0  # usage_count
                pattern_type = row[5] or 'regex'  # pattern_type
                description = row[7]  # description
                search_method = row[8]  # search_method

                if intent_name not in patterns_by_intent:
                    patterns_by_intent[intent_name] = {
                        'patterns': [],
                        'description': description,
                        'search_method': search_method
                    }

                try:
                    # Compile regex pattern
                    compiled_pattern = re.compile(pattern_text, re.IGNORECASE)

                    # FIXED: Now using safe float values
                    confidence_weight = priority * max(success_rate, 0.1)

                    patterns_by_intent[intent_name]['patterns'].append({
                        'pattern': pattern_text,
                        'priority': priority,
                        'success_rate': success_rate,
                        'usage_count': usage_count,
                        'pattern_type': pattern_type,
                        'compiled': compiled_pattern,
                        'confidence_weight': confidence_weight  # FIXED
                    })
                    total_patterns += 1

                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern_text}': {e}")
                    continue

            # Cache the results
            self._database_patterns = patterns_by_intent
            self._pattern_cache_timestamp = current_time

            logger.info(f" Loaded {total_patterns} database patterns for {len(patterns_by_intent)} intents")
            return patterns_by_intent

        except Exception as e:
            logger.error(f" Error loading database patterns: {e}")
            # Always rollback on error
            if self._session:
                try:
                    self._session.rollback()
                except:
                    pass
            return {}

    def _classify_intent_with_database(self, user_input: str) -> Dict[str, Any]:
        """ENHANCED: Use your database regex patterns for intent classification."""

        if not self._database_patterns:
            self._database_patterns = self._load_database_patterns()

        best_match = {"intent": "UNKNOWN", "confidence": 0.0, "method": "database_patterns"}
        all_matches = []

        for intent_name, intent_data in self._database_patterns.items():
            for pattern_info in intent_data['patterns']:
                try:
                    match = pattern_info['compiled'].search(user_input)
                    if match:
                        # Calculate confidence based on priority, success rate, and usage
                        base_confidence = pattern_info['confidence_weight']

                        # Boost confidence for frequently used successful patterns
                        usage_boost = min(pattern_info['usage_count'] / 100.0, 0.3)  # Max 0.3 boost
                        confidence = min(base_confidence + usage_boost, 1.0)

                        match_info = {
                            'intent': intent_name,
                            'confidence': confidence,
                            'matched_pattern': pattern_info['pattern'],
                            'extracted_groups': match.groups(),
                            'method': 'database_patterns',
                            'priority': pattern_info['priority'],
                            'success_rate': pattern_info['success_rate'],
                            'usage_count': pattern_info['usage_count'],
                            'search_method': intent_data.get('search_method')
                        }

                        all_matches.append(match_info)

                        if confidence > best_match['confidence']:
                            best_match = match_info

                except Exception as e:
                    logger.warning(f"Error matching pattern '{pattern_info['pattern']}': {e}")
                    continue

        # Add all matches for analysis
        if all_matches:
            best_match['all_database_matches'] = all_matches[:5]  # Top 5 matches

        return best_match

    def _extract_params_from_database_pattern(self, user_input: str, intent: str,
                                              matched_pattern: str, extracted_groups: tuple) -> Dict[str, Any]:
        """FIXED: Extract parameters using database pattern results with proper variable handling."""

        params = {}
        user_input_lower = user_input.lower()

        if intent == "FIND_PART":
            if extracted_groups:
                # Get the most relevant extraction (usually the last non-empty group)
                extracted = None
                for group in reversed(extracted_groups):
                    if group and group.strip():
                        extracted = group.strip()
                        break

                if extracted:
                    # Determine if this is a part number or description based on pattern
                    if any(phrase in matched_pattern.lower() for phrase in
                           ['part number for', 'number for', 'part.*for']):
                        # This is a description request like "part number for BEARING ASSEMBLY"
                        cleaned_description = re.sub(r'\s+', ' ', extracted.strip())
                        params.update({
                            "search_text": cleaned_description,
                            "entity_type": "part",
                            "fields": ['name', 'part_number', 'oem_mfg', 'model', 'notes'],
                            "extraction_method": "database_pattern_description"
                        })
                        logger.debug(f" Database pattern extracted description: '{cleaned_description}'")

                    else:
                        # This should be a direct part number like "find part A115957"
                        part_candidate = extracted.upper()  # FIXED: Proper variable definition

                        # FIXED: Correct validation with proper closing parenthesis
                        if (re.match(r'^[A-Za-z0-9\-\.]{3,}$', part_candidate) and
                                part_candidate not in ['FOR', 'THE', 'A', 'AN', 'OF', 'IN', 'ON', 'AT', 'TO', 'NUMBER',
                                                       'PART']):
                            params.update({
                                "part_number": part_candidate,
                                "entity_type": "part",
                                "extraction_method": "database_pattern_direct"
                            })
                            logger.debug(f" Database pattern extracted part number: '{part_candidate}'")
                        else:
                            # Fallback to description search
                            params.update({
                                "search_text": extracted,
                                "entity_type": "part",
                                "fields": ['name', 'part_number', 'oem_mfg', 'model', 'notes'],
                                "extraction_method": "database_pattern_fallback"
                            })

        elif intent == "SHOW_IMAGES":
            params["entity_type"] = "image"
            if extracted_groups:
                subject = extracted_groups[-1] if extracted_groups[-1] else extracted_groups[0]
                if subject:
                    params.update({
                        "search_text": subject.strip(),
                        "extraction_method": "database_pattern_image"
                    })

        elif intent == "LOCATION_SEARCH":
            params["entity_type"] = "position"
            if extracted_groups:
                if len(extracted_groups) >= 2:
                    location_type = extracted_groups[0].lower() if extracted_groups[0] else ""
                    location_id = extracted_groups[1].upper() if extracted_groups[1] else ""

                    if location_type in ['area', 'zone'] and location_id:
                        params["area"] = location_id
                    elif location_type == 'room' and location_id:
                        params["location"] = location_id
                elif len(extracted_groups) == 1 and extracted_groups[0]:
                    location_value = extracted_groups[0].strip().upper()
                    if re.match(r'^[A-Z0-9]{1,4}$', location_value):
                        params["area"] = location_value

                params["extraction_method"] = "database_pattern_location"

        return params

    def update_pattern_usage(self, intent_info: Dict[str, Any], search_result: Dict[str, Any]):
        """ENHANCED: Update pattern usage statistics in your database."""

        if (intent_info.get('method') == 'database_patterns' and
                'matched_pattern' in intent_info and
                self._session):

            try:
                success = search_result.get('status') == 'success'
                result_count = search_result.get('count', 0)
                matched_pattern = intent_info['matched_pattern']

                # Calculate success boost based on result count
                success_value = 1.0 if success and result_count > 0 else 0.0

                # Update usage statistics
                update_query = text("""
                    UPDATE intent_pattern 
                    SET usage_count = usage_count + 1,
                        success_rate = CASE 
                            WHEN usage_count = 0 THEN :success_value
                            ELSE (success_rate * usage_count + :success_value) / (usage_count + 1)
                        END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE pattern_text = :pattern AND is_active = true
                """)

                self._session.execute(update_query, {
                    'success_value': success_value,
                    'pattern': matched_pattern
                })
                self._session.commit()

                logger.debug(
                    f" Updated pattern usage: {matched_pattern} (success: {success}, results: {result_count})")

                # Invalidate cache to reload updated statistics
                self._pattern_cache_timestamp = None

            except Exception as e:
                logger.warning(f" Could not update pattern statistics: {e}")
                if self._session:
                    try:
                        self._session.rollback()
                    except:
                        pass

    def _load_entity_synonyms_fixed(self) -> Dict[str, Dict[str, str]]:
        """FIXED: Load entity synonyms with proper transaction handling."""
        if not self._session:
            logger.warning("No database session available for loading synonyms")
            return {}

        try:
            # CRITICAL FIX: Ensure clean transaction state
            if self._session.in_transaction():
                self._session.rollback()

            # FIXED: Simple query without ORDER BY issues
            query = text("""
                SELECT es.synonym_value, es.canonical_value, es.confidence_score, et.name
                FROM entity_synonym es
                JOIN entity_type et ON et.id = es.entity_type_id
                WHERE es.confidence_score > 0.5
                LIMIT 200
            """)

            results = self._session.execute(query).fetchall()

            synonyms = {}
            for row in results:
                synonym_value = row[0]
                canonical_value = row[1]
                confidence_score = row[2]
                entity_type_name = row[3]

                if entity_type_name not in synonyms:
                    synonyms[entity_type_name] = {}

                synonyms[entity_type_name][synonym_value.lower()] = {
                    "canonical": canonical_value,
                    "confidence": confidence_score
                }

            logger.debug(f" Loaded {len(results)} entity synonyms for {len(synonyms)} entity types")
            return synonyms

        except Exception as e:
            logger.error(f" Error loading entity synonyms: {e}")
            # Always rollback on error
            if self._session:
                try:
                    self._session.rollback()
                except:
                    pass
            return {}

class PatternTemplateGenerator:
    """Generate pattern variations from templates automatically"""

    def generate_variations(self, template: PatternTemplate) -> List[str]:
        """
        Convert template like "{action} {entity} {location}"
        into variations like "find pump in area A"
        """
        variations = []
        param_types = template.parameter_types or {}

        # Generate all combinations
        param_combinations = []
        for param, values in param_types.items():
            param_combinations.append([(param, value) for value in values])

        if param_combinations:
            for combo in itertools.product(*param_combinations):
                variation = template.template_text
                for param, value in combo:
                    variation = variation.replace(f"{{{param}}}", value)
                variations.append(variation)
        else:
            # If no parameter types defined, return original template
            variations.append(template.template_text)

        return variations
  
# todo rmove

class SearchIntentHierarchy(Base):
    __tablename__ = 'search_intent_hierarchy'
    id = Column(Integer, primary_key=True)
    parent_intent_id = Column(Integer, ForeignKey('search_intent.id'))
    child_intent_id = Column(Integer, ForeignKey('search_intent.id'))
    inheritance_type = Column(String(50))
    priority_modifier = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class IntentContext(Base):
    __tablename__ = 'intent_context'
    id = Column(Integer, primary_key=True)
    intent_id = Column(Integer, ForeignKey('search_intent.id'))
    context_type = Column(String(50))
    context_value = Column(String(200))
    boost_factor = Column(Float, default=1.0)
    is_required = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PatternTemplate(Base):
    __tablename__ = 'pattern_template'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    template_text = Column(Text, nullable=False)
    parameter_types = Column(JSON)
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class PatternVariation(Base):
    __tablename__ = 'pattern_variation'
    id = Column(Integer, primary_key=True)
    template_id = Column(Integer, ForeignKey('pattern_template.id'))
    intent_id = Column(Integer, ForeignKey('search_intent.id'))
    variation_text = Column(Text, nullable=False)
    confidence_weight = Column(Float, default=1.0)
    language_code = Column(String(5), default='en')
    created_at = Column(DateTime, default=datetime.utcnow)


class EntityType(Base):
    __tablename__ = 'entity_type'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    validation_regex = Column(Text)
    normalization_rules = Column(JSON)
    is_core_entity = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class EntitySynonym(Base):
    __tablename__ = 'entity_synonym'
    id = Column(Integer, primary_key=True)
    entity_type_id = Column(Integer, ForeignKey('entity_type.id'))
    canonical_value = Column(String(200), nullable=False)
    synonym_value = Column(String(200), nullable=False)
    confidence_score = Column(Float, default=1.0)
    source = Column(String(50))
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint('entity_type_id', 'synonym_value'),)      '''