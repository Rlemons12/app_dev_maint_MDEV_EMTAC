"""
nlp package
-----------
Exports NLP models, trackers, and pipeline components.
"""

from .models import (
    SearchSession,
    SearchQuery,
    SearchResultClick, SearchAnalytics,)
from .tracker import SearchQueryTracker, SearchSessionManager
from .ml_models import IntentClassifierML
from .spacy_search import SpaCyEnhancedAggregateSearch
from .feedback import FeedbackLearner, record_feedback, get_feedback_for_query, average_rating_for_query
from .factories import create_search_session, create_search_query
from .entities import (
    SearchIntentHierarchy,
    IntentContext,
    PatternTemplate,
    PatternVariation,
    EntityType,
    EntitySynonym,
)

__all__ = [
    # ORM models
    "SearchSession",
    "SearchQuery",
    "SearchResultClick",
    "SearchAnalytics",
    "SearchIntentHierarchy",
    "IntentContext",
    "PatternTemplate",
    "PatternVariation",
    "EntityType",
    "EntitySynonym",

    # Trackers
    "SearchQueryTracker",
    "SearchSessionManager",

    # ML
    "IntentClassifierML",

    # SpaCy pipeline
    "SpaCyEnhancedAggregateSearch",

    # Feedback
    "FeedbackLearner",
    "record_feedback",
    "get_feedback_for_query",
    "average_rating_for_query",

    # Factories
    "create_search_session",
    "create_search_query",
]
