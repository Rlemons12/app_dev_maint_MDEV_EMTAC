"""
models.py
---------
SQLAlchemy ORM classes for NLP search.
Split out from the old nlp_search.py monolith.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float,
    DateTime, ForeignKey, JSON, UniqueConstraint
)
from sqlalchemy.orm import relationship
from modules.configuration.base import Base


# ------------------------------
# Sessions & Queries
# ------------------------------
class SearchSession(Base):
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

    queries = relationship(
        "SearchQuery",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SearchQuery(Base):
    __tablename__ = 'search_query'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('search_session.id'))
    parent_query_id = Column(Integer, ForeignKey('search_query.id'))
    query_text = Column(Text, nullable=False)
    normalized_query = Column(Text)
    detected_intent_id = Column(Integer, ForeignKey('search_intent.id'))
    intent_confidence = Column(Float)
    extracted_entities = Column(JSON)
    search_method = Column(String(100))
    execution_time_ms = Column(Integer)
    result_count = Column(Integer)
    was_successful = Column(Boolean, default=False)
    user_satisfaction_score = Column(Integer)
    was_refined = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("SearchSession", back_populates="queries")
    child_queries = relationship("SearchQuery", backref="parent_query", remote_side=[id])


class SearchResultClick(Base):
    __tablename__ = 'search_result_click'

    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey('search_query.id'))
    result_id = Column(String(200))
    clicked_at = Column(DateTime, default=datetime.utcnow)


# ------------------------------
# ML Models & Feedback
# ------------------------------
class MLModel(Base):
    __tablename__ = 'ml_model'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    model_type = Column(String(50))
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
    feedback_type = Column(String(50))
    feedback_value = Column(String(500))
    rating = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------------
# Intent Hierarchy & Context
# ------------------------------
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


# ------------------------------
# Patterns & Entities
# ------------------------------
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

    __table_args__ = (UniqueConstraint('entity_type_id', 'synonym_value'),)

class SearchAnalytics(Base):
    """
    Analytics and performance tracking for search operations - FIXED to match actual database schema
    """
    __tablename__ = 'search_analytics'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100))  # FIXED: added this column
    session_id = Column(String(100))
    query_text = Column(Text)  # FIXED: was 'user_input'
    detected_intent = Column(String(100))  # FIXED: was ForeignKey
    intent_confidence = Column(Float)  # FIXED: was 'confidence_score'
    search_method = Column(String(100))
    execution_time_ms = Column(Integer)
    result_count = Column(Integer)
    success = Column(Boolean)  # FIXED: added this column
    error_message = Column(Text)  # FIXED: added this column
    user_agent = Column(Text)  # FIXED: added this column
    ip_address = Column(String(45))  # FIXED: added this column (inet type maps to string)
    created_at = Column(DateTime, default=datetime.utcnow)