"""
entities.py
-----------
ORM models for intent/entity hierarchy, patterns, and synonyms.
Extracted directly from the original nlp_search.py.
"""

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class SearchIntentHierarchy(Base):
    __tablename__ = "search_intent_hierarchy"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    parent_id = Column(Integer, ForeignKey("search_intent_hierarchy.id"), nullable=True)

    parent = relationship("SearchIntentHierarchy", remote_side=[id], backref="children")

    def __repr__(self):
        return f"<SearchIntentHierarchy id={self.id} name={self.name}>"


class IntentContext(Base):
    __tablename__ = "intent_context"

    id = Column(Integer, primary_key=True)
    intent_id = Column(Integer, ForeignKey("search_intent_hierarchy.id"))
    context = Column(String, nullable=False)

    intent = relationship("SearchIntentHierarchy", backref="contexts")

    def __repr__(self):
        return f"<IntentContext id={self.id} context={self.context}>"


class PatternTemplate(Base):
    __tablename__ = "pattern_template"

    id = Column(Integer, primary_key=True)
    template = Column(String, nullable=False)

    def __repr__(self):
        return f"<PatternTemplate id={self.id} template={self.template}>"


class PatternVariation(Base):
    __tablename__ = "pattern_variation"

    id = Column(Integer, primary_key=True)
    template_id = Column(Integer, ForeignKey("pattern_template.id"))
    variation = Column(String, nullable=False)

    template = relationship("PatternTemplate", backref="variations")

    def __repr__(self):
        return f"<PatternVariation id={self.id} variation={self.variation}>"


class EntityType(Base):
    __tablename__ = "entity_type"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    def __repr__(self):
        return f"<EntityType id={self.id} name={self.name}>"


class EntitySynonym(Base):
    __tablename__ = "entity_synonym"

    id = Column(Integer, primary_key=True)
    entity_type_id = Column(Integer, ForeignKey("entity_type.id"))
    synonym = Column(String, nullable=False)

    entity_type = relationship("EntityType", backref="synonyms")

    def __repr__(self):
        return f"<EntitySynonym id={self.id} synonym={self.synonym}>"


__all__ = [
    "SearchIntentHierarchy",
    "IntentContext",
    "PatternTemplate",
    "PatternVariation",
    "EntityType",
    "EntitySynonym",
]
