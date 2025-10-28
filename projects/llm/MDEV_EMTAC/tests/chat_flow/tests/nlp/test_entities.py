# tests/nlp/test_entities.py

import pytest
from modules.emtac_ai.search.nlp.entities import (
    SearchIntentHierarchy,
    IntentContext,
    PatternTemplate,
    PatternVariation,
    EntityType,
    EntitySynonym,
)


@pytest.mark.parametrize(
    "cls, expected_tablename",
    [
        (SearchIntentHierarchy, "search_intent_hierarchy"),
        (IntentContext, "intent_context"),
        (PatternTemplate, "pattern_template"),
        (PatternVariation, "pattern_variation"),
        (EntityType, "entity_type"),
        (EntitySynonym, "entity_synonym"),
    ],
)
def test_entities_have_correct_tablenames(cls, expected_tablename):
    """Ensure each entity has the correct __tablename__ defined."""
    assert cls.__tablename__ == expected_tablename


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (SearchIntentHierarchy, {"name": "mechanical"}),
        (IntentContext, {"intent": SearchIntentHierarchy(name="search.part"), "context": "equipment"}),
        (PatternTemplate, {"template": "find {part} in {location}"}),
        (PatternVariation, {"variation": "lookup valve at station 5"}),
        (EntityType, {"name": "part"}),
        (EntitySynonym, {"entity_type": EntityType(name="part"), "synonym": "valve"}),
    ],
)
def test_entities_repr_runs(cls, kwargs):
    """Ensure each entity can be instantiated and __repr__ returns a string."""
    obj = cls(**kwargs)
    repr_str = repr(obj)
    assert isinstance(repr_str, str)
    assert cls.__name__ in repr_str
