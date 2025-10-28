# tests/nlp/test_spacy_pipeline.py

import pytest
from modules.emtac_ai.search.nlp.spacy_search import SpaCyEnhancedAggregateSearch


def test_spacy_pipeline_analyze_user_input():
    """
    Sanity check that the SpaCyEnhancedAggregateSearch pipeline runs
    and returns a dict with the expected keys.
    """
    nlp = SpaCyEnhancedAggregateSearch()

    query = "find valve in area 2"
    result = nlp.analyze_user_input(query)

    # Core structure
    assert isinstance(result, dict)
    assert "query" in result
    assert "intent" in result
    assert "entities" in result
    assert "confidence_score" in result
    assert "processing_method" in result

    # Specific expectations
    assert result["query"] == query
    assert isinstance(result["intent"], str)
    assert isinstance(result["entities"], list)
    assert isinstance(result["confidence_score"], float)
    assert result["processing_method"] in ("spacy", "fallback")

    # Entities should capture at least "2" as a number or CARDINAL
    texts = [ent["text"] for ent in result["entities"]]
    assert any("2" in t for t in texts)
