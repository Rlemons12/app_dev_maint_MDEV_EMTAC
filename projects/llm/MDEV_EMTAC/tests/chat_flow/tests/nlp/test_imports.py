import importlib
import pytest


def test_nlp_imports_cleanly():
    """
    Verify that modules.emtac_ai.search.nlp imports without errors
    and exposes all expected symbols in __all__.
    """
    module_name = "modules.emtac_ai.search.nlp"
    nlp = importlib.import_module(module_name)

    assert hasattr(nlp, "__all__"), f"{module_name} must define __all__"
    missing = []
    for symbol in nlp.__all__:
        if not hasattr(nlp, symbol):
            missing.append(symbol)
    assert not missing, f"Missing exports in {module_name}: {missing}"


@pytest.mark.parametrize("symbol", [
    "SearchQueryTracker", "SearchSessionManager",
    "SearchSession", "SearchQuery", "SearchResultClick",
    "MLModel", "UserFeedback",
    "SearchIntentHierarchy", "IntentContext",
    "PatternTemplate", "PatternVariation",
    "EntityType", "EntitySynonym",
    "IntentClassifierML", "FeedbackLearner",
    "record_feedback", "get_feedback_for_query", "average_rating_for_query",
    "SpaCyEnhancedAggregateSearch",
    "create_search_session", "create_search_query",
])
def test_symbol_imports(symbol):
    """
    Check that each expected symbol can be imported directly from the package.
    """
    module_name = "modules.emtac_ai.search.nlp"
    nlp = importlib.import_module(module_name)
    assert hasattr(nlp, symbol), f"{symbol} not found in {module_name}"
