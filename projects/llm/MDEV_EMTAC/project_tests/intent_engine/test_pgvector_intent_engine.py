"""
test_pgvector_intent_engine.py

Full unit test for PgVectorIntentEngine.

This test:
    - Boots project root using demos/bootstrap.py
    - Mocks DB session and pgvector SQL execution
    - Provides a fake embedder that returns deterministic vectors
    - Tests:
        ✔ normal ranking behavior
        ✔ missing DB rows → general_chat
        ✔ embedding failure → general_chat
        ✔ SQL execution error → general_chat
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

# ============================================================================
# 1. BOOTSTRAP PROJECT ROOT
# ============================================================================
def _bootstrap():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    bootstrap_path = os.path.abspath(
        os.path.join(test_dir, "../demos/bootstrap.py")
    )

    if not os.path.exists(bootstrap_path):
        raise RuntimeError(f"bootstrap.py missing at {bootstrap_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("bootstrap", bootstrap_path)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)

    project_root = bootstrap.bootstrap_paths()
    print(f"[BOOTSTRAP] Project root added: {project_root}")

    assert project_root in sys.path


_bootstrap()

# ============================================================================
# 2. IMPORT THE ENGINE
# ============================================================================
from modules.emtac_ai.intent_ner.engine.pgvector_intent_engine import (
    PgVectorIntentEngine
)


# ============================================================================
# 3. FAKE EMBEDDER
# ============================================================================
class FakeEmbedder:
    """Deterministic embedder for testing."""
    def __init__(self, vector=None, fail=False):
        self.vector = vector or [0.1, 0.2, 0.3]
        self.fail = fail

    def encode(self, text):
        if self.fail:
            raise RuntimeError("Embedding failure simulated")
        return self.vector


# ============================================================================
# 4. MOCKED SESSION / QUERY RESULTS
# ============================================================================
class FakeRow:
    def __init__(self, intent, confidence):
        self.intent = intent
        self.confidence = confidence


class FakeSession:
    def __init__(self, result=None, fail=False):
        self.result = result
        self.fail = fail
        self.closed = False

    def execute(self, sql, params):
        if self.fail:
            raise RuntimeError("Simulated SQL execution error")
        # Return object with .fetchone()
        return MagicMock(fetchone=MagicMock(return_value=self.result))

    def close(self):
        self.closed = True


def fake_session_factory(result=None, fail=False):
    """Produces fake session objects just like DatabaseConfig.get_main_session."""
    def _factory():
        return FakeSession(result=result, fail=fail)
    return _factory


# ============================================================================
# 5. TESTS
# ============================================================================

def test_pgvector_intent_normal_ranking():
    """Engine should return the row with highest confidence."""
    row = FakeRow(intent="Troubleshooting", confidence=0.88)

    engine = PgVectorIntentEngine(
        session_factory=fake_session_factory(result=row),
        embedder=FakeEmbedder()
    )

    out = engine.detect_intent("motor won't start")

    assert out["intent"] == "Troubleshooting"
    assert abs(out["confidence"] - 0.88) < 1e-6


def test_pgvector_intent_no_rows():
    """If DB returns no row → return general_chat."""
    engine = PgVectorIntentEngine(
        session_factory=fake_session_factory(result=None),
        embedder=FakeEmbedder()
    )

    out = engine.detect_intent("unknown text")
    assert out["intent"] == "general_chat"
    assert out["confidence"] == 0.0


def test_pgvector_embedder_failure():
    """If embedder fails → return general_chat."""
    engine = PgVectorIntentEngine(
        session_factory=fake_session_factory(result=None),
        embedder=FakeEmbedder(fail=True)
    )

    out = engine.detect_intent("anything")
    assert out["intent"] == "general_chat"
    assert out["confidence"] == 0.0


def test_pgvector_sql_failure():
    """If SQL execution fails → return general_chat."""
    engine = PgVectorIntentEngine(
        session_factory=fake_session_factory(fail=True),
        embedder=FakeEmbedder()
    )

    out = engine.detect_intent("test query")
    assert out["intent"] == "general_chat"
    assert out["confidence"] == 0.0


# ============================================================================
# 6. ALLOW DIRECT RUN
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-q"])
