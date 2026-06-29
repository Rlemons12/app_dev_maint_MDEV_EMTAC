import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from importlib import import_module
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath("E:/emtac/projects/llm/MDEV_EMTAC")
sys.path.insert(0, PROJECT_ROOT)


# -------------------------------------------------------
# Fake Model
# -------------------------------------------------------
class FakeProblem:
    def __init__(self, id, name="Overheating", desc="Problem desc"):
        self.id = id
        self.name = name
        self.description = desc


# -------------------------------------------------------
# Fixture: Patch DBServices and reload troubleshooting_router
# -------------------------------------------------------
@pytest.fixture
def mock_db_services():
    with patch("modules.services.DBServices") as mock_db_class:

        mock_instance = MagicMock()
        mock_instance.troubleshooting = MagicMock()

        mock_db_class.return_value = mock_instance

        importlib = __import__("importlib")
        importlib.invalidate_caches()

        mod = import_module("modules.emtac_ai.intent_ner.routers.troubleshooting_router")
        importlib.reload(mod)

        return SimpleNamespace(
            db=mock_instance,
            router=mod.troubleshooting_router,
        )


# -------------------------------------------------------
# TESTS
# -------------------------------------------------------
def test_by_problem_id(mock_db_services):
    mock = mock_db_services

    fake = FakeProblem(10)
    mock.db.troubleshooting.get_problem.return_value = fake
    mock.db.troubleshooting.find_related.return_value = {"tree": "T1"}

    r = mock.router(
        text="motor problem",
        intent="Troubleshooting",
        confidence=0.9,
        entities={"problem_id": 10},
    )

    assert r["matched_on"] == "problem_id"
    assert r["results"][0]["id"] == 10
    assert r["related"] == {"tree": "T1"}


def test_by_problem_name_single_match(mock_db_services):
    mock = mock_db_services

    fake = FakeProblem(20, name="Thermal Fault")
    mock.db.troubleshooting.search_problems.return_value = [fake]
    mock.db.troubleshooting.find_related.return_value = {"steps": ["A", "B", "C"]}

    r = mock.router(
        text="thermal fault",
        intent="Troubleshooting",
        confidence=0.9,
        entities={"problem_name": "Thermal Fault"},
    )

    assert r["matched_on"] == "problem_name"
    assert r["results"][0]["name"] == "Thermal Fault"
    assert "steps" in r["related"]


def test_by_problem_name_multiple_matches(mock_db_services):
    mock = mock_db_services

    p1 = FakeProblem(30, name="Leak")
    p2 = FakeProblem(31, name="Leakage Detector")

    mock.db.troubleshooting.search_problems.return_value = [p1, p2]

    mock.db.troubleshooting.find_related.side_effect = AssertionError(
        "find_related should NOT be called for multiple matches"
    )

    r = mock.router(
        text="leak",
        intent="Troubleshooting",
        confidence=0.9,
        entities={"problem_name": "leak"},
    )

    assert r["matched_on"] == "problem_name"
    assert len(r["results"]) == 2
    assert r["related"] is None


def test_fallback_resolved(mock_db_services):
    mock = mock_db_services

    fake = FakeProblem(40, "Pressure Loss")

    mock.db.troubleshooting.resolve_query.return_value = {
        "status": "resolved",
        "problem": fake,
        "tree": {"root": "cause1"},
    }

    r = mock.router(
        text="pressure loss issue",
        intent="Troubleshooting",
        confidence=0.9,
        entities={},
    )

    assert r["matched_on"] == "resolved"
    assert r["results"][0]["id"] == 40
    assert r["related"] == {"root": "cause1"}


def test_fallback_multiple_matches(mock_db_services):
    mock = mock_db_services

    p1 = FakeProblem(50, "Noise")
    p2 = FakeProblem(51, "High Noise")

    mock.db.troubleshooting.resolve_query.return_value = {
        "status": "multiple_matches",
        "choices": [p1, p2],
    }

    r = mock.router(
        text="noise problem",
        intent="Troubleshooting",
        confidence=0.9,
        entities={},
    )

    assert r["matched_on"] == "multiple_results"
    assert len(r["results"]) == 2
    assert r["related"] is None


def test_fallback_no_results(mock_db_services):
    mock = mock_db_services

    mock.db.troubleshooting.resolve_query.return_value = {
        "status": "no_match"
    }

    r = mock.router(
        text="qwerty nonexistent",
        intent="Troubleshooting",
        confidence=0.9,
        entities={},
    )

    assert r["matched_on"] == "no_results"
    assert r["results"] == []
    assert r["related"] is None
