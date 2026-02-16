import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from importlib import import_module, reload

PROJECT_ROOT = os.path.abspath("E:/emtac/projects/llm/MDEV_EMTAC")
sys.path.insert(0, PROJECT_ROOT)


# -------------------------------------------------------
# Fake Model for Tools
# -------------------------------------------------------
class FakeTool:
    def __init__(self, id, name="Wrench", ttype="hand", size="M10", material="steel", desc=""):
        self.id = id
        self.name = name
        self.type = ttype
        self.size = size
        self.material = material
        self.description = desc


# -------------------------------------------------------
# Fixture: Patch DBServices and reload module
# -------------------------------------------------------
@pytest.fixture
def mock_db_services():
    """Patch DBServices and reload tools_router so it uses the mock DB instance."""
    with patch("modules.services.DBServices") as mock_db_class:

        mock_db_instance = MagicMock()

        # Sub-services used by router
        mock_db_instance.tools = MagicMock()

        mock_db_class.return_value = mock_db_instance

        # Reload router module to bind DB = mock_db_instance
        mod = import_module("modules.emtac_ai.intent_ner.routers.tools_router")
        reload(mod)

        yield mock_db_instance


# -------------------------------------------------------
# TESTS
# -------------------------------------------------------

def test_by_tool_id(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.tools_router import tools_router

    fake = FakeTool(10)
    mock_db_services.tools.get.return_value = fake

    # Related lookups
    mock_db_services.tools.get_positions_for_tool.return_value = ["P1"]
    mock_db_services.tools.get_tasks_for_tool.return_value = ["T1"]
    mock_db_services.tools.get_problems_for_tool.return_value = ["PR1"]
    mock_db_services.tools.get_solutions_for_tool.return_value = ["S1"]

    r = tools_router(
        text="",
        intent="Tools",
        confidence=0.95,
        entities={"tool_id": 10},
    )

    assert r["matched_on"] == "by_tool_id"
    assert r["results"][0]["id"] == 10
    assert r["related"]["positions"] == ["P1"]
    assert r["related"]["tasks"] == ["T1"]


def test_name_search(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.tools_router import tools_router

    fake = FakeTool(20, name="Hammer")
    mock_db_services.tools.find.return_value = [fake]

    r = tools_router(
        text="",
        intent="Tools",
        confidence=0.9,
        entities={"name": "Hammer"},
    )

    assert r["matched_on"] == "by_name"
    assert r["results"][0]["name"] == "Hammer"


def test_metadata_search(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.tools_router import tools_router

    fake = FakeTool(30, ttype="air", size="1/2in", material="aluminum")
    mock_db_services.tools.find.return_value = [fake]

    r = tools_router(
        text="",
        intent="Tools",
        confidence=0.9,
        entities={"type": "air", "size": "1/2in", "material": "aluminum"},
    )

    assert r["matched_on"] == "by_metadata"
    assert r["results"][0]["type"] == "air"


def test_position_search(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.tools_router import tools_router

    mock_db_services.tools.find_by_position.return_value = [FakeTool(40)]

    r = tools_router(
        text="",
        intent="Tools",
        confidence=0.9,
        entities={"position": 200},
    )

    assert r["matched_on"] == "by_position"
    assert r["results"][0]["id"] == 40


def test_fallback_search(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.tools_router import tools_router

    mock_db_services.tools.search_name.return_value = [FakeTool(55)]

    r = tools_router(
        text="impact driver",
        intent="Tools",
        confidence=0.9,
        entities={},
    )

    assert r["matched_on"] == "fallback"
    assert r["results"][0]["id"] == 55


def test_no_results(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.tools_router import tools_router

    mock_db_services.tools.search_name.return_value = []

    r = tools_router(
        text="unicorn laser hammer",
        intent="Tools",
        confidence=0.9,
        entities={},
    )

    assert r["matched_on"] == "no_results"
    assert r["results"] == []
