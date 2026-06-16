import sys
import os

PROJECT_ROOT = os.path.abspath("E:/emtac/projects/llm/MDEV_EMTAC")
sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import MagicMock, patch
from importlib import import_module, reload


class FakeDrawing:
    """Mock drawing object matching the actual Drawing model"""

    def __init__(
        self,
        id=1,
        number="A-1234",
        name="Test Drawing",
        equipment="Bag Maker",
        rev="01",
        spare=None,
        dtype="Other",
        file_path="A-1234.pdf"
    ):
        self.id = id
        self.drw_number = number
        self.drw_name = name
        self.drw_equipment_name = equipment
        self.drw_revision = rev
        self.drw_spare_part_number = spare
        self.drw_type = dtype
        self.file_path = file_path


@pytest.fixture
def mock_db_services():
    """Patch DBServices and reload router so BaseRouter uses the mock."""
    with patch("modules.services.DBServices") as mock_db_class:

        mock_db_instance = MagicMock()
        mock_db_instance.drawings = MagicMock()
        mock_db_instance.drawings.find = MagicMock()
        mock_db_instance.drawings.find_related = MagicMock()

        mock_db_class.return_value = mock_db_instance

        dr_module = import_module("modules.emtac_ai.intent_ner.routers.drawings_router")
        reload(dr_module)

        yield mock_db_instance


# ------------------------------------------------------------
# TESTS
# ------------------------------------------------------------

def test_drawings_router_matches_by_drawing_number(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.drawings_router import drawings_router

    fake = FakeDrawing(id=10, number="A-5088-23-111")
    mock_db_services.drawings.find.return_value = [fake]
    mock_db_services.drawings.find_related.return_value = None

    result = drawings_router(
        text="Show drawing A-5088-23-111",
        intent="Drawings",
        confidence=0.92,
        entities={"drawing_number": "A-5088-23-111"}
    )

    assert result["matched_on"] == "search_by_drawing_number"
    assert len(result["results"]) == 1
    assert result["results"][0]["number"] == "A-5088-23-111"
    assert result["results"][0]["id"] == 10

    mock_db_services.drawings.find.assert_called_once_with(drw_number="A-5088-23-111")
    mock_db_services.drawings.find_related.assert_called_once_with(10)


def test_drawings_router_matches_by_name(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.drawings_router import drawings_router

    fake = FakeDrawing(id=20, number="A-1111", name="Hydraulic Schematic")
    mock_db_services.drawings.find.return_value = [fake]

    result = drawings_router(
        text="Show Hydraulic Schematic drawing",
        intent="Drawings",
        confidence=0.88,
        entities={"drawing_name": "Hydraulic Schematic"}
    )

    assert result["matched_on"] == "search_by_drawing_name"
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "Hydraulic Schematic"
    assert result["results"][0]["number"] == "A-1111"

    mock_db_services.drawings.find.assert_called_once_with(drw_name="Hydraulic Schematic")


def test_drawings_router_no_match_returns_empty(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.drawings_router import drawings_router

    mock_db_services.drawings.find.return_value = []

    result = drawings_router(
        text="Show drawing NONEXISTENT",
        intent="Drawings",
        confidence=0.92,
        entities={"drawing_number": "NONEXISTENT"}
    )

    assert result["matched_on"] == "no_results"
    assert result["results"] == []
    assert result["related"] is None


def test_drawings_router_matches_by_equipment(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.drawings_router import drawings_router

    fake1 = FakeDrawing(id=30, equipment="Bag Maker", number="A-1000")
    fake2 = FakeDrawing(id=31, equipment="Bag Maker", number="A-1001")

    mock_db_services.drawings.find.return_value = [fake1, fake2]

    result = drawings_router(
        text="Show all Bag Maker drawings",
        intent="Drawings",
        confidence=0.85,
        entities={"equipment_name": "Bag Maker"}
    )

    assert result["matched_on"] == "search_by_equipment"
    assert len(result["results"]) == 2
    assert all(r["equipment_name"] == "Bag Maker" for r in result["results"])

    mock_db_services.drawings.find.assert_called_once_with(drw_equipment_name="Bag Maker")


def test_drawings_router_fuzzy_search_fallback(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.drawings_router import drawings_router

    fake = FakeDrawing(id=40, number="A-2000")
    mock_db_services.drawings.find.side_effect = [[], [fake]]

    result = drawings_router(
        text="Show drawing A-2000",
        intent="Drawings",
        confidence=0.75,
        entities={"drawing_number": "A-2000"}
    )

    assert result["matched_on"] == "search_text"
    assert len(result["results"]) == 1
    assert result["results"][0]["number"] == "A-2000"
    assert mock_db_services.drawings.find.call_count == 2


def test_drawings_router_priority_order(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.drawings_router import drawings_router

    fake_by_number = FakeDrawing(id=70, number="A-5000", name="Primary")
    mock_db_services.drawings.find.return_value = [fake_by_number]
    mock_db_services.drawings.find_related.return_value = None

    result = drawings_router(
        text="Show drawing A-5000 Primary",
        intent="Drawings",
        confidence=0.90,
        entities={
            "drawing_number": "A-5000",
            "drawing_name": "Primary",
        }
    )

    assert result["matched_on"] == "search_by_drawing_number"
    assert result["results"][0]["number"] == "A-5000"

    mock_db_services.drawings.find.assert_called_once_with(drw_number="A-5000")
