import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from importlib import import_module, reload

PROJECT_ROOT = os.path.abspath("E:/emtac/projects/llm/MDEV_EMTAC")
sys.path.insert(0, PROJECT_ROOT)


# -------------------------------------------------------
# Fake Models
# -------------------------------------------------------
class FakePart:
    def __init__(self, id, pn="P-100", name="Valve", oem="OEM"):
        self.id = id
        self.part_number = pn
        self.name = name
        self.oem_mfg = oem
        self.model = None
        self.class_flag = None
        self.notes = None


class FakeFTSRow(dict):
    """Used for FTS fallback tests"""
    pass


# -------------------------------------------------------
# Fixture: Patch DBServices class (not the DB instance)
# -------------------------------------------------------
@pytest.fixture
def mock_db_services():
    """Patch DBServices and reload router so it uses the mock."""
    with patch("modules.services.DBServices") as mock_db_class:
        mock_db_instance = MagicMock()

        # mock sub-services
        mock_db_instance.parts = MagicMock()
        mock_db_instance.drawing_part_associations = MagicMock()

        mock_db_class.return_value = mock_db_instance

        # reload router so it uses our patched DBServices
        mod = import_module("modules.emtac_ai.intent_ner.routers.parts_router")
        reload(mod)

        yield mock_db_instance


# -------------------------------------------------------
# TESTS
# -------------------------------------------------------

def test_by_part_id(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.parts_router import parts_router

    fake = FakePart(10)
    mock_db_services.parts.get.return_value = fake

    r = parts_router(
        text="",
        intent="Parts",
        confidence=0.9,
        entities={"part_id": 10},
    )

    assert r["matched_on"] == "by_part_id"
    assert r["results"][0]["id"] == 10


def test_part_number(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.parts_router import parts_router

    fake = FakePart(20, pn="A-5000")
    mock_db_services.parts.find.return_value = [fake]

    r = parts_router(
        text="",
        intent="Parts",
        confidence=0.9,
        entities={"part_number": "A-5000"},
    )

    assert r["matched_on"] == "by_part_number"
    assert r["results"][0]["part_number"] == "A-5000"


def test_part_name(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.parts_router import parts_router

    fake = FakePart(5, name="Hydraulic Valve")
    mock_db_services.parts.find.return_value = [fake]

    r = parts_router(
        text="",
        intent="Parts",
        confidence=0.9,
        entities={"part_name": "Hydraulic Valve"},
    )

    assert r["matched_on"] == "by_part_name"
    assert r["results"][0]["name"] == "Hydraulic Valve"


def test_oem(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.parts_router import parts_router

    fake = FakePart(99, oem="SMC")
    mock_db_services.parts.find.return_value = [fake]

    r = parts_router(
        text="",
        intent="Parts",
        confidence=0.9,
        entities={"oem_mfg": "SMC"},
    )

    assert r["matched_on"] == "by_oem_mfg"
    assert r["results"][0]["oem_mfg"] == "SMC"


def test_position_lookup(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.parts_router import parts_router

    mock_db_services.parts.find_by_position.return_value = [FakePart(7)]

    r = parts_router(
        text="",
        intent="Parts",
        confidence=0.9,
        entities={"position": 100},
    )

    assert r["matched_on"] == "by_position"
    assert r["results"][0]["id"] == 7


def test_drawing_lookup(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.parts_router import parts_router

    mock_db_services.drawing_part_associations.get_parts_by_drawing.return_value = [FakePart(44)]

    r = parts_router(
        text="",
        intent="Parts",
        confidence=0.9,
        entities={"drawing_id": 5},
    )

    assert r["matched_on"] == "by_drawing"
    assert r["results"][0]["id"] == 44


def test_image_lookup(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.parts_router import parts_router

    mock_db_services.parts.find_by_image.return_value = [FakePart(33)]

    r = parts_router(
        text="",
        intent="Parts",
        confidence=0.9,
        entities={"image_id": 77},
    )

    assert r["matched_on"] == "by_image"
    assert r["results"][0]["id"] == 33


def test_fts_fallback(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.parts_router import parts_router

    mock_db_services.parts.search_text.return_value = [
        FakeFTSRow(part_id=200, part_number="M-1", name="Motor", rank=0.8)
    ]
    mock_db_services.parts.search.return_value = []

    r = parts_router(
        text="motor",
        intent="Parts",
        confidence=0.9,
        entities={},
    )

    assert r["matched_on"] == "fts"
    assert r["results"][0]["id"] == 200
    assert "rank" in r["results"][0]


def test_metadata_fallback(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.parts_router import parts_router

    mock_db_services.parts.search_text.return_value = []
    mock_db_services.parts.search.return_value = [FakePart(55)]

    r = parts_router(
        text="valve",
        intent="Parts",
        confidence=0.9,
        entities={},
    )

    assert r["matched_on"] == "metadata"
    assert r["results"][0]["id"] == 55


def test_no_results(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.parts_router import parts_router

    mock_db_services.parts.search_text.return_value = []
    mock_db_services.parts.search.return_value = []

    r = parts_router(
        text="zzz",
        intent="Parts",
        confidence=0.9,
        entities={},
    )

    assert r["matched_on"] == "no_results"
    assert r["results"] == []