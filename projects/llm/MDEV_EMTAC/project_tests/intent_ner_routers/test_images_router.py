import sys
import os

PROJECT_ROOT = os.path.abspath("E:/emtac/projects/llm/MDEV_EMTAC")
sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import MagicMock, patch
from importlib import import_module, reload


# -------------------------------------------------------------------
# Fake objects used for return values
# -------------------------------------------------------------------
class FakeImage:
    def __init__(self, id, title="t", desc="d", path="/x"):
        self.id = id
        self.title = title
        self.description = desc
        self.file_path = path


class FakeHybridRow(dict):
    pass


# -------------------------------------------------------------------
# Patch DBServices just like drawings tests
# -------------------------------------------------------------------
@pytest.fixture
def mock_db_services():

    with patch("modules.services.DBServices") as mock_db_class:

        mock_db_instance = MagicMock()

        # Attach the two service layers used in router
        mock_db_instance.images_service = MagicMock()
        mock_db_instance.image_position_service = MagicMock()

        mock_db_class.return_value = mock_db_instance

        # Reload router so it picks up our patched DBServices()
        module = import_module("modules.emtac_ai.intent_ner.routers.images_router")
        reload(module)

        yield mock_db_instance


# -------------------------------------------------------------------
# TEST CASES
# -------------------------------------------------------------------

def test_image_id_lookup(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.images_router import images_router

    fake = FakeImage(10)
    mock_db_services.images_service.get.return_value = fake
    mock_db_services.images_service.find_related.return_value = None

    result = images_router(
        text="",
        intent="Images",
        confidence=0.9,
        entities={"image_id": 10},
    )

    assert result["matched_on"] == "image_id"
    assert result["results"][0]["id"] == 10


def test_title_lookup(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.images_router import images_router

    fake = FakeImage(20)
    mock_db_services.images_service.find.return_value = [fake]

    result = images_router(
        text="X",
        intent="Images",
        confidence=0.9,
        entities={"title": "Pump"},
    )

    assert result["matched_on"] == "title"
    assert result["results"][0]["id"] == 20


def test_file_name_lookup(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.images_router import images_router

    fake = FakeImage(30)
    mock_db_services.images_service.find.return_value = [fake]

    result = images_router(
        text="",
        intent="Images",
        confidence=0.9,
        entities={"file_name": "pump.png"},
    )

    assert result["matched_on"] == "file_name"


def test_position_lookup(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.images_router import images_router

    mock_db_services.image_position_service.get_images_by_position.return_value = [
        FakeImage(3)
    ]

    result = images_router(
        text="",
        intent="Images",
        confidence=0.9,
        entities={"position": 5},
    )

    assert result["matched_on"] == "position"
    assert result["results"][0]["id"] == 3


def test_hierarchy_lookup(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.images_router import images_router

    mock_db_services.image_position_service.get_images_by_hierarchy.return_value = [
        FakeImage(9)
    ]

    result = images_router(
        text="",
        intent="Images",
        confidence=0.9,
        entities={"area_id": 101},
    )

    assert result["matched_on"] == "hierarchy"
    assert result["results"][0]["id"] == 9


def test_complete_document_lookup(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.images_router import images_router

    mock_db_services.images_service.find.return_value = [FakeImage(44)]

    result = images_router(
        text="",
        intent="Images",
        confidence=0.9,
        entities={"complete_document": 12},
    )

    assert result["matched_on"] == "complete_document"
    assert result["results"][0]["id"] == 44


def test_hybrid_fallback(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.images_router import images_router

    mock_db_services.images_service.search_images.return_value = [
        FakeHybridRow(image_id=200, title="X", file_path="x", score=0.8)
    ]

    result = images_router(
        text="search",
        intent="Images",
        confidence=0.9,
        entities={},
    )

    assert result["matched_on"] == "hybrid_pgvector"
    assert result["results"][0]["id"] == 200


def test_basic_find_fallback(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.images_router import images_router

    mock_db_services.images_service.search_images.return_value = []
    mock_db_services.images_service.find.return_value = [FakeImage(88)]

    result = images_router(
        text="Pump",
        intent="Images",
        confidence=0.9,
        entities={},
    )

    assert result["matched_on"] == "basic_find"
    assert result["results"][0]["id"] == 88


def test_no_results(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.images_router import images_router

    mock_db_services.images_service.search_images.return_value = []
    mock_db_services.images_service.find.return_value = []

    result = images_router(
        text="X",
        intent="Images",
        confidence=0.9,
        entities={},
    )

    assert result["matched_on"] == "no_results"
    assert result["results"] == []
