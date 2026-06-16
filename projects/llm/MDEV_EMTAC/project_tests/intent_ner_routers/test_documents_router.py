import sys
import os
import pytest
from unittest.mock import MagicMock, patch
import importlib

# Ensure correct project root
PROJECT_ROOT = os.path.abspath("E:/emtac/projects/llm/MDEV_EMTAC")
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------
# Fake CompleteDocument & Fake Chunk Objects
# ---------------------------------------------------------------------
class FakeCompleteDocument:
    def __init__(self, id=1, title="Doc Title", file_path="doc.pdf", embedding=None):
        self.id = id
        self.title = title
        self.file_path = file_path
        self.embedding = embedding


class FakeChunkRow(dict):
    """Simulates FTS chunk search row"""
    pass


class FakeFTSRow(dict):
    """Simulates full document FTS row"""
    pass


# ---------------------------------------------------------------------
# Pytest fixture for mocking DB services
# ---------------------------------------------------------------------
@pytest.fixture
def mock_db_services():
    with patch("modules.services.DBServices") as mock_db_class:

        mock = MagicMock()

        # Services accessed in the router
        mock.complete_documents = MagicMock()
        mock.documents = MagicMock()

        mock_db_class.return_value = mock

        # Reload module so it picks up the patched DBServices
        dr_module = importlib.import_module(
            "modules.emtac_ai.intent_ner.routers.documents_router"
        )
        importlib.reload(dr_module)

        yield mock


# =====================================================================
# TESTS
# =====================================================================

def test_documents_router_doc_id_lookup(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.documents_router import documents_router

    fake = FakeCompleteDocument(id=10, title="SOP-10")
    mock_db_services.complete_documents.get.return_value = fake
    mock_db_services.complete_documents.find_related.return_value = ["related-doc"]

    result = documents_router(
        text="Show SOP 10",
        intent="Documents",
        confidence=0.95,
        entities={"doc_id": 10}
    )

    assert result["matched_on"] == "doc_id"
    assert result["results"][0]["id"] == 10
    assert result["related"] == ["related-doc"]

    mock_db_services.complete_documents.get.assert_called_once_with(10)
    mock_db_services.complete_documents.find_related.assert_called_once_with(10)


def test_documents_router_matches_title(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.documents_router import documents_router

    fake = FakeCompleteDocument(id=12, title="Safety Procedure")
    mock_db_services.complete_documents.find.return_value = [fake]

    result = documents_router(
        text="Show Safety Procedure",
        intent="Documents",
        confidence=0.9,
        entities={"title": "Safety Procedure"}
    )

    assert result["matched_on"] == "title"
    assert result["results"][0]["title"] == "Safety Procedure"

    mock_db_services.complete_documents.find.assert_called_once_with(title="Safety Procedure")


def test_documents_router_matches_file_name(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.documents_router import documents_router

    fake = FakeCompleteDocument(id=15, title="Welding Guide", file_path="weld.pdf")
    mock_db_services.complete_documents.find.return_value = [fake]

    result = documents_router(
        text="Open weld.pdf",
        intent="Documents",
        confidence=0.88,
        entities={"file_name": "weld.pdf"}
    )

    assert result["matched_on"] == "file_name"
    assert result["results"][0]["file_path"] == "weld.pdf"

    mock_db_services.complete_documents.find.assert_called_once_with(file_path="weld.pdf")


def test_documents_router_matches_position(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.documents_router import documents_router

    fake1 = FakeCompleteDocument(id=21, title="Pump Maintenance")
    fake2 = FakeCompleteDocument(id=22, title="Pump Lubrication")

    mock_db_services.complete_documents.find_by_position.return_value = [fake1, fake2]

    result = documents_router(
        text="Show docs for position 100",
        intent="Documents",
        confidence=0.89,
        entities={"position": 100}
    )

    assert result["matched_on"] == "position"
    assert len(result["results"]) == 2

    mock_db_services.complete_documents.find_by_position.assert_called_once_with(100)


def test_documents_router_complete_document_fts(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.documents_router import documents_router

    mock_db_services.complete_documents.find.return_value = []  # skip earlier steps
    mock_db_services.complete_documents.get.return_value = None
    mock_db_services.complete_documents.find_by_position.return_value = []

    mock_db_services.complete_documents.search_text.return_value = [
        FakeFTSRow({"complete_document_id": 50, "title": "FTS Match", "rank": 0.9})
    ]

    result = documents_router(
        text="Search FTS term",
        intent="Documents",
        confidence=0.8,
        entities={}
    )

    assert result["matched_on"] == "complete_document_fts"
    assert result["results"][0]["id"] == 50
    assert result["results"][0]["title"] == "FTS Match"
    assert result["results"][0]["rank"] == 0.9


def test_documents_router_chunk_fts(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.documents_router import documents_router

    # Skip earlier matches
    mock_db_services.complete_documents.find.return_value = []
    mock_db_services.complete_documents.get.return_value = None
    mock_db_services.complete_documents.find_by_position.return_value = []
    mock_db_services.complete_documents.search_text.return_value = []

    mock_db_services.documents.search_fts.return_value = [
        FakeChunkRow({"id": 200, "document_id": 20, "snippet": "Chunk text", "rank": 0.5})
    ]

    result = documents_router(
        text="Find chunk",
        intent="Documents",
        confidence=0.75,
        entities={}
    )

    assert result["matched_on"] == "document_chunk_fts"
    assert result["results"][0]["chunk_id"] == 200


def test_documents_router_no_results(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.documents_router import documents_router

    # Everything returns nothing
    mock_db_services.complete_documents.get.return_value = None
    mock_db_services.complete_documents.find.return_value = []
    mock_db_services.complete_documents.find_by_position.return_value = []
    mock_db_services.complete_documents.search_text.return_value = []
    mock_db_services.documents.search_fts.return_value = []

    result = documents_router(
        text="Nothing here",
        intent="Documents",
        confidence=0.5,
        entities={}
    )

    assert result["matched_on"] == "no_results"
    assert result["results"] == []


def test_documents_router_priority_doc_id_over_title(mock_db_services):
    from modules.emtac_ai.intent_ner.routers.documents_router import documents_router

    fake_doc = FakeCompleteDocument(id=77, title="Override Title")
    mock_db_services.complete_documents.get.return_value = fake_doc
    mock_db_services.complete_documents.find_related.return_value = None

    result = documents_router(
        text="Show document",
        intent="Documents",
        confidence=0.92,
        entities={
            "doc_id": 77,
            "title": "Some Other Title"
        }
    )

    assert result["matched_on"] == "doc_id"
    mock_db_services.complete_documents.get.assert_called_once_with(77)

    # Ensure title-based find was NOT used
    mock_db_services.complete_documents.find.assert_not_called()
