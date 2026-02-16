import io
from modules.emtacdb.emtacdb_fts import CompleteDocument


def test_add_document_docx_path_stubbed(client, monkeypatch):
    """
    Stub DOCX conversion on the actual CompleteDocument class.
    This isolates external Word/COM dependency while letting the real
    upload pipeline execute fully.
    """

    # Stub DOCX → PDF conversion
    monkeypatch.setattr(
        CompleteDocument,
        "_convert_docx_to_pdf",
        lambda cls, docx_path, request_id=None: "fake.pdf",
    )

    # Stub PDF text extraction so content exists
    monkeypatch.setattr(
        CompleteDocument,
        "_extract_pdf_text",
        lambda cls, pdf_path, request_id=None: "Stubbed DOCX content",
    )

    data = {
        "title": "DOCX Upload",
        "area": "A",
        "equipment_group": "B",
        "model": "C",
        "asset_number": "D",
        "location": "E",
        "site_location": "F",
        "files": (io.BytesIO(b"dummy docx content"), "myfile.docx"),
    }

    resp = client.post(
        "/documents/add_document",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert 200 <= resp.status_code < 300, resp.data
