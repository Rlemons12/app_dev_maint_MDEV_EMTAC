import io
from werkzeug.datastructures import FileStorage


def test_add_document_fails_when_copy_fails(client, monkeypatch):

    def boom(*args, **kwargs):
        raise RuntimeError("forced save failure")

    monkeypatch.setattr(FileStorage, "save", boom)

    data = {
        "title": "Rollback Copy Fail",
        "area": "A",
        "equipment_group": "B",
        "model": "C",
        "asset_number": "D",
        "location": "E",
        "site_location": "F",
        "files": (io.BytesIO(b"hello world"), "test.txt"),
    }

    resp = client.post(
        "/documents/add_document",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert resp.status_code >= 400
