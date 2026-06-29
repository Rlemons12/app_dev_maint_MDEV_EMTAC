import io


def test_add_document_with_txt_file(client):
    data = {
        "title": "Test Doc",
        "area": "AreaX",
        "equipment_group": "GroupY",
        "model": "ModelZ",
        "asset_number": "A123",
        "location": "Loc1",
        "site_location": "SiteRoom",
        "files": (io.BytesIO(b"hello world"), "test.txt"),
    }

    resp = client.post(
        "/documents/add_document",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True
    )

    assert 200 <= resp.status_code < 300, resp.data
