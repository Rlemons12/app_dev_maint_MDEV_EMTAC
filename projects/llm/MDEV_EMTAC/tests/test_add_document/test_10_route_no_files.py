def test_add_document_no_files(client):
    resp = client.post("/documents/add_document", data={})
    assert resp.status_code in (400, 422, 500)
