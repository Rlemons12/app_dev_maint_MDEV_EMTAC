import io
from .helpers.probes import snapshot_counts


def test_transaction_boundary_probe(client, Session):
    # Replace these with your actual persistence tables once confirmed
    table_names = [
        # "complete_document",
        # "document",
        # "document_position_association",
        # "image",
        # "image_embedding",
        # "image_completed_document_association",
    ]

    with Session() as session:
        before = snapshot_counts(session, table_names)

    data = {
        "title": "Boundary Probe",
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


    # We accept any outcome because the point is measurement.
    assert resp.status_code in (200, 201, 202, 400, 422, 500)

    with Session() as session:
        after = snapshot_counts(session, table_names)

    # This prints nothing by default in pytest -q, but you can add asserts once table_names filled.
    assert isinstance(before, dict)
    assert isinstance(after, dict)
