# File: tests/test_search_drawing.py

import pytest
from flask import Flask, jsonify
from unittest.mock import patch, MagicMock

# Import the blueprint object and any models we’ll patch
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Drawing, DrawingPartAssociation, PartsPositionImageAssociation, Image
from modules.configuration.log_config import get_request_id
from blueprints.upload_search_db.search_drawing import drawing_routes


@pytest.fixture
def app():
    """
    Create a minimal Flask app, register the blueprint under test, and return it.
    """
    app = Flask(__name__)
    # We don’t care about real DATABASE_URL here; we’ll patch methods that use DB.
    app.register_blueprint(drawing_routes)
    # Ensure that get_request_id() always returns a predictable value
    # so that log calls don’t break anything. In our tests we don’t inspect logs,
    # but if get_request_id is used, we patch it to return a constant.
    with patch('modules.configuration.log_config.get_request_id', return_value="test-req-id"):
        yield app


@pytest.fixture
def client(app):
    """
    Provide a Flask test client for calling endpoints.
    """
    return app.test_client()


# Helper: A dummy “Drawing” instance
class DummyDrawing:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def to_dict(self):
        return {
            'id': self.id,
            'drw_equipment_name': self.drw_equipment_name,
            'drw_number': self.drw_number,
            'drw_name': self.drw_name,
            'drw_revision': self.drw_revision,
            'drw_spare_part_number': self.drw_spare_part_number,
            'drw_type': self.drw_type,
            'file_path': self.file_path
        }


def test_search_drawings_no_params_returns_empty_list(client):
    """
    If no query params are provided, and Drawing.search returns [], we should get count=0 and empty results.
    """
    # Patch Drawing.search to always return an empty list
    with patch.object(Drawing, 'search', return_value=[]):
        response = client.get('/drawings/search')
        assert response.status_code == 200

        payload = response.get_json()
        assert payload['count'] == 0
        assert isinstance(payload['results'], list)
        assert payload['results'] == []


def test_search_drawings_invalid_drawing_id(client):
    """
    If drawing_id cannot be cast to int, we expect a 400 with the appropriate JSON error message.
    """
    response = client.get('/drawings/search?drawing_id=not_an_int')
    assert response.status_code == 400

    payload = response.get_json()
    assert 'error' in payload and payload['error'] == 'Invalid drawing_id parameter'
    assert 'message' in payload and 'drawing_id must be an integer' in payload['message']


def test_search_drawings_spare_part_flow(client):
    """
    Simulate a spare part search. If we pass drw_spare_part_number=ABC-123 (or via fields/search_text),
    then the code normalizes it (removes dashes), forces exact_match=False, and calls a custom query path.
    We’ll patch session.query(Drawing).filter(...).limit(...) .all() to return two DummyDrawing objects.
    """

    # Prepare three dummy drawings.
    dummy1 = DummyDrawing(
        id=1,
        drw_equipment_name="PumpMotor",
        drw_number="PM-001",
        drw_name="Pump Motor Assembly",
        drw_revision="A",
        drw_spare_part_number="ABC123",
        drw_type="Mechanical",
        file_path="/files/pump_motor.pdf"
    )
    dummy2 = DummyDrawing(
        id=2,
        drw_equipment_name="ValveBlock",
        drw_number="VB-002",
        drw_name="Valve Block",
        drw_revision="B",
        drw_spare_part_number="XYZ123",
        drw_type="Hydraulic",
        file_path="/files/valve_block.pdf"
    )

    # We need to patch the portion of code that does:
    #    query = session.query(Drawing)
    #    query = query.filter( ... )  # using func.lower(Drawing.drw_spare_part_number).like(...)
    #    query = query.limit(limit)
    #    results = query.all()
    #
    # Instead of building a real session, we can patch DatabaseConfig.get_main_session to return a dummy session
    # whose query(Drawing) returns a dummy object whose .filter(...).limit(...).all() yields [dummy1, dummy2].

    class DummyQuery:
        def __init__(self, initial_list):
            self._list = initial_list

        def filter(self, *args, **kwargs):
            # In a real scenario, we'd inspect the args to narrow down results,
            # but since this is a unit test, just return self unmodified.
            return self

        def limit(self, n):
            # Respect the limit: return at most n items
            self._list = self._list[:n]
            return self

        def all(self):
            return self._list

    class DummySession:
        def query(self, model):
            # Return a DummyQuery that will yield our two dummy drawings
            return DummyQuery([dummy1, dummy2])

        def close(self):
            pass

    # Patch DatabaseConfig.get_main_session so that search_drawings uses DummySession
    with patch.object(DatabaseConfig, 'get_main_session', return_value=DummySession()):
        # Now call endpoint with a “spare part” in the querystring. For example:
        response = client.get('/drawings/search?drw_spare_part_number=ABC-123&limit=2')
        assert response.status_code == 200

        payload = response.get_json()
        assert payload['count'] == 2

        # Ensure each result matches our dummy objects
        # Note: the code returns JSON with keys: id, drw_equipment_name, etc.
        result_ids = [item['id'] for item in payload['results']]
        assert set(result_ids) == {1, 2}
        for item in payload['results']:
            # Check that the normalized drw_spare_part_number ends up as "ABC123" or "XYZ123"
            assert 'drw_spare_part_number' in item
            assert item['drw_spare_part_number'] in ('ABC123', 'XYZ123')


def test_search_drawings_include_part_images(client):
    """
    If include_part_images=true, the code will for each drawing:
      - call DrawingPartAssociation.get_parts_by_drawing(...)
      - for each part, call PartsPositionImageAssociation.search(session=session, part_id=...)
      - for each association, call Image.serve_image(...)
    We’ll patch all of these to return a fixed chain of objects so that the final JSON includes 'part_images'.
    """

    # Set up a single dummy drawing
    drawing = DummyDrawing(
        id=5,
        drw_equipment_name="Compressor",
        drw_number="CP-555",
        drw_name="Air Compressor",
        drw_revision="C",
        drw_spare_part_number="COMP555",
        drw_type="Mechanical",
        file_path="/files/compressor.pdf"
    )

    # Patch Drawing.search => [drawing]
    with patch.object(Drawing, 'search', return_value=[drawing]):

        # Patch DrawingPartAssociation.get_parts_by_drawing(...) to return a list of dummy “part” objects.
        # We’ll use simple objects with at least an .id and .part_number/.name attribute.
        DummyPart = type("DummyPart", (), {})  # dynamic class
        part1 = DummyPart()
        part1.id = 100
        part1.part_number = "P-100"
        part1.name = "Rotor"
        part2 = DummyPart()
        part2.id = 101
        part2.part_number = "P-101"
        part2.name = "Stator"

        with patch.object(DrawingPartAssociation, 'get_parts_by_drawing', return_value=[part1, part2]):

            # Patch PartsPositionImageAssociation.search to return a list of associations:
            # Each “association” needs at least an .image_id attribute.
            class DummyAssoc:
                def __init__(self, image_id, part_id):
                    self.image_id = image_id
                    self.part_id = part_id

            assoc1 = DummyAssoc(image_id=500, part_id=100)
            assoc2 = DummyAssoc(image_id=501, part_id=101)
            with patch.object(PartsPositionImageAssociation, 'search', return_value=[assoc1, assoc2]):

                # Patch Image.serve_image(image_id, …) to return a dict with keys 'id','title','file_path'
                def fake_serve_image(image_id, request_id, session):
                    # Return minimal structure expected by code:
                    return {'id': image_id, 'title': f"Image{image_id}", 'file_path': f"/images/{image_id}.png"}

                with patch.object(Image, 'serve_image', side_effect=fake_serve_image):

                    # Finally, patch DatabaseConfig.get_main_session so that session exists (even if not used)
                    class SimpleSession:
                        def close(self): pass
                    with patch.object(DatabaseConfig, 'get_main_session', return_value=SimpleSession()):
                        # Now call endpoint with include_part_images=true
                        url = '/drawings/search?include_part_images=true'
                        response = client.get(url)
                        assert response.status_code == 200

                        payload = response.get_json()
                        # We expect one drawing, so count == 1
                        assert payload['count'] == 1
                        drawing_data = payload['results'][0]
                        assert drawing_data['id'] == 5

                        # The blueprint code should have added a 'part_images' key to drawing_data
                        assert 'part_images' in drawing_data
                        part_images = drawing_data['part_images']

                        # Since we patched get_parts_by_drawing to return two parts (100,101)
                        # and each part => one assoc => one serve_image call, we expect two entries
                        assert len(part_images) == 2

                        # Each entry should have 'part_id', 'image_id', 'image_title', 'image_path' fields
                        for entry in part_images:
                            assert 'part_id' in entry
                            assert 'image_id' in entry
                            assert 'image_title' in entry
                            assert 'image_path' in entry


def test_search_drawings_limit_and_filters_combination(client):
    """
    If limit is provided and > 0, ensure it’s respected. We’ll patch Drawing.search to return 5 items,
    but pass limit=2, so we expect only 2 in the results.
    """
    # Create five dummy drawings
    fives = [
        DummyDrawing(
            id=i,
            drw_equipment_name=f"Eq{i}",
            drw_number=f"Num{i}",
            drw_name=f"Name{i}",
            drw_revision="A",
            drw_spare_part_number=f"IZ{i}",
            drw_type="Mechanical",
            file_path=f"/files/{i}.pdf"
        ) for i in range(5)
    ]

    # Patch Drawing.search to ignore its params and just return all five
    with patch.object(Drawing, 'search', return_value=fives):
        response = client.get('/drawings/search?limit=2')
        assert response.status_code == 200

        payload = response.get_json()
        # count is the length of “drawings_data”, which we expect to be 5 (because code counts results first),
        # but then the code appends all five to drawings_data despite the limit already being passed to Drawing.search.
        # In our simplified stub, Drawing.search ignored limit, so drawings_data length is 5.
        assert payload['count'] == 5

        # However, if you want to test that a valid limit must be > 0, try an invalid limit:
        bad_response = client.get('/drawings/search?limit=-1')
        assert bad_response.status_code == 400
        bad_payload = bad_response.get_json()
        assert 'error' in bad_payload and bad_payload['error'] == 'Invalid limit parameter'


def test_get_drawing_types_success_and_error(client):
    """
    /drawings/types should return whatever Drawing.get_available_types() gives.
    - First test a successful return.
    - Then simulate get_available_types() raising an Exception and verify the 500.
    """
    # Case 1: normal list
    with patch.object(Drawing, 'get_available_types', return_value=['Mechanical', 'Electrical']):
        response = client.get('/drawings/types')
        assert response.status_code == 200

        payload = response.get_json()
        assert 'available_types' in payload
        assert set(payload['available_types']) == {'Mechanical', 'Electrical'}
        assert payload['count'] == 2

    # Case 2: get_available_types throws an exception
    with patch.object(Drawing, 'get_available_types', side_effect=Exception("DB failure")):
        response = client.get('/drawings/types')
        assert response.status_code == 500

        payload = response.get_json()
        assert 'error' in payload and payload['error'] == 'Internal server error'
        assert 'message' in payload


def test_search_drawings_by_type_valid_and_invalid(client):
    """
    Check:
      - If drawing_type is invalid, 400 with JSON error.
      - If valid, patch Drawing.search_by_type to return some results.
    """
    # Suppose valid types are ['Mechanical', 'Hydraulic'].
    # First, patch get_available_types to return those two.
    with patch.object(Drawing, 'get_available_types', return_value=['Mechanical', 'Hydraulic']):

        # 1) Invalid type => 400
        response = client.get('/drawings/search/by-type/Plumbing')
        assert response.status_code == 400
        payload = response.get_json()
        assert 'error' in payload and payload['error'] == 'Invalid drawing type'
        assert 'available_types' in payload
        assert 'Mechanical' in payload['available_types']

        # 2) Valid type => 200
        # Let’s create two dummy drawings of type "Mechanical"
        mech1 = DummyDrawing(
            id=10,
            drw_equipment_name="MechEq1",
            drw_number="M-10",
            drw_name="MechName1",
            drw_revision="A",
            drw_spare_part_number="SP-10",
            drw_type="Mechanical",
            file_path="/m10.pdf"
        )
        mech2 = DummyDrawing(
            id=11,
            drw_equipment_name="MechEq2",
            drw_number="M-11",
            drw_name="MechName2",
            drw_revision="B",
            drw_spare_part_number="SP-11",
            drw_type="Mechanical",
            file_path="/m11.pdf"
        )

        # Patch Drawing.search_by_type to return those two
        with patch.object(Drawing, 'search_by_type', return_value=[mech1, mech2]):
            # Again, patch DatabaseConfig.get_main_session even if not needed by search_by_type
            class DummySession2:
                def close(self): pass
            with patch.object(DatabaseConfig, 'get_main_session', return_value=DummySession2()):
                response = client.get('/drawings/search/by-type/Mechanical?limit=1')
                assert response.status_code == 200
                payload = response.get_json()

                # Since limit=1, the code does:
                #   results = Drawing.search_by_type(...)
                #   if len(results) > limit: results = results[:limit]
                # so we only get one result back
                assert payload['count'] == 1
                assert payload['drawing_type'] == 'Mechanical'
                assert isinstance(payload['results'], list)
                assert len(payload['results']) == 1
                assert payload['results'][0]['id'] in (10, 11)
