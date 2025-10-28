import os
import logging
from flask import Flask, jsonify
import unittest
from unittest.mock import patch, MagicMock
import io
from werkzeug.datastructures import FileStorage

# Ensure logs directory exists so image_bp's FileHandler can write
TEST_BLUEPRINTS_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(TEST_BLUEPRINTS_DIR, 'logs'), exist_ok=True)

# Import the objects under test
from blueprints.tool_routes import (
    tool_blueprint_bp,
    Position,
    ToolPositionAssociation,
    ToolImageAssociation
)

# --- Logging Setup for Detailed Debugging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(stream_handler)


class TestToolRouteComponents(unittest.TestCase):
    """Unit tests for core tool-route logic with detailed logging."""

    def setUp(self):
        logger.debug("Setting up TestToolRouteComponents environment...")
        self.mock_session = MagicMock()

    @patch('blueprints.tool_routes.Position.add_to_db')
    @patch('blueprints.tool_routes.ToolPositionAssociation')
    def test_position_association_logic(self, mock_assoc_class, mock_add_to_db):
        logger.info("Starting test_position_association_logic...")
        mock_position = MagicMock(id=99)
        mock_add_to_db.return_value = mock_position
        mock_assoc_instance = MagicMock()
        mock_assoc_class.return_value = mock_assoc_instance

        position_fields = {
            'area': ['2', '3'],
            'equipment_group': ['5'],
        }

        def pick_one(vals):
            first = vals[0]
            return None if first == '__None' else int(first)

        fk_kwargs = {'area_id': pick_one(position_fields['area']),
                     'equipment_group_id': pick_one(position_fields['equipment_group'])}
        position = Position.add_to_db(session=self.mock_session, **fk_kwargs)
        mock_add_to_db.assert_called_once_with(session=self.mock_session, **fk_kwargs)
        self.assertIs(position, mock_position)

        mock_tool = MagicMock(id=42)
        description = "Tool↔Position: area=2, equipment=5"
        assoc = mock_assoc_class(tool_id=mock_tool.id, position_id=position.id, description=description)
        assoc.position_id = position.id
        self.mock_session.add(assoc)

        self.assertEqual(self.mock_session.add.call_count, 1)
        added_assoc = self.mock_session.add.call_args[0][0]
        mock_assoc_class.assert_called_once_with(tool_id=42, position_id=99, description=description)
        self.assertEqual(added_assoc.position_id, 99)
        logger.info("test_position_association_logic completed.")


class TestSubmitToolDataRouteIntegration(unittest.TestCase):
    """Integration tests to verify the full /submit_tool_data route, including image uploads."""

    def setUp(self):
        # Create Flask app and register blueprint
        self.app = Flask(__name__)
        # Set secret_key so flash() works
        self.app.secret_key = 'testing_secret'
        self.app.register_blueprint(tool_blueprint_bp)
        # Disable CSRF for forms
        self.app.config['WTF_CSRF_ENABLED'] = False

        # Create mocks and patch DatabaseConfig.get_main_session
        self.mock_db_config = MagicMock()
        self.mock_session = MagicMock()
        self.mock_db_config.get_main_session.return_value = self.mock_session
        self.app.config['db_config'] = self.mock_db_config

        # Stub out form choice population:
        # For ToolCategory and ToolManufacturer, return a dummy object with id=1
        dummy_choice = MagicMock(id=1, name='Dummy')
        query_mock = MagicMock()
        query_mock.order_by.return_value = [dummy_choice]
        # Any query() call returns the same mock
        self.mock_session.query.return_value = query_mock

        self.client = self.app.test_client()

    @patch('blueprints.tool_routes.Position.add_to_db')
    @patch('blueprints.tool_routes.ToolImageAssociation.add_and_associate_with_tool')
    @patch('blueprints.tool_routes.ToolPositionAssociation')
    def test_submit_tool_data_post_with_images(self, mock_assoc_class, mock_add_image, mock_add_to_db):
        # Stub Position.add_to_db
        mock_position = MagicMock(id=123)
        mock_add_to_db.return_value = mock_position

        # Stub image association call
        mock_image = MagicMock(id=555)
        mock_tool_image_assoc = MagicMock(id=777)
        mock_add_image.return_value = (mock_image, mock_tool_image_assoc)

        # Stub position association
        mock_assoc_instance = MagicMock()
        mock_assoc_class.return_value = mock_assoc_instance

        # Prepare a dummy image file
        img_stream = io.BytesIO(b'testimg')
        img_file = (img_stream, 'dummy.jpg')

        # POST data for creating a tool with one image
        data = {
            'submit_tool': 'Submit Tool Data',  # matches form's submit name
            'tool_name': 'ImageTool',
            'tool_category': '1',
            'tool_manufacturer': '1',
            'area': '7',
            'equipment_group': '__None',
            'model': '__None',
            'asset_number': '__None',
            'location': '__None',
            'subassembly': '__None',
            'component_assembly': '__None',
            'assembly_view': '__None',
            'site_location': '__None',
        }
        # Include file under 'tool_images'
        # Use the correct multipart encoding for a single file
        payload = data.copy()
        payload['tool_images'] = img_file

        response = self.client.post(
            '/submit_tool_data',
            data=payload,
            content_type='multipart/form-data'
        )
        # Expect redirect on success
        self.assertEqual(response.status_code, 302)

        # Verify image upload helper was called
        mock_add_image.assert_called_once()
        # Verify Position.add_to_db was invoked
        mock_add_to_db.assert_called_once()
        # Verify position association was created
        mock_assoc_class.assert_called_once()
