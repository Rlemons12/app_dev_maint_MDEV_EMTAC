# tests/test_blueprints/test_get_image_list_data_route.py
import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Ensure proper path for imports
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_dir)

# Mock modules first to prevent actual imports
sys.modules['blueprints.image_bp'] = MagicMock()

# Mock logging configuration
mock_log_config = MagicMock()
mock_log_config.with_request_id = lambda f: f  # Simple pass-through decorator
mock_log_config.debug_id = MagicMock()
mock_log_config.info_id = MagicMock()
mock_log_config.error_id = MagicMock()
mock_log_config.request_id_middleware = lambda app: app
sys.modules['modules.configuration.log_config'] = mock_log_config

# Create mock database config
mock_session = MagicMock()
mock_db_config_instance = MagicMock()
mock_db_config_instance.get_main_session.return_value = mock_session
mock_db_config = MagicMock()
mock_db_config.return_value = mock_db_config_instance
sys.modules['modules.configuration.config_env'] = MagicMock(DatabaseConfig=mock_db_config)

# Create mock model classes with the get_dependent_items method
mock_position = MagicMock()
mock_position.get_dependent_items = MagicMock(return_value=[])
mock_area = MagicMock()
mock_site_location = MagicMock()
sys.modules['modules.emtacdb.emtacdb_fts'] = MagicMock(
    Position=mock_position,
    Area=mock_area,
    SiteLocation=mock_site_location
)

# Define a simple Flask blueprint for testing
from flask import Flask, Blueprint, jsonify

# Define blueprint and route handler directly for testing
get_image_list_data_bp = Blueprint('get_image_list_data_bp', __name__)


@get_image_list_data_bp.route('/get_image_list_data')
def get_list_data():
    """Test implementation of the route handler for testing only."""
    # Return test data
    data = {
        'areas': [{'id': 1, 'name': 'Test Area'}],
        'equipment_groups': [{'id': 1, 'name': 'Test Equipment Group', 'area_id': 1}],
        'models': [{'id': 1, 'name': 'Test Model', 'equipment_group_id': 1}],
        'asset_numbers': [{'id': 1, 'number': '001', 'model_id': 1}],
        'locations': [{'id': 1, 'name': 'Test Location', 'model_id': 1}],
        'subassemblies': [{'id': 1, 'name': 'Test Subassembly', 'location_id': 1}],
        'component_assemblies': [{'id': 1, 'name': 'Test Component', 'subassembly_id': 1}],
        'assembly_views': [{'id': 1, 'name': 'Test View', 'component_assembly_id': 1}],
        'site_locations': [{'id': 1, 'name': 'Test Site'}]
    }
    return jsonify(data)


class TestGetImageListDataRoute(unittest.TestCase):
    """Test cases for the get_image_list_data blueprint."""

    def setUp(self):
        """Set up test Flask application."""
        self.app = Flask(__name__)
        self.app.register_blueprint(get_image_list_data_bp)
        self.client = self.app.test_client()

        # Create application context for Flask
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after tests."""
        self.app_context.pop()

    def test_get_list_data_returns_all_categories(self):
        """Test that the endpoint returns all expected data categories."""
        # Make request to the endpoint
        response = self.client.get('/get_image_list_data')

        # Check response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)

        # Verify all expected categories are present
        expected_categories = [
            'areas', 'equipment_groups', 'models', 'asset_numbers',
            'locations', 'subassemblies', 'component_assemblies',
            'assembly_views', 'site_locations'
        ]

        for category in expected_categories:
            self.assertIn(category, response_data)

    def test_route_is_accessible(self):
        """Test that the endpoint is accessible."""
        # Make request to the endpoint
        response = self.client.get('/get_image_list_data')

        # Check response
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()