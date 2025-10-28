import sys
import os

# Ensure `AuMaintdb` is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pytest
from flask import url_for
from ai_emtac import create_app  # Ensure this imports your Flask app factory
from modules.configuration.config_env import DatabaseConfig


@pytest.fixture
def client():
    """Flask test client fixture for handling requests."""
    app = create_app()  # Make sure create_app() initializes the Flask app
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing

    # Provide a mock database config
    db_config = DatabaseConfig()
    app.config['db_config'] = db_config

    with app.test_client() as client:
        with app.app_context():
            db_config.get_main_session().begin()  # Start a test transaction
        yield client


def test_submit_tool_data_get(client):
    """Test GET request to /submit_tool_data"""
    response = client.get('/submit_tool_data')
    assert response.status_code == 200
    assert b"Submit Tool Data" in response.data  # Adjust this based on actual HTML content


def test_submit_tool_data_post(client):
    """Test POST request to /submit_tool_data with valid form data"""
    test_data = {
        "tool_name": "Test Wrench",
        "size": "15mm",
        "type": "Wrench",
        "material": "Steel",
        "description": "A test wrench for unit testing",
        "tool_category": 1,
        "tool_manufacturer": 1,
        "submit_tool": True  # This simulates the submission of the 'submit_tool' button
    }

    response = client.post('/submit_tool_data', data=test_data, follow_redirects=True)

    assert response.status_code == 200  # Ensure the form submission is processed
    assert b"Tool added successfully" in response.data  # Adjust this based on your success message


def test_submit_tool_data_ajax(client):
    """Test AJAX request to /submit_tool_data"""
    response = client.post('/submit_tool_data', json={"submit_tool": True},
                           headers={'X-Requested-With': 'XMLHttpRequest'})

    assert response.status_code in [200, 400, 500]  # Adjust based on actual API behavior
    assert response.is_json  # Ensure JSON response
    json_data = response.get_json()
    assert 'success' in json_data  # Check for success field in response


def test_submit_tool_data_no_db_config(client):
    """Test error handling when database configuration is missing"""
    client.application.config.pop('db_config', None)  # Simulate missing db_config
    response = client.get('/submit_tool_data')

    assert response.status_code == 500  # Expecting server error due to missing database
    assert b"Something went wrong" in response.data  # Adjust error message as per actual response


def test_submit_tool_data_invalid_form(client):
    """Test submitting invalid form data"""
    response = client.post('/submit_tool_data', data={"tool_name": ""}, follow_redirects=True)

    assert response.status_code == 200
    assert b"No valid form submission detected." in response.data  # Check for expected error message
