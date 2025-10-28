import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from flask import Flask, session
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


class TestUserCreation(unittest.TestCase):
    def setUp(self):
        """Set up test Flask application with proper template paths."""
        self.app = Flask(__name__,
                         template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../templates')))
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.app.secret_key = 'test_secret_key'

        # Register required routes
        @self.app.route('/login')
        def login():
            return "Login page"

        # Import and register the blueprint
        from blueprints.create_user_bp import create_user_bp
        self.app.register_blueprint(create_user_bp)

        # Test client
        self.client = self.app.test_client()

        # Test user data
        self.test_user_data = {
            'employee_id': 'EMP12345',
            'first_name': 'Test',
            'last_name': 'User',
            'current_shift': 'Day',
            'primary_area': 'IT',
            'age': '30',  # String because form data is strings
            'education_level': 'Bachelor',
            'start_date': '2025-01-01',
            'password': 'secure_password',
            'text_to_voice': 'option1',
            'voice_to_text': 'option2'
        }

    @patch('modules.emtacdb.emtacdb_fts.User.create_new_user')
    def test_successful_user_creation(self, mock_create_user):
        """Test successful user creation through the route."""
        # Mock the return value
        mock_create_user.return_value = (True, "User created successfully")

        # Act: Make the POST request with test data
        with self.app.test_client() as client:
            with client.session_transaction() as sess:
                # Set any session data if needed
                pass

            # Test the route
            response = client.post('/submit_user_creation', data=self.test_user_data, follow_redirects=True)

            # Assert: Verify the expected behavior
            mock_create_user.assert_called_once()
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Login page", response.data)  # Should redirect to login page

            # Verify all the arguments were passed correctly
            call_args = mock_create_user.call_args[1]
            self.assertEqual(call_args['employee_id'], self.test_user_data['employee_id'])
            self.assertEqual(call_args['first_name'], self.test_user_data['first_name'])
            self.assertEqual(call_args['last_name'], self.test_user_data['last_name'])
            self.assertEqual(call_args['password'], self.test_user_data['password'])

    @patch('modules.emtacdb.emtacdb_fts.User.create_new_user')
    def test_duplicate_user_creation(self, mock_create_user):
        """Test duplicate user creation error handling."""
        # Mock the return value for duplicate user
        error_message = f"A user with employee ID {self.test_user_data['employee_id']} already exists."
        mock_create_user.return_value = (False, error_message)

        # Act: Make the POST request
        with self.app.test_client() as client:
            # We need with_flashed_messages to capture flash messages
            response = client.post('/submit_user_creation',
                                   data=self.test_user_data,
                                   follow_redirects=True)

            # Assert: Verify behavior
            mock_create_user.assert_called_once()
            self.assertEqual(response.status_code, 200)

            # Test for redirect to registration page (by checking the title)
            self.assertIn(b"User Registration", response.data)

            # To test flash messages, you would need to configure your test client
            # to capture them, but that's a bit more complex

    def test_get_registration_form(self):
        """Test that the registration form page loads correctly."""
        # Act: Make a GET request to the form page
        response = self.client.get('/create_user')

        # Assert: Verify the form page loads
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"User Registration", response.data)
        self.assertIn(b"Employee ID", response.data)
        self.assertIn(b"Password", response.data)

    # Additional test for the core method
    @patch('modules.configuration.config_env.DatabaseConfig')
    @patch('logging.getLogger')
    def test_create_new_user_directly(self, mock_logger, mock_db_config):
        """Test the User.create_new_user method directly."""
        from modules.emtacdb.emtacdb_fts import User

        # Setup mocks
        mock_session = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_instance.get_main_session.return_value = mock_session
        mock_db_config.return_value = mock_db_instance

        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        # Need to patch the User constructor and set_password
        with patch.object(User, '__new__', return_value=MagicMock()) as mock_user_new:
            mock_user = mock_user_new.return_value
            mock_user.set_password = MagicMock()

            # Call the method
            success, message = User.create_new_user(
                employee_id=self.test_user_data['employee_id'],
                first_name=self.test_user_data['first_name'],
                last_name=self.test_user_data['last_name'],
                password=self.test_user_data['password'],
                current_shift=self.test_user_data['current_shift'],
                primary_area=self.test_user_data['primary_area'],
                age=self.test_user_data['age'],
                education_level=self.test_user_data['education_level'],
                start_date=datetime.strptime(self.test_user_data['start_date'], '%Y-%m-%d')
            )

            # Verify correct behavior
            self.assertTrue(success)
            self.assertEqual(message, "User created successfully")
            mock_user.set_password.assert_called_once_with(self.test_user_data['password'])
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()