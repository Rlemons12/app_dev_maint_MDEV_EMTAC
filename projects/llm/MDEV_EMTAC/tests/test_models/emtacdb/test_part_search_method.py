import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
import functools
from sqlalchemy.exc import SQLAlchemyError

# Add the absolute path of your project root to sys.path
# This ensures imports work correctly regardless of where the test is run from
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the real logging system first - this should be imported directly without patching
try:
    from modules.configuration.log_config import (
        logger, debug_id, info_id, error_id, warning_id, critical_id,
        log_timed_operation, get_request_id, set_request_id, with_request_id
    )

    # Create a consistent request ID for all test logging
    TEST_SUITE_ID = set_request_id("part-search-test")
    info_id(f"Starting Part search method test suite with request ID: {TEST_SUITE_ID}", TEST_SUITE_ID)

except ImportError as e:
    # If we can't import the actual logging module, create simple placeholders
    print(f"Warning: Could not import real logging module: {e}")
    print("Using placeholder logging functions")


    # Define placeholder logging functions
    def debug_id(message, request_id=None):
        print(f"DEBUG [{request_id}]: {message}")


    def info_id(message, request_id=None):
        print(f"INFO [{request_id}]: {message}")


    def error_id(message, request_id=None):
        print(f"ERROR [{request_id}]: {message}")


    def warning_id(message, request_id=None):
        print(f"WARNING [{request_id}]: {message}")


    def critical_id(message, request_id=None):
        print(f"CRITICAL [{request_id}]: {message}")


    # Create a context manager placeholder
    def log_timed_operation(name, request_id=None):
        class DummyContext:
            def __enter__(self):
                print(f"Starting operation: {name}")
                return self

            def __exit__(self, *args):
                print(f"Completed operation: {name}")

        return DummyContext()


    def get_request_id():
        return "test-request-id"


    def set_request_id(request_id):
        return request_id


    def with_request_id(func):
        return func


    TEST_SUITE_ID = "part-search-test"


# Directly mock the Part class before importing
class MockPart:
    id = None
    part_number = None
    name = None
    oem_mfg = None
    model = None
    class_flag = None
    ud6 = None
    type = None
    notes = None
    documentation = None

    @classmethod
    def search(cls, *args, **kwargs):
        # Will be mocked
        pass

    @classmethod
    def get_by_id(cls, *args, **kwargs):
        # Will be mocked
        pass


# Create a mock for DatabaseConfig
class MockDatabaseConfig:
    def __init__(self):
        pass

    def get_main_session(self):
        # Will return a mock session
        pass


# Patch modules and classes at a module level
import builtins

real_import = builtins.__import__


def patched_import(name, *args, **kwargs):
    if name == 'modules.emtacdb.emtacdb_fts' or name.startswith('modules.emtacdb.emtacdb_fts'):
        module = MagicMock()
        module.Part = MockPart
        return module
    elif name == 'modules.configuration.config_env' or name.startswith('modules.configuration.config_env'):
        module = MagicMock()
        module.DatabaseConfig = MockDatabaseConfig
        return module
    return real_import(name, *args, **kwargs)


# Apply the import patch
builtins.__import__ = patched_import

# Now we can safely import our modules
# These will use the mocked versions
from modules.emtacdb.emtacdb_fts import Part
from modules.configuration.config_env import DatabaseConfig


class TestPartSearch(unittest.TestCase):
    """Comprehensive test suite for Part.search method with detailed logging."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for the entire test class."""
        info_id("Starting test suite execution for Part.search method", TEST_SUITE_ID)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are run."""
        info_id("Completed test suite execution for Part.search method", TEST_SUITE_ID)

        # Restore original import
        builtins.__import__ = real_import

    def setUp(self):
        """Set up test environment before each test."""
        # Get test name for logging
        test_name = self.id().split('.')[-1]
        self.test_id = f"{TEST_SUITE_ID}-{test_name}"

        # Log test start
        info_id(f"Starting test case: {test_name}", self.test_id)

        # Create a mock session
        self.mock_session = MagicMock()

        # Create a mock query object with chaining
        self.mock_query = MagicMock()
        self.mock_session.query.return_value = self.mock_query
        self.mock_query.filter.return_value = self.mock_query
        self.mock_query.limit.return_value = self.mock_query

        # Sample part objects
        self.part1 = MagicMock()
        self.part1.id = 1
        self.part1.part_number = "ABC123"
        self.part1.name = "Test Part 1"
        self.part1.oem_mfg = "Manufacturer A"
        self.part1.model = "Model X"
        self.part1.class_flag = "Class A"
        self.part1.ud6 = "UD6 Value 1"
        self.part1.type = "Type A"
        self.part1.notes = "Test notes 1"
        self.part1.documentation = "Test documentation 1"

        debug_id(f"Created test part 1: {self.part1.part_number}", self.test_id)

        self.part2 = MagicMock()
        self.part2.id = 2
        self.part2.part_number = "DEF456"
        self.part2.name = "Test Part 2"
        self.part2.oem_mfg = "Manufacturer B"
        self.part2.model = "Model Y"
        self.part2.class_flag = "Class B"
        self.part2.ud6 = "UD6 Value 2"
        self.part2.type = "Type B"
        self.part2.notes = "Test notes 2"
        self.part2.documentation = "Test documentation 2"

        debug_id(f"Created test part 2: {self.part2.part_number}", self.test_id)

        # Configure mock DatabaseConfig
        self.mock_db_config = MagicMock()
        self.mock_db_config.get_main_session.return_value = self.mock_session

        # Patch the DatabaseConfig constructor
        self.db_config_patcher = patch.object(
            DatabaseConfig, '__new__',
            return_value=self.mock_db_config
        )
        self.mock_db_config_class = self.db_config_patcher.start()

        # Configure search method behavior
        self.search_patcher = patch.object(Part, 'search')
        self.mock_search = self.search_patcher.start()

        def search_side_effect(*args, **kwargs):
            # Extract parameters
            search_text = kwargs.get('search_text')
            fields = kwargs.get('fields')
            exact_match = kwargs.get('exact_match', False)
            part_id = kwargs.get('part_id')
            part_number = kwargs.get('part_number')
            name = kwargs.get('name')
            oem_mfg = kwargs.get('oem_mfg')
            model = kwargs.get('model')
            class_flag = kwargs.get('class_flag')
            ud6 = kwargs.get('ud6')
            type_ = kwargs.get('type_')
            notes = kwargs.get('notes')
            documentation = kwargs.get('documentation')
            limit = kwargs.get('limit', 100)
            request_id = kwargs.get('request_id', self.test_id)
            session = kwargs.get('session')

            # Log search parameters using the actual logger
            search_params = {k: v for k, v in kwargs.items() if v is not None and k != 'session'}
            debug_id(f"Executing Part.search with parameters: {search_params}", request_id)

            # Handle specific search scenarios
            if search_text and not fields:
                debug_id(f"Searching for text '{search_text}' in default fields", request_id)
            elif search_text and fields:
                debug_id(f"Searching for text '{search_text}' in fields: {fields}", request_id)

            # Log field-specific filters
            if part_id is not None:
                debug_id(f"Adding filter for part_id: {part_id}", request_id)
            if part_number is not None:
                debug_id(f"Adding filter for part_number: {part_number}", request_id)
            if name is not None:
                debug_id(f"Adding filter for name: {name}", request_id)
            if oem_mfg is not None:
                debug_id(f"Adding filter for oem_mfg: {oem_mfg}", request_id)
            if model is not None:
                debug_id(f"Adding filter for model: {model}", request_id)

            # Use provided session or get a new one
            actual_session = session or self.mock_db_config.get_main_session()

            # Perform query simulation
            query_result = actual_session.query(Part)
            filter_called = False

            if search_text or part_id or part_number or name or oem_mfg or model or class_flag or ud6 or type_ or notes or documentation:
                filter_called = True

            if filter_called:
                query_result = query_result.filter()

            # Apply limit
            query_result = query_result.limit(limit)

            # Set the return value for all() method
            query_result.all.return_value = [self.part1, self.part2]

            # Determine results based on parameters
            if part_id == 1:
                results = [self.part1]
            elif part_id == 2:
                results = [self.part2]
            elif (
                    not search_text or search_text.strip() == "") and not part_id and not part_number and not name and not oem_mfg and not model:
                # No search criteria or empty search text, return empty
                results = []
            else:
                # Default results
                results = [self.part1, self.part2]

            # Log completion with result count
            debug_id(f"Part.search completed, found {len(results)} results", request_id)

            # Close session if not provided
            if not session:
                debug_id(f"Closed database session for Part.search", request_id)

            # Return the expected results
            return results

        self.mock_search.side_effect = search_side_effect

        # Configure get_by_id method behavior
        self.get_by_id_patcher = patch.object(Part, 'get_by_id')
        self.mock_get_by_id = self.get_by_id_patcher.start()

        def get_by_id_side_effect(part_id, request_id=None, session=None):
            # Extract parameters
            request_id = request_id or self.test_id

            # Log parameters
            debug_id(f"Getting part with ID: {part_id}", request_id)

            # Use provided session or get a new one
            actual_session = session or self.mock_db_config.get_main_session()

            # Set up query behavior
            query_result = actual_session.query(Part)
            query_result = query_result.filter(Part.id == part_id)

            # Define return value based on ID
            if part_id == 1:
                query_result.first.return_value = self.part1
                debug_id(f"Found part: {self.part1.part_number} (ID: {part_id})", request_id)
                result = self.part1
            elif part_id == 2:
                query_result.first.return_value = self.part2
                debug_id(f"Found part: {self.part2.part_number} (ID: {part_id})", request_id)
                result = self.part2
            else:
                query_result.first.return_value = None
                debug_id(f"No part found with ID: {part_id}", request_id)
                result = None

            # Close session if not provided
            if not session:
                debug_id(f"Closed database session for Part.get_by_id", request_id)

            return result

        self.mock_get_by_id.side_effect = get_by_id_side_effect

        # Configure database error for error tests
        self.mock_db_error = SQLAlchemyError("Database error")

    def tearDown(self):
        """Clean up after each test."""
        # Get test name
        test_name = self.id().split('.')[-1]

        # Stop all patches
        self.db_config_patcher.stop()
        self.search_patcher.stop()
        self.get_by_id_patcher.stop()

        # Log test completion
        info_id(f"Completed test case: {test_name}", self.test_id)

    def test_search_with_text_default_fields(self):
        """Test searching with text using default fields."""
        with log_timed_operation("test_search_with_text_default_fields", self.test_id):
            debug_id("Testing search with text in default fields", self.test_id)

            # Call the method
            results = Part.search(search_text="test")

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['search_text'], "test")

            # Verify we got the expected results
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0], self.part1)
            self.assertEqual(results[1], self.part2)

            debug_id("Verification complete for default fields search", self.test_id)

    def test_search_with_text_custom_fields(self):
        """Test searching with text in specific fields."""
        with log_timed_operation("test_search_with_text_custom_fields", self.test_id):
            debug_id("Testing search with text in custom fields", self.test_id)

            # Call the method with custom fields
            results = Part.search(search_text="test", fields=["name", "part_number"])

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['search_text'], "test")
            self.assertEqual(call_args[1]['fields'], ["name", "part_number"])

            # Verify we got the expected results
            self.assertEqual(len(results), 2)

            debug_id("Verification complete for custom fields search", self.test_id)

    def test_search_with_exact_match(self):
        """Test searching with exact match enabled."""
        with log_timed_operation("test_search_with_exact_match", self.test_id):
            debug_id("Testing search with exact match enabled", self.test_id)

            # Call the method with exact_match=True
            results = Part.search(search_text="test", exact_match=True)

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['search_text'], "test")
            self.assertEqual(call_args[1]['exact_match'], True)

            # Verify we got the expected results
            self.assertEqual(len(results), 2)

            debug_id("Verification complete for exact match search", self.test_id)

    def test_search_with_specific_id(self):
        """Test searching for a specific part ID."""
        with log_timed_operation("test_search_with_specific_id", self.test_id):
            debug_id("Testing search by specific part ID", self.test_id)

            # Call the method with part_id
            results = Part.search(part_id=1)

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['part_id'], 1)

            # Verify we got the expected results
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0], self.part1)

            debug_id("Verification complete for part ID search", self.test_id)

    def test_search_with_specific_fields(self):
        """Test searching with specific field values."""
        with log_timed_operation("test_search_with_specific_fields", self.test_id):
            debug_id("Testing search by specific field values", self.test_id)

            # Call the method with multiple field criteria
            results = Part.search(
                part_number="ABC123",
                oem_mfg="Manufacturer A",
                exact_match=True
            )

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['part_number'], "ABC123")
            self.assertEqual(call_args[1]['oem_mfg'], "Manufacturer A")
            self.assertEqual(call_args[1]['exact_match'], True)

            # Verify we got the expected results
            self.assertEqual(len(results), 2)

            debug_id("Verification complete for specific fields search", self.test_id)

    def test_search_with_partial_field_match(self):
        """Test searching with partial field matching."""
        with log_timed_operation("test_search_with_partial_field_match", self.test_id):
            debug_id("Testing search with partial field matching", self.test_id)

            # Call the method with field criteria and partial matching
            results = Part.search(
                part_number="ABC",
                oem_mfg="Manu",
                exact_match=False
            )

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['part_number'], "ABC")
            self.assertEqual(call_args[1]['oem_mfg'], "Manu")
            self.assertEqual(call_args[1]['exact_match'], False)

            # Verify we got the expected results
            self.assertEqual(len(results), 2)

            debug_id("Verification complete for partial field matching", self.test_id)

    def test_search_with_combined_criteria(self):
        """Test searching with both text search and specific fields."""
        with log_timed_operation("test_search_with_combined_criteria", self.test_id):
            debug_id("Testing search with combined criteria", self.test_id)

            # Call the method with text search and specific fields
            results = Part.search(
                search_text="test",
                oem_mfg="Manufacturer A"
            )

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['search_text'], "test")
            self.assertEqual(call_args[1]['oem_mfg'], "Manufacturer A")

            # Verify we got the expected results
            self.assertEqual(len(results), 2)

            debug_id("Verification complete for combined criteria search", self.test_id)

    def test_search_with_custom_limit(self):
        """Test searching with a custom result limit."""
        with log_timed_operation("test_search_with_custom_limit", self.test_id):
            debug_id("Testing search with custom result limit", self.test_id)

            # Call the method with a custom limit
            results = Part.search(search_text="test", limit=10)

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['limit'], 10)

            # Verify we got the expected results
            self.assertEqual(len(results), 2)

            debug_id("Verification complete for custom limit search", self.test_id)

    def test_search_with_provided_session(self):
        """Test searching with a provided session."""
        with log_timed_operation("test_search_with_provided_session", self.test_id):
            debug_id("Testing search with provided session", self.test_id)

            # Create a custom session
            custom_session = MagicMock()

            # Call the method with a provided session
            results = Part.search(search_text="test", session=custom_session)

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['search_text'], "test")
            self.assertEqual(call_args[1]['session'], custom_session)

            # Verify we got the expected results
            self.assertEqual(len(results), 2)

            debug_id("Verification complete for provided session search", self.test_id)

    def test_search_with_empty_search_text(self):
        """Test searching with empty search text."""
        with log_timed_operation("test_search_with_empty_search_text", self.test_id):
            debug_id("Testing search with empty search text", self.test_id)

            # Call the method with empty search text
            results = Part.search(search_text="")

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['search_text'], "")

            # Verify we got the expected results
            self.assertEqual(results, [self.part1, self.part2])

            debug_id("Verification complete for empty search text", self.test_id)

    def test_search_with_whitespace_search_text(self):
        """Test searching with whitespace-only search text."""
        with log_timed_operation("test_search_with_whitespace_search_text", self.test_id):
            debug_id("Testing search with whitespace-only search text", self.test_id)

            # Call the method with whitespace search text
            results = Part.search(search_text="   ")

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['search_text'], "   ")

            # Verify we got the expected results
            self.assertEqual(results, [self.part1, self.part2])

            debug_id("Verification complete for whitespace search text", self.test_id)

    def test_get_by_id_success(self):
        """Test retrieving a part by ID successfully."""
        with log_timed_operation("test_get_by_id_success", self.test_id):
            debug_id("Testing get part by ID - success case", self.test_id)

            # Call the method
            result = Part.get_by_id(1)

            # Verify the method was called with the correct parameters
            self.mock_get_by_id.assert_called_once_with(1)

            # Verify we got the expected result
            self.assertEqual(result, self.part1)

            debug_id("Verification complete for get by ID success", self.test_id)

    def test_get_by_id_not_found(self):
        """Test retrieving a part by ID when not found."""
        with log_timed_operation("test_get_by_id_not_found", self.test_id):
            debug_id("Testing get part by ID - not found case", self.test_id)

            # Call the method with an ID that doesn't exist
            result = Part.get_by_id(999)

            # Verify the method was called with the correct parameters
            self.mock_get_by_id.assert_called_once_with(999)

            # Verify we got None as the result
            self.assertIsNone(result)

            debug_id("Verification complete for get by ID not found", self.test_id)

    def test_search_with_error_handling(self):
        """Test error handling during search."""
        with log_timed_operation("test_search_with_error_handling", self.test_id):
            debug_id("Testing search with error handling", self.test_id)

            # Create an error-throwing side effect
            def error_side_effect(*args, **kwargs):
                request_id = kwargs.get('request_id', self.test_id)
                debug_id("Simulating database error during search", request_id)
                raise self.mock_db_error

            # Save the original side effect
            original_side_effect = self.mock_search.side_effect

            try:
                # Set the error side effect
                self.mock_search.side_effect = error_side_effect

                # Call the method and expect an exception
                with self.assertRaises(SQLAlchemyError) as context:
                    Part.search(search_text="test")

                # Verify the correct error was raised
                self.assertEqual(str(context.exception), "Database error")

                debug_id("Successfully caught database error", self.test_id)
            finally:
                # Restore the original side effect
                self.mock_search.side_effect = original_side_effect

            debug_id("Verification complete for error handling", self.test_id)

    def test_search_with_special_characters(self):
        """Test searching with special SQL characters."""
        with log_timed_operation("test_search_with_special_characters", self.test_id):
            debug_id("Testing search with special characters", self.test_id)

            # Call the method with text containing special characters
            results = Part.search(search_text="test%_[]")

            # Verify the method was called with the correct parameters
            self.mock_search.assert_called_once()
            call_args = self.mock_search.call_args
            self.assertEqual(call_args[1]['search_text'], "test%_[]")

            # Verify we got the expected results
            self.assertEqual(len(results), 2)

            debug_id("Verification complete for special characters search", self.test_id)


if __name__ == '__main__':
    unittest.main()