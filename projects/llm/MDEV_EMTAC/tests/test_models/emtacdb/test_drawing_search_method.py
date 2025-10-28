import unittest
from unittest.mock import patch, MagicMock, call
import pytest
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import Drawing
from modules.configuration.config_env import DatabaseConfig


class TestDrawingSearch(unittest.TestCase):
    """Comprehensive test suite for Drawing.search method."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock session
        self.mock_session = MagicMock()

        # Create a mock query object
        self.mock_query = MagicMock()
        self.mock_session.query.return_value = self.mock_query

        # Set up the filter and all methods to return the query object for chaining
        self.mock_query.filter.return_value = self.mock_query
        self.mock_query.limit.return_value = self.mock_query

        # Create some sample Drawing objects for test results
        self.drawing1 = MagicMock(spec=Drawing)
        self.drawing1.id = 1
        self.drawing1.drw_number = "DRW-001"
        self.drawing1.drw_name = "Test Drawing 1"
        self.drawing1.drw_equipment_name = "Equipment A"

        self.drawing2 = MagicMock(spec=Drawing)
        self.drawing2.id = 2
        self.drawing2.drw_number = "DRW-002"
        self.drawing2.drw_name = "Test Drawing 2"
        self.drawing2.drw_equipment_name = "Equipment B"

        # Set up mock database config
        self.mock_db_config = MagicMock(spec=DatabaseConfig)
        self.mock_db_config.get_main_session.return_value = self.mock_session

        # Patch the DatabaseConfig to return our mock
        self.db_config_patcher = patch('modules.emtacdb.emtacdb_fts.DatabaseConfig',
                                       return_value=self.mock_db_config)
        self.mock_db_config_class = self.db_config_patcher.start()

        # Patch logging functions to avoid actual logging during tests
        self.debug_id_patcher = patch('modules.emtacdb.emtacdb_fts.debug_id')
        self.mock_debug_id = self.debug_id_patcher.start()

        self.info_id_patcher = patch('modules.emtacdb.emtacdb_fts.info_id')
        self.mock_info_id = self.info_id_patcher.start()

        self.error_id_patcher = patch('modules.emtacdb.emtacdb_fts.error_id')
        self.mock_error_id = self.error_id_patcher.start()

        self.log_timed_operation_patcher = patch('modules.emtacdb.emtacdb_fts.log_timed_operation')
        self.mock_log_timed_operation = self.log_timed_operation_patcher.start()
        self.mock_context_manager = MagicMock()
        self.mock_log_timed_operation.return_value = self.mock_context_manager
        self.mock_context_manager.__enter__ = MagicMock(return_value=None)
        self.mock_context_manager.__exit__ = MagicMock(return_value=None)

        # Patch get_request_id to return a consistent request ID for testing
        self.get_request_id_patcher = patch('modules.emtacdb.emtacdb_fts.get_request_id',
                                            return_value="test-request-id")
        self.mock_get_request_id = self.get_request_id_patcher.start()

    def tearDown(self):
        """Clean up after each test."""
        # Stop all patches
        self.db_config_patcher.stop()
        self.debug_id_patcher.stop()
        self.info_id_patcher.stop()
        self.error_id_patcher.stop()
        self.log_timed_operation_patcher.stop()
        self.get_request_id_patcher.stop()

    def test_search_with_text_default_fields(self):
        """Test searching with text using default fields."""
        # Set up the test
        self.mock_query.all.return_value = [self.drawing1, self.drawing2]

        # Call the method
        results = Drawing.search(search_text="test")

        # Verify the results
        self.assertEqual(results, [self.drawing1, self.drawing2])

        # Verify the query was constructed correctly
        self.mock_session.query.assert_called_once_with(Drawing)

        # Verify proper filtering was applied (checking for ilike on default fields)
        self.mock_query.filter.assert_called_once()
        filter_arg = self.mock_query.filter.call_args[0][0]
        self.assertEqual(str(filter_arg), "drw_number ILIKE :drw_number_1 OR drw_name ILIKE :drw_name_1 OR "
                                          "drw_equipment_name ILIKE :drw_equipment_name_1 OR "
                                          "drw_spare_part_number ILIKE :drw_spare_part_number_1")

        # Verify limit was applied
        self.mock_query.limit.assert_called_once_with(100)

        # Verify session handling
        self.mock_db_config.get_main_session.assert_called_once()
        self.mock_session.close.assert_called_once()

        # Verify logging
        self.mock_debug_id.assert_any_call(
            "Starting Drawing.search with parameters: {'search_text': 'test', 'exact_match': False, 'limit': 100}",
            "test-request-id")
        self.mock_debug_id.assert_any_call("Drawing.search completed, found 2 results", "test-request-id")

    def test_search_with_text_custom_fields(self):
        """Test searching with text in specific fields."""
        # Set up the test
        self.mock_query.all.return_value = [self.drawing1]

        # Call the method with custom fields
        results = Drawing.search(search_text="test", fields=["drw_name", "drw_number"])

        # Verify the results
        self.assertEqual(results, [self.drawing1])

        # Verify the query was constructed correctly
        self.mock_session.query.assert_called_once_with(Drawing)

        # Verify proper filtering was applied (checking for ilike on specified fields only)
        self.mock_query.filter.assert_called_once()
        filter_arg = self.mock_query.filter.call_args[0][0]
        self.assertEqual(str(filter_arg), "drw_name ILIKE :drw_name_1 OR drw_number ILIKE :drw_number_1")

        # Verify logging of custom fields
        self.mock_debug_id.assert_any_call("Searching for text 'test' in fields: ['drw_name', 'drw_number']",
                                           "test-request-id")

    def test_search_with_exact_match(self):
        """Test searching with exact match enabled."""
        # Set up the test
        self.mock_query.all.return_value = [self.drawing1]

        # Call the method with exact_match=True
        results = Drawing.search(search_text="test", exact_match=True)

        # Verify the results
        self.assertEqual(results, [self.drawing1])

        # Verify the query was constructed correctly
        self.mock_session.query.assert_called_once_with(Drawing)

        # Verify proper filtering was applied (checking for exact matches)
        self.mock_query.filter.assert_called_once()
        filter_arg = self.mock_query.filter.call_args[0][0]
        self.assertEqual(str(filter_arg), "drw_number = :drw_number_1 OR drw_name = :drw_name_1 OR "
                                          "drw_equipment_name = :drw_equipment_name_1 OR "
                                          "drw_spare_part_number = :drw_spare_part_number_1")

    def test_search_with_specific_id(self):
        """Test searching for a specific drawing ID."""
        # Set up the test
        self.mock_query.all.return_value = [self.drawing1]

        # Call the method with drawing_id
        results = Drawing.search(drawing_id=1)

        # Verify the results
        self.assertEqual(results, [self.drawing1])

        # Verify the query was constructed correctly
        self.mock_session.query.assert_called_once_with(Drawing)

        # Verify proper filtering was applied (checking for exact id match)
        self.mock_query.filter.assert_called_once()
        filter_arg = self.mock_query.filter.call_args[0][0]
        self.assertEqual(str(filter_arg), "drawing.id = :id_1")

    def test_search_with_specific_fields(self):
        """Test searching with specific field values."""
        # Set up the test
        self.mock_query.all.return_value = [self.drawing1]

        # Call the method with multiple field criteria
        results = Drawing.search(
            drw_number="DRW-001",
            drw_equipment_name="Equipment A",
            exact_match=True
        )

        # Verify the results
        self.assertEqual(results, [self.drawing1])

        # Verify the query was constructed correctly
        self.mock_session.query.assert_called_once_with(Drawing)

        # Verify proper filtering was applied (checking for AND of exact matches)
        self.mock_query.filter.assert_called_once()
        filter_arg = self.mock_query.filter.call_args[0][0]
        self.assertEqual(str(filter_arg),
                         "drawing.drw_number = :drw_number_1 AND drawing.drw_equipment_name = :drw_equipment_name_1")

    def test_search_with_combined_criteria(self):
        """Test searching with both text search and specific fields."""
        # Set up the test
        self.mock_query.all.return_value = [self.drawing1]

        # Call the method with text search and specific fields
        results = Drawing.search(
            search_text="test",
            drw_equipment_name="Equipment A"
        )

        # Verify the results
        self.assertEqual(results, [self.drawing1])

        # Verify proper filtering combination
        self.mock_query.filter.assert_called_once()
        filter_arg = self.mock_query.filter.call_args[0][0]
        self.assertTrue("OR" in str(filter_arg))  # Text search with OR
        self.assertTrue("AND" in str(filter_arg))  # Combined with field with AND

    def test_search_with_custom_limit(self):
        """Test searching with a custom result limit."""
        # Set up the test
        self.mock_query.all.return_value = [self.drawing1]

        # Call the method with a custom limit
        results = Drawing.search(search_text="test", limit=10)

        # Verify limit was applied
        self.mock_query.limit.assert_called_once_with(10)

    def test_search_with_provided_session(self):
        """Test searching with a provided session."""
        # Set up the test
        custom_session = MagicMock()
        custom_query = MagicMock()
        custom_session.query.return_value = custom_query
        custom_query.filter.return_value = custom_query
        custom_query.limit.return_value = custom_query
        custom_query.all.return_value = [self.drawing1]

        # Call the method with a provided session
        results = Drawing.search(search_text="test", session=custom_session)

        # Verify the custom session was used
        custom_session.query.assert_called_once_with(Drawing)

        # Verify our mock session was not used
        self.mock_session.query.assert_not_called()

        # Verify the provided session was not closed
        custom_session.close.assert_not_called()

    def test_search_with_request_id(self):
        """Test searching with a provided request ID."""
        # Set up the test
        self.mock_query.all.return_value = [self.drawing1]

        # Call the method with a custom request ID
        results = Drawing.search(search_text="test", request_id="custom-request-id")

        # Verify request ID was used in logging
        self.mock_debug_id.assert_any_call(
            "Starting Drawing.search with parameters: {'search_text': 'test', 'exact_match': False, 'limit': 100}",
            "custom-request-id"
        )

        # Verify get_request_id was not called (custom ID was used)
        self.mock_get_request_id.assert_not_called()

    def test_search_with_empty_search_text(self):
        """Test searching with empty search text."""
        # Set up the test
        self.mock_query.all.return_value = []

        # Call the method with empty search text
        results = Drawing.search(search_text="")

        # Verify the results
        self.assertEqual(results, [])

        # Verify the query construction
        # It should not have any text search filters since text is empty
        filter_calls = self.mock_query.filter.call_args_list
        self.assertEqual(len(filter_calls), 0)

    def test_search_with_whitespace_search_text(self):
        """Test searching with whitespace-only search text."""
        # Set up the test
        self.mock_query.all.return_value = []

        # Call the method with whitespace search text
        results = Drawing.search(search_text="   ")

        # Verify the results
        self.assertEqual(results, [])

        # Verify the query construction
        # It should not have any text search filters since text is effectively empty
        filter_calls = self.mock_query.filter.call_args_list
        self.assertEqual(len(filter_calls), 0)

    def test_search_with_database_error(self):
        """Test handling of database errors during search."""
        # Set up the test to raise an exception
        self.mock_query.all.side_effect = SQLAlchemyError("Database error")

        # Call the method and expect an exception
        with self.assertRaises(SQLAlchemyError):
            Drawing.search(search_text="test")

        # Verify error logging
        self.mock_error_id.assert_called_once_with(
            "Error in Drawing.search: Database error",
            "test-request-id"
        )

        # Verify session was still closed
        self.mock_session.close.assert_called_once()

    def test_get_by_id_success(self):
        """Test retrieving a drawing by ID successfully."""
        # Set up the test
        self.mock_query.filter.return_value.first.return_value = self.drawing1

        # Call the method
        result = Drawing.get_by_id(1)

        # Verify the result
        self.assertEqual(result, self.drawing1)

        # Verify the query was constructed correctly
        self.mock_session.query.assert_called_once_with(Drawing)
        self.mock_query.filter.assert_called_once()

        # Verify logging
        self.mock_debug_id.assert_any_call("Getting drawing with ID: 1", "test-request-id")
        self.mock_debug_id.assert_any_call(f"Found drawing: {self.drawing1.drw_number} (ID: 1)", "test-request-id")

    def test_get_by_id_not_found(self):
        """Test retrieving a drawing by ID when it doesn't exist."""
        # Set up the test
        self.mock_query.filter.return_value.first.return_value = None

        # Call the method
        result = Drawing.get_by_id(999)

        # Verify the result
        self.assertIsNone(result)

        # Verify logging
        self.mock_debug_id.assert_any_call("No drawing found with ID: 999", "test-request-id")

    def test_get_by_id_with_error(self):
        """Test error handling when retrieving a drawing by ID."""
        # Set up the test to raise an exception
        self.mock_query.filter.return_value.first.side_effect = SQLAlchemyError("Database error")

        # Call the method
        result = Drawing.get_by_id(1)

        # Verify the result is None on error
        self.assertIsNone(result)

        # Verify error logging
        self.mock_error_id.assert_called_once_with(
            "Error retrieving drawing with ID 1: Database error",
            "test-request-id"
        )


if __name__ == '__main__':
    unittest.main()