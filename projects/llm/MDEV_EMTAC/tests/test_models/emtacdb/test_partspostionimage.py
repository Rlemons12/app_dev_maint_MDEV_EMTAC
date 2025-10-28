import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import PartsPositionImageAssociation


# Setup the mock session and mock logging
@pytest.fixture
def mock_session():
    """Fixture to mock SQLAlchemy session"""
    session = MagicMock()
    yield session
    session.close()


# Test for the search method
def test_search(mock_session, caplog):
    filters = {'part_id': 1, 'position_id': 2}

    # Mocking the query to return a mock result
    mock_result = [MagicMock(id=1), MagicMock(id=2)]
    mock_session.query.return_value.filter_by.return_value.all.return_value = mock_result

    # Call the search method
    PartsPositionImageAssociation.search(mock_session, **filters)

    # Assertions
    mock_session.query.return_value.filter_by.assert_called_with(**filters)
    assert len(mock_session.query.return_value.filter_by.return_value.all.return_value) == 2

    # Check if logging contains expected messages
    assert "Starting search with filters" in caplog.text
    assert "Search returned 2 result(s)" in caplog.text
    assert "Completed search in" in caplog.text


# Test for get_corresponding_position_ids method
def test_get_corresponding_position_ids(mock_session, caplog):
    area_id = 1
    equipment_group_id = 2

    # Mocking _get_positions_by_hierarchy to return mock results
    mock_result = [MagicMock(id=1), MagicMock(id=2)]
    mock_session.query.return_value.filter_by.return_value.all.return_value = mock_result

    # Call the method
    PartsPositionImageAssociation.get_corresponding_position_ids(
        mock_session, area_id=area_id, equipment_group_id=equipment_group_id
    )

    # Assertions
    assert len(mock_session.query.return_value.filter_by.return_value.all.return_value) == 2

    # Check if logging contains expected messages
    assert "Starting get_corresponding_position_ids" in caplog.text
    assert "Found 2 Position IDs" in caplog.text
    assert "Completed get_corresponding_position_ids" in caplog.text


# Test for _get_positions_by_hierarchy method
def test_get_positions_by_hierarchy(mock_session, caplog):
    filters = {'area_id': 1, 'equipment_group_id': 2}

    # Mocking the query to return mock results
    mock_result = [MagicMock(id=1), MagicMock(id=2)]
    mock_session.query.return_value.filter_by.return_value.all.return_value = mock_result

    # Call the method
    PartsPositionImageAssociation._get_positions_by_hierarchy(mock_session, **filters)

    # Assertions
    assert len(mock_session.query.return_value.filter_by.return_value.all.return_value) == 2

    # Check if logging contains expected messages
    assert "Applying filters to query" in caplog.text
    assert "Found 2 positions" in caplog.text
    assert "Completed _get_positions_by_hierarchy" in caplog.text


# Test for error handling in search method
def test_search_error(mock_session, caplog):
    filters = {'part_id': 1, 'position_id': 2}

    # Simulate SQLAlchemy error
    mock_session.query.side_effect = SQLAlchemyError("Database error")

    with pytest.raises(SQLAlchemyError):
        PartsPositionImageAssociation.search(mock_session, **filters)

    # Check if logging contains the error message
    assert "Error during search operation" in caplog.text


# Test for error handling in get_corresponding_position_ids method
def test_get_corresponding_position_ids_error(mock_session, caplog):
    # Simulate SQLAlchemy error
    mock_session.query.side_effect = SQLAlchemyError("Database error")

    with pytest.raises(SQLAlchemyError):
        PartsPositionImageAssociation.get_corresponding_position_ids(mock_session)

    # Check if logging contains the error message
    assert "Error during get_corresponding_position_ids" in caplog.text


# Test for logging in _get_positions_by_hierarchy method
def test_get_positions_by_hierarchy_logging(mock_session, caplog):
    filters = {'area_id': 1, 'equipment_group_id': 2}

    # Mocking the query to return mock results
    mock_result = [MagicMock(id=1), MagicMock(id=2)]
    mock_session.query.return_value.filter_by.return_value.all.return_value = mock_result

    # Call the method
    PartsPositionImageAssociation._get_positions_by_hierarchy(mock_session, **filters)

    # Assertions
    assert len(mock_session.query.return_value.filter_by.return_value.all.return_value) == 2

    # Check if logging contains expected messages
    assert "Applying filters to query" in caplog.text
    assert "Found 2 positions" in caplog.text
    assert "Completed _get_positions_by_hierarchy" in caplog.text
