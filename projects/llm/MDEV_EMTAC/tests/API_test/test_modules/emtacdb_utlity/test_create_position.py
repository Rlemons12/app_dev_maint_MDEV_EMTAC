import pytest
from unittest.mock import MagicMock, patch
from your_module import create_position  # Replace with your actual module name

@pytest.fixture
def mock_main_session(mocker):
    mock_session = MagicMock()
    mocker.patch('your_module.db_config.MainSession', return_value=mock_session)
    return mock_session

@pytest.fixture
def mock_logger(mocker):
    return mocker.patch('your_module.logger')

def test_create_position_existing(mock_main_session, mock_logger):
    # Arrange
    mock_main_session.query.return_value.filter_by.return_value.first.return_value = MagicMock(id=1)

    # Act
    result = create_position(
        area_id=1,
        equipment_group_id=2,
        model_id=3,
        asset_number_id=4,
        location_id=5,
        site_location_id=6,
        session=None
    )

    # Assert
    assert result == 1
    mock_main_session.query.assert_called()
    mock_main_session.commit.assert_not_called()

def test_create_position_new(mock_main_session, mock_logger, mocker):
    # Arrange
    mock_main_session.query.return_value.filter_by.return_value.first.return_value = None

    # Mock entities
    mock_main_session.query.return_value.filter_by.side_effect = [
        MagicMock(),  # area_entity
        MagicMock(),  # equipment_group_entity
        MagicMock(),  # model_entity
        MagicMock(),  # asset_number_entity
        MagicMock(),  # location_entity
        MagicMock()   # site_location_entity
    ]

    # Mock the Position object
    MockPosition = mocker.patch('your_module.Position')
    mock_position_instance = MagicMock()
    mock_position_instance.id = 2
    MockPosition.return_value = mock_position_instance

    # Act
    result = create_position(
        area_id=1,
        equipment_group_id=2,
        model_id=3,
        asset_number_id=4,
        location_id=5,
        site_location_id=6,
        session=None
    )

    # Assert
    assert result == 2
    mock_main_session.add.assert_called_with(mock_position_instance)
    mock_main_session.commit.assert_called()

def test_create_position_exception(mock_main_session, mock_logger, mocker):
    # Arrange
    mock_main_session.commit.side_effect = Exception("Database Error")

    # Act
    result = create_position(
        area_id=1,
        equipment_group_id=2,
        model_id=3,
        asset_number_id=4,
        location_id=5,
        site_location_id=6,
        session=None
    )

    # Assert
    assert result is None
    mock_main_session.rollback.assert_called()
