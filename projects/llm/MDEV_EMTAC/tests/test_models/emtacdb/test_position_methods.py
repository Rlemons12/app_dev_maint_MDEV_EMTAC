import unittest
from unittest.mock import MagicMock
from sqlalchemy.orm import sessionmaker
from modules.emtacdb.emtacdb_fts import Position  # Assuming the Position class is in the 'models' module


class TestPositionMethods(unittest.TestCase):

    def setUp(self):
        """Set up a mock session and mock Position objects."""
        # Mock session and relationships
        self.mock_session = MagicMock()

        # Create mock Position objects
        self.mock_position_1 = MagicMock(spec=Position)
        self.mock_position_1.id = 1
        self.mock_position_1.area_id = 1
        self.mock_position_1.equipment_group_id = 2
        self.mock_position_1.model_id = 3

        self.mock_position_2 = MagicMock(spec=Position)
        self.mock_position_2.id = 2
        self.mock_position_2.area_id = 1
        self.mock_position_2.equipment_group_id = 2
        self.mock_position_2.model_id = 4

        # Simulating a query returning multiple positions based on filters
        self.mock_session.query.return_value.filter_by.return_value.all.return_value = [self.mock_position_1,
                                                                                        self.mock_position_2]

    def test_get_corresponding_position_ids_with_area_filter(self):
        """Test filtering by area_id."""
        area_id = 1
        position_ids = Position.get_corresponding_position_ids(self.mock_session, area_id=area_id)

        # Verify the query result for the area filter
        self.assertEqual(position_ids, [1, 2])

    def test_get_corresponding_position_ids_with_multiple_filters(self):
        """Test filtering with multiple parameters (area_id, equipment_group_id)."""
        area_id = 1
        equipment_group_id = 2
        position_ids = Position.get_corresponding_position_ids(self.mock_session, area_id=area_id,
                                                               equipment_group_id=equipment_group_id)

        # Ensure that the query is using the correct filters
        self.mock_session.query.assert_called_once_with(Position)
        self.mock_session.query.return_value.filter_by.assert_called_once_with(area_id=area_id,
                                                                               equipment_group_id=equipment_group_id)

        # Ensure the correct position IDs are returned based on these filters
        self.assertEqual(position_ids, [1, 2])

    def test_get_corresponding_position_ids_with_no_match(self):
        """Test when no matching positions exist."""
        # Simulate a situation where no positions match the criteria
        self.mock_session.query.return_value.filter_by.return_value.all.return_value = []
        position_ids = Position.get_corresponding_position_ids(self.mock_session, area_id=999)

        # Assert that no positions are returned if no match is found
        self.assertEqual(position_ids, [])

    def test_get_positions_by_hierarchy_with_multiple_filters(self):
        """Test fetching positions based on multiple filters."""
        area_id = 1
        equipment_group_id = 2
        positions = Position._get_positions_by_hierarchy(self.mock_session, area_id=area_id,
                                                         equipment_group_id=equipment_group_id)

        # Assert that positions are correctly fetched based on the given filters
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0].id, 1)
        self.assertEqual(positions[1].id, 2)

    def test_get_positions_by_hierarchy_with_single_filter(self):
        """Test fetching positions with a single filter."""
        area_id = 1
        positions = Position._get_positions_by_hierarchy(self.mock_session, area_id=area_id)

        # Ensure that only positions with the matching area_id are returned
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0].id, 1)
        self.assertEqual(positions[1].id, 2)

    def test_get_positions_by_hierarchy_with_no_filter(self):
        """Test fetching positions with no filters (all positions)."""
        positions = Position._get_positions_by_hierarchy(self.mock_session)

        # Ensure that the query returns all positions
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0].id, 1)
        self.assertEqual(positions[1].id, 2)


if __name__ == '__main__':
    unittest.main()
