# tests/test_models/emtacdb/test_image_tool_association.py

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import your models and configuration
from modules.emtacdb.emtacdb_fts import Base, Image, ToolImageAssociation, Tool


class TestImageToolAssociationMethods(unittest.TestCase):
    """Test cases for ToolImageAssociation class methods related to tool associations"""

    def setUp(self):
        """Set up test database and session"""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test.db")

        # Create test database
        self.engine = create_engine(f'sqlite:///{self.test_db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # Create test directories
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)

        # Create a test image file
        self.test_img_path = os.path.join(self.temp_dir, "test_image.jpg")
        with open(self.test_img_path, 'wb') as f:
            f.write(b'test image content')

        # Create a test image and tool for testing
        self.test_image = Image(
            title='Test Image',
            description='Test Description',
            file_path=self.test_img_path
        )
        self.session.add(self.test_image)

        self.test_tool = Tool(
            name='Test Tool',
            description='Test Tool Description'
        )
        self.session.add(self.test_tool)
        self.session.commit()

    def tearDown(self):
        """Clean up after tests"""
        self.session.close()

        # Close the engine connection before removing files
        self.engine.dispose()

        try:
            # Remove temporary directory with error handling
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not remove temp directory: {e}")

    def test_associate_with_tool(self):
        """Test the associate_with_tool method"""
        # Call the method to associate image with tool
        association = ToolImageAssociation.associate_with_tool(
            self.session,
            self.test_image.id,
            self.test_tool.id,
            description="Test association"
        )

        # Verify the association was created
        self.assertIsNotNone(association)
        self.assertEqual(association.image_id, self.test_image.id)
        self.assertEqual(association.tool_id, self.test_tool.id)
        self.assertEqual(association.description, "Test association")

        # Verify it's in the database
        self.session.commit()
        retrieved_assoc = self.session.query(ToolImageAssociation).filter_by(
            image_id=self.test_image.id,
            tool_id=self.test_tool.id
        ).first()

        self.assertIsNotNone(retrieved_assoc)
        self.assertEqual(retrieved_assoc.description, "Test association")

    def test_associate_with_tool_existing_association(self):
        """Test associating when an association already exists"""
        # Create an initial association
        initial_assoc = ToolImageAssociation(
            image_id=self.test_image.id,
            tool_id=self.test_tool.id,
            description="Initial description"
        )
        self.session.add(initial_assoc)
        self.session.commit()

        # Call the method again with updated description
        updated_assoc = ToolImageAssociation.associate_with_tool(
            self.session,
            self.test_image.id,
            self.test_tool.id,
            description="Updated description"
        )

        # Verify the association was updated, not duplicated
        self.session.commit()
        associations = self.session.query(ToolImageAssociation).filter_by(
            image_id=self.test_image.id,
            tool_id=self.test_tool.id
        ).all()

        # Should only be one association
        self.assertEqual(len(associations), 1)
        # Description should be updated
        self.assertEqual(associations[0].description, "Updated description")
        # Should return the existing association
        self.assertEqual(updated_assoc.id, initial_assoc.id)

    def test_add_and_associate_with_tool(self):
        """Test adding a new image and associating it with a tool in one step"""
        # Create a new test image file
        new_img_path = os.path.join(self.temp_dir, "new_test_image.jpg")
        with open(new_img_path, 'wb') as f:
            f.write(b'new test image content')

        # Mock the add_to_db method to avoid complexity
        with patch('modules.emtacdb.emtacdb_fts.Image.add_to_db') as mock_add_to_db:
            mock_image = MagicMock(spec=Image)
            mock_image.id = 999
            mock_add_to_db.return_value = mock_image

            # Mock the associate_with_tool method
            with patch.object(ToolImageAssociation, 'associate_with_tool') as mock_associate:
                mock_assoc = MagicMock(spec=ToolImageAssociation)
                mock_assoc.id = 888
                mock_associate.return_value = mock_assoc

                # Call the method
                image, association = ToolImageAssociation.add_and_associate_with_tool(
                    self.session,
                    "New Image",
                    new_img_path,
                    self.test_tool.id,
                    description="New image description",
                    association_description="New association description"
                )

                # Verify add_to_db was called with correct args
                mock_add_to_db.assert_called_once_with(
                    self.session,
                    "New Image",
                    new_img_path,
                    "New image description"
                )

                # Verify associate_with_tool was called with correct args
                mock_associate.assert_called_once_with(
                    self.session,
                    image_id=mock_image.id,
                    tool_id=self.test_tool.id,
                    description="New association description"
                )

                # Verify returned objects
                self.assertEqual(image, mock_image)
                self.assertEqual(association, mock_assoc)

    def test_add_and_associate_with_tool_integration(self):
        """Integration test for add_and_associate_with_tool without mocking"""
        # Create a new test image file
        new_img_path = os.path.join(self.temp_dir, "integration_test_image.jpg")
        with open(new_img_path, 'wb') as f:
            f.write(b'integration test image content')

        # Mock only Image.add_to_db to avoid dependency issues
        with patch('modules.emtacdb.emtacdb_fts.Image.add_to_db') as mock_add_to_db:
            # Create a mock image to return
            mock_image = MagicMock(spec=Image)
            mock_image.id = 777
            mock_image.title = "Integration Test Image"
            mock_image.description = "Integration test description"
            mock_add_to_db.return_value = mock_image

            # Call the method to add and associate
            image, association = ToolImageAssociation.add_and_associate_with_tool(
                self.session,
                "Integration Test Image",
                new_img_path,
                self.test_tool.id,
                description="Integration test description",
                association_description="Integration test association"
            )

            # Verify add_to_db was called with correct arguments
            mock_add_to_db.assert_called_once_with(
                self.session,
                "Integration Test Image",
                new_img_path,
                "Integration test description"
            )

            # Verify the image was created (returned from mock)
            self.assertEqual(image.id, 777)
            self.assertEqual(image.title, "Integration Test Image")
            self.assertEqual(image.description, "Integration test description")

            # Verify the association was created
            self.assertIsNotNone(association)
            self.assertEqual(association.image_id, image.id)
            self.assertEqual(association.tool_id, self.test_tool.id)
            self.assertEqual(association.description, "Integration test association")

            # Verify association is in the database
            self.session.commit()
            retrieved_assoc = self.session.query(ToolImageAssociation).filter_by(
                image_id=image.id,
                tool_id=self.test_tool.id
            ).first()
            self.assertIsNotNone(retrieved_assoc)


if __name__ == '__main__':
    unittest.main()