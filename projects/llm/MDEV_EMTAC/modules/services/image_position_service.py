from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id, error_id, warning_id, debug_id, with_request_id
)

from modules.emtacdb.emtacdb_fts import (
    Image,
    Position,
    ImagePositionAssociation
)


class ImagePositionService:
    """
    Service layer for managing Image <-> Position associations.

    Provides:
        - associate_image_position()
        - dissociate_image_position()
        - get_positions_by_image()
        - get_images_by_position()
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------------------------------------------------
    # ASSOCIATION CREATION
    # ------------------------------------------------------------------
    @with_request_id
    def associate(self, image_id: int, position_id: int) -> Optional[ImagePositionAssociation]:
        """Create or fetch an association."""
        with self.db_config.main_session() as session:
            try:
                assoc = ImagePositionAssociation.associate_image_position(
                    image_id=image_id,
                    position_id=position_id,
                    session=session
                )
                return assoc
            except SQLAlchemyError as e:
                error_id(f"ImagePositionService.associate failed: {e}")
                raise

    # ------------------------------------------------------------------
    # ASSOCIATION REMOVAL
    # ------------------------------------------------------------------
    @with_request_id
    def dissociate(self, image_id: int, position_id: int) -> bool:
        """Remove an association."""
        with self.db_config.main_session() as session:
            try:
                return ImagePositionAssociation.dissociate_image_position(
                    image_id=image_id,
                    position_id=position_id,
                    session=session
                )
            except SQLAlchemyError as e:
                error_id(f"ImagePositionService.dissociate failed: {e}")
                raise

    # ------------------------------------------------------------------
    # LOOKUP: positions for an image
    # ------------------------------------------------------------------
    @with_request_id
    def get_positions(self, **filters) -> List[Position]:
        """Wrapper for ImagePositionAssociation.get_positions_by_image."""
        with self.db_config.main_session() as session:
            try:
                return ImagePositionAssociation.get_positions_by_image(
                    session=session,
                    **filters
                )
            except SQLAlchemyError as e:
                error_id(f"ImagePositionService.get_positions failed: {e}")
                raise

    # ------------------------------------------------------------------
    # LOOKUP: images for a position
    # ------------------------------------------------------------------
    @with_request_id
    def get_images(self, **filters) -> List[Image]:
        """Wrapper for ImagePositionAssociation.get_images_by_position."""
        with self.db_config.main_session() as session:
            try:
                return ImagePositionAssociation.get_images_by_position(
                    session=session,
                    **filters
                )
            except SQLAlchemyError as e:
                error_id(f"ImagePositionService.get_images failed: {e}")
                raise

