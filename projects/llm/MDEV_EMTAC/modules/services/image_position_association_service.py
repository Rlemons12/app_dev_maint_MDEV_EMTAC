# modules/services/image_position_association_service.py

from typing import Optional, List, Dict, Any, Tuple

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as SASession

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id,
    error_id,
    warning_id,
    debug_id,
    with_request_id,
    log_timed_operation,
    get_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    ImagePositionAssociation,
    Position,
    Image,
)


class ImagePositionAssociationService:
    """
    Service layer for ImagePositionAssociation.

    Responsibilities:
    - High-level associate/dissociate operations
    - Find positions for a given image
    - Find images for a given position
    - Handle logging, sessions, and error management
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    def _get_session(self, session: Optional[SASession]) -> Tuple[SASession, bool]:
        if session is not None:
            return session, False
        return self.db_config.get_main_session(), True

    # ---------------------------------------------------------
    # Associate / dissociate
    # ---------------------------------------------------------
    @with_request_id
    def associate(
        self,
        image_id: int,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> Optional[ImagePositionAssociation]:
        """
        Create an image-position association using the model's associate_image_position.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(
            f"[ImagePositionAssociationService.associate] image_id={image_id}, position_id={position_id}",
            rid,
        )

        try:
            with log_timed_operation("ImagePositionAssociationService.associate", rid):
                assoc = ImagePositionAssociation.associate_image_position(
                    image_id=image_id,
                    position_id=position_id,
                    request_id=rid,
                    session=sess,
                )
                return assoc
        except Exception as e:  # model already logs SQLAlchemyError; keep broad here
            error_id(f"Error in ImagePositionAssociationService.associate: {e}", rid, exc_info=True)
            return None
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def dissociate(
        self,
        image_id: int,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> bool:
        """
        Remove an image-position association using the model's dissociate method.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(
            f"[ImagePositionAssociationService.dissociate] image_id={image_id}, position_id={position_id}",
            rid,
        )

        try:
            with log_timed_operation("ImagePositionAssociationService.dissociate", rid):
                ok = ImagePositionAssociation.dissociate_image_position(
                    image_id=image_id,
                    position_id=position_id,
                    request_id=rid,
                    session=sess,
                )
                return ok
        except Exception as e:
            error_id(f"Error in ImagePositionAssociationService.dissociate: {e}", rid, exc_info=True)
            return False
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # Queries
    # ---------------------------------------------------------
    @with_request_id
    def get_positions_by_image(
        self,
        image_id: Optional[int] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        position_id: Optional[int] = None,
        area_id: Optional[int] = None,
        equipment_group_id: Optional[int] = None,
        model_id: Optional[int] = None,
        asset_number_id: Optional[int] = None,
        location_id: Optional[int] = None,
        subassembly_id: Optional[int] = None,
        component_assembly_id: Optional[int] = None,
        assembly_view_id: Optional[int] = None,
        site_location_id: Optional[int] = None,
        exact_match: bool = False,
        limit: int = 100,
        session: Optional[SASession] = None,
    ) -> List[Position]:
        """
        Thin wrapper around the ImagePositionAssociation.get_positions_by_image method.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id("[ImagePositionAssociationService.get_positions_by_image] called", rid)

        try:
            with log_timed_operation(
                "ImagePositionAssociationService.get_positions_by_image", rid
            ):
                return ImagePositionAssociation.get_positions_by_image(
                    image_id=image_id,
                    title=title,
                    description=description,
                    file_path=file_path,
                    position_id=position_id,
                    area_id=area_id,
                    equipment_group_id=equipment_group_id,
                    model_id=model_id,
                    asset_number_id=asset_number_id,
                    location_id=location_id,
                    subassembly_id=subassembly_id,
                    component_assembly_id=component_assembly_id,
                    assembly_view_id=assembly_view_id,
                    site_location_id=site_location_id,
                    exact_match=exact_match,
                    limit=limit,
                    request_id=rid,
                    session=sess,
                )
        except Exception as e:
            error_id(f"Error in get_positions_by_image: {e}", rid, exc_info=True)
            return []
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_images_by_position(
        self,
        position_id: Optional[int] = None,
        area_id: Optional[int] = None,
        equipment_group_id: Optional[int] = None,
        model_id: Optional[int] = None,
        asset_number_id: Optional[int] = None,
        location_id: Optional[int] = None,
        subassembly_id: Optional[int] = None,
        component_assembly_id: Optional[int] = None,
        assembly_view_id: Optional[int] = None,
        site_location_id: Optional[int] = None,
        image_id: Optional[int] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        exact_match: bool = False,
        limit: int = 100,
        session: Optional[SASession] = None,
    ) -> List[Image]:
        """
        Wrapper around ImagePositionAssociation.get_images_by_position.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id("[ImagePositionAssociationService.get_images_by_position] called", rid)

        try:
            with log_timed_operation(
                "ImagePositionAssociationService.get_images_by_position", rid
            ):
                return ImagePositionAssociation.get_images_by_position(
                    position_id=position_id,
                    area_id=area_id,
                    equipment_group_id=equipment_group_id,
                    model_id=model_id,
                    asset_number_id=asset_number_id,
                    location_id=location_id,
                    subassembly_id=subassembly_id,
                    component_assembly_id=component_assembly_id,
                    assembly_view_id=assembly_view_id,
                    site_location_id=site_location_id,
                    image_id=image_id,
                    title=title,
                    description=description,
                    file_path=file_path,
                    exact_match=exact_match,
                    limit=limit,
                    request_id=rid,
                    session=sess,
                )
        except Exception as e:
            error_id(f"Error in get_images_by_position: {e}", rid, exc_info=True)
            return []
        finally:
            if created_here:
                sess.close()
