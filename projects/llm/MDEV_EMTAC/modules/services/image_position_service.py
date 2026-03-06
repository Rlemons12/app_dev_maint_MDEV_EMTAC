from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.log_config import (
    info_id,
    error_id,
    warning_id,
    debug_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    Image,
    Position,
    ImagePositionAssociation,
)


class ImagePositionService:
    """
    Pure domain service for Image <-> Position associations.

    HARD RULES:
    - NEVER open sessions
    - NEVER close sessions
    - NEVER commit
    - NEVER rollback
    - Orchestrator owns transactions
    """

    # ------------------------------------------------------------------
    # ASSOCIATION CREATION
    # ------------------------------------------------------------------

    @with_request_id
    def associate(
        self,
        session: Session,
        *,
        image_id: int,
        position_id: int,
        request_id: Optional[str] = None,
    ) -> Optional[ImagePositionAssociation]:

        if not session:
            raise RuntimeError("Session required for associate")

        # Prevent duplicates
        existing = (
            session.query(ImagePositionAssociation)
            .filter_by(image_id=image_id, position_id=position_id)
            .first()
        )

        if existing:
            debug_id(
                f"Association already exists image={image_id}, position={position_id}",
                request_id,
            )
            return existing

        assoc = ImagePositionAssociation(
            image_id=image_id,
            position_id=position_id,
        )

        session.add(assoc)
        session.flush()

        debug_id(
            f"Association staged image={image_id}, position={position_id}",
            request_id,
        )

        return assoc

    # ------------------------------------------------------------------
    # ASSOCIATION REMOVAL
    # ------------------------------------------------------------------

    @with_request_id
    def dissociate(
        self,
        session: Session,
        *,
        image_id: int,
        position_id: int,
        request_id: Optional[str] = None,
    ) -> bool:

        assoc = (
            session.query(ImagePositionAssociation)
            .filter_by(image_id=image_id, position_id=position_id)
            .first()
        )

        if not assoc:
            warning_id(
                f"No association found image={image_id}, position={position_id}",
                request_id,
            )
            return False

        session.delete(assoc)

        info_id(
            f"Association staged for deletion image={image_id}, position={position_id}",
            request_id,
        )

        return True

    # ------------------------------------------------------------------
    # LOOKUP: positions for an image
    # ------------------------------------------------------------------

    @with_request_id
    def get_positions(
        self,
        session: Session,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> List[Position]:

        return (
            session.query(Position)
            .join(ImagePositionAssociation,
                  ImagePositionAssociation.position_id == Position.id)
            .filter(ImagePositionAssociation.image_id == image_id)
            .all()
        )

    # ------------------------------------------------------------------
    # LOOKUP: images for a position
    # ------------------------------------------------------------------

    @with_request_id
    def get_images(
        self,
        session: Session,
        *,
        position_id: int,
        request_id: Optional[str] = None,
    ) -> List[Image]:

        return (
            session.query(Image)
            .join(ImagePositionAssociation,
                  ImagePositionAssociation.image_id == Image.id)
            .filter(ImagePositionAssociation.position_id == position_id)
            .all()
        )
