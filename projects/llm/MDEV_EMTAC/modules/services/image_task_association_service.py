# modules/services/image_task_association_service.py

from typing import Optional, List
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    info_id,
    warning_id,
    debug_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    ImageTaskAssociation,
    Image,
    Task,
)


class ImageTaskAssociationService:
    """
    Pure domain service for Image <-> Task associations.

    HARD RULES:
    - NEVER open sessions
    - NEVER close sessions
    - NEVER commit
    - NEVER rollback
    - Orchestrator owns transactions
    """

    # ---------------------------------------------------------
    # ASSOCIATE
    # ---------------------------------------------------------

    @with_request_id
    def associate(
        self,
        session: Session,
        *,
        image_id: int,
        task_id: int,
        request_id: Optional[str] = None,
    ) -> Optional[ImageTaskAssociation]:

        # Validate existence
        image = session.get(Image, image_id)
        task = session.get(Task, task_id)

        if not image:
            warning_id(f"Image id={image_id} not found", request_id)
            return None

        if not task:
            warning_id(f"Task id={task_id} not found", request_id)
            return None

        # Prevent duplicates
        existing = (
            session.query(ImageTaskAssociation)
            .filter_by(image_id=image_id, task_id=task_id)
            .first()
        )

        if existing:
            debug_id(
                f"Association already exists image={image_id}, task={task_id}",
                request_id,
            )
            return existing

        assoc = ImageTaskAssociation(
            image_id=image_id,
            task_id=task_id,
        )

        session.add(assoc)
        session.flush()

        debug_id(
            f"Association staged image={image_id}, task={task_id}",
            request_id,
        )

        return assoc

    # ---------------------------------------------------------
    # DISSOCIATE
    # ---------------------------------------------------------

    @with_request_id
    def dissociate(
        self,
        session: Session,
        *,
        image_id: int,
        task_id: int,
        request_id: Optional[str] = None,
    ) -> bool:

        assoc = (
            session.query(ImageTaskAssociation)
            .filter_by(image_id=image_id, task_id=task_id)
            .first()
        )

        if not assoc:
            warning_id(
                f"No association found image={image_id}, task={task_id}",
                request_id,
            )
            return False

        session.delete(assoc)

        info_id(
            f"Association staged for deletion image={image_id}, task={task_id}",
            request_id,
        )

        return True

    # ---------------------------------------------------------
    # LOOKUPS
    # ---------------------------------------------------------

    @with_request_id
    def get_tasks_for_image(
        self,
        session: Session,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> List[Task]:

        return (
            session.query(Task)
            .join(ImageTaskAssociation,
                  ImageTaskAssociation.task_id == Task.id)
            .filter(ImageTaskAssociation.image_id == image_id)
            .all()
        )

    @with_request_id
    def get_images_for_task(
        self,
        session: Session,
        *,
        task_id: int,
        request_id: Optional[str] = None,
    ) -> List[Image]:

        return (
            session.query(Image)
            .join(ImageTaskAssociation,
                  ImageTaskAssociation.image_id == Image.id)
            .filter(ImageTaskAssociation.task_id == task_id)
            .all()
        )

    # ---------------------------------------------------------
    # BULK REMOVE
    # ---------------------------------------------------------

    @with_request_id
    def remove_all_for_image(
        self,
        session: Session,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> int:

        assocs = (
            session.query(ImageTaskAssociation)
            .filter_by(image_id=image_id)
            .all()
        )

        count = len(assocs)

        for assoc in assocs:
            session.delete(assoc)

        session.flush()

        return count
