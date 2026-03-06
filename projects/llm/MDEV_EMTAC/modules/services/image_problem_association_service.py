# modules/services/image_problem_association_service.py

from typing import Optional, List
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    info_id,
    warning_id,
    debug_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    ImageProblemAssociation,
    Image,
    Problem,
)


class ImageProblemAssociationService:
    """
    Pure domain service for Image <-> Problem associations.

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
        problem_id: int,
        request_id: Optional[str] = None,
    ) -> Optional[ImageProblemAssociation]:

        image = session.get(Image, image_id)
        problem = session.get(Problem, problem_id)

        if not image:
            warning_id(f"Image id={image_id} not found", request_id)
            return None

        if not problem:
            warning_id(f"Problem id={problem_id} not found", request_id)
            return None

        existing = (
            session.query(ImageProblemAssociation)
            .filter_by(image_id=image_id, problem_id=problem_id)
            .first()
        )

        if existing:
            debug_id(
                f"ImageProblemAssociation already exists image={image_id}, problem={problem_id}",
                request_id,
            )
            return existing

        assoc = ImageProblemAssociation(
            image_id=image_id,
            problem_id=problem_id,
        )

        session.add(assoc)
        session.flush()

        debug_id(
            f"ImageProblemAssociation staged image={image_id}, problem={problem_id}",
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
        problem_id: int,
        request_id: Optional[str] = None,
    ) -> bool:

        assoc = (
            session.query(ImageProblemAssociation)
            .filter_by(image_id=image_id, problem_id=problem_id)
            .first()
        )

        if not assoc:
            warning_id(
                f"No ImageProblemAssociation found image={image_id}, problem={problem_id}",
                request_id,
            )
            return False

        session.delete(assoc)

        info_id(
            f"ImageProblemAssociation staged for deletion image={image_id}, problem={problem_id}",
            request_id,
        )

        return True

    # ---------------------------------------------------------
    # LOOKUPS
    # ---------------------------------------------------------

    @with_request_id
    def get_problems_for_image(
        self,
        session: Session,
        *,
        image_id: int,
        request_id: Optional[str] = None,
    ) -> List[Problem]:

        return (
            session.query(Problem)
            .join(ImageProblemAssociation,
                  ImageProblemAssociation.problem_id == Problem.id)
            .filter(ImageProblemAssociation.image_id == image_id)
            .all()
        )

    @with_request_id
    def get_images_for_problem(
        self,
        session: Session,
        *,
        problem_id: int,
        request_id: Optional[str] = None,
    ) -> List[Image]:

        return (
            session.query(Image)
            .join(ImageProblemAssociation,
                  ImageProblemAssociation.image_id == Image.id)
            .filter(ImageProblemAssociation.problem_id == problem_id)
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
            session.query(ImageProblemAssociation)
            .filter_by(image_id=image_id)
            .all()
        )

        count = len(assocs)

        for assoc in assocs:
            session.delete(assoc)

        session.flush()

        info_id(
            f"Removed {count} ImageProblemAssociation rows for image_id={image_id}",
            request_id,
        )

        return count
