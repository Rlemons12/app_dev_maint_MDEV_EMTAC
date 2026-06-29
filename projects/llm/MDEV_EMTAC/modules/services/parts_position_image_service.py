# modules/services/parts_position_image_service.py

from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as SASession
from sqlalchemy.orm import Session
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
    PartsPositionImageAssociation,
    Position,
    Part,
    Image,
    BillOfMaterial,
)


class PartsPositionImageService:
    """
    Service layer for the PartsPositionImageAssociation model.

    Responsibilities:
    - Create / delete part-position-image links
    - Search associations by flexible filters
    - Traverse hierarchy to get corresponding Position IDs
    - Convenience methods to get Images for a part/position
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # ---------------------------------------------------------
    # Session helper
    # ---------------------------------------------------------
    def _get_session(self, session: Optional[SASession]) -> Tuple[SASession, bool]:
        """
        Returns a (session, created_here) tuple.
        """
        if session is not None:
            return session, False
        return self.db_config.get_main_session(), True

    # ---------------------------------------------------------
    # Basic CRUD-like methods
    # ---------------------------------------------------------
    @with_request_id
    def get_by_id(
        self,
        association_id: int,
        session: Optional[SASession] = None,
    ) -> Optional[PartsPositionImageAssociation]:
        """
        Get a single PartsPositionImageAssociation by its ID.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(f"[PartsPositionImageService.get_by_id] association_id={association_id}", rid)

        try:
            with log_timed_operation("PartsPositionImageService.get_by_id", rid):
                assoc = (
                    sess.query(PartsPositionImageAssociation)
                    .filter(PartsPositionImageAssociation.id == association_id)
                    .first()
                )
                if assoc:
                    debug_id(f"Found association id={assoc.id}", rid)
                else:
                    debug_id(f"No association found for id={association_id}", rid)
                return assoc
        except SQLAlchemyError as e:
            error_id(f"Error in get_by_id({association_id}): {e}", rid, exc_info=True)
            return None
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def create_association(
        self,
        part_id: int,
        position_id: int,
        image_id: int,
        session: Optional[SASession] = None,
    ) -> Optional[PartsPositionImageAssociation]:
        """
        Create a new PartsPositionImageAssociation.

        NOTE:
        - Does NOT create Part/Position/Image themselves, only links existing rows.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(
            f"[PartsPositionImageService.create_association] part_id={part_id}, "
            f"position_id={position_id}, image_id={image_id}",
            rid,
        )

        try:
            with log_timed_operation("PartsPositionImageService.create_association", rid):
                # Validate foreign keys
                part = sess.query(Part).filter(Part.id == part_id).first()
                position = sess.query(Position).filter(Position.id == position_id).first()
                image = sess.query(Image).filter(Image.id == image_id).first()

                if not part:
                    warning_id(f"Part id={part_id} not found", rid)
                    return None
                if not position:
                    warning_id(f"Position id={position_id} not found", rid)
                    return None
                if not image:
                    warning_id(f"Image id={image_id} not found", rid)
                    return None

                # Check for existing association
                existing = (
                    sess.query(PartsPositionImageAssociation)
                    .filter(
                        PartsPositionImageAssociation.part_id == part_id,
                        PartsPositionImageAssociation.position_id == position_id,
                        PartsPositionImageAssociation.image_id == image_id,
                    )
                    .first()
                )
                if existing:
                    debug_id("Association already exists, returning existing", rid)
                    return existing

                assoc = PartsPositionImageAssociation(
                    part_id=part_id,
                    position_id=position_id,
                    image_id=image_id,
                )
                sess.add(assoc)
                sess.commit()
                debug_id(f"Created association id={assoc.id}", rid)
                return assoc

        except SQLAlchemyError as e:
            error_id(f"Error in create_association: {e}", rid, exc_info=True)
            if created_here:
                sess.rollback()
            return None
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def delete_association(
        self,
        association_id: int,
        session: Optional[SASession] = None,
    ) -> bool:
        """
        Delete a PartsPositionImageAssociation by ID.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(f"[PartsPositionImageService.delete_association] id={association_id}", rid)

        try:
            with log_timed_operation("PartsPositionImageService.delete_association", rid):
                assoc = (
                    sess.query(PartsPositionImageAssociation)
                    .filter(PartsPositionImageAssociation.id == association_id)
                    .first()
                )
                if not assoc:
                    warning_id(f"Association id={association_id} not found", rid)
                    return False

                sess.delete(assoc)
                sess.commit()
                debug_id(f"Deleted association id={association_id}", rid)
                return True

        except SQLAlchemyError as e:
            error_id(f"Error deleting association {association_id}: {e}", rid, exc_info=True)
            if created_here:
                sess.rollback()
            return False
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # Search / query helpers
    # ---------------------------------------------------------
    def search(
            self,
            *,
            session: Session,
            image_ids: list[int],
            request_id: str,
    ):
        if not image_ids:
            return []

        # 🔑 Normalize & dedupe
        image_ids = sorted({int(i) for i in image_ids})

        debug_id(
            f"[PartsPositionImageService] searching image_ids={image_ids}",
            request_id,
        )

        return (
            session.query(PartsPositionImageAssociation)
            .filter(PartsPositionImageAssociation.image_id.in_(image_ids))
            .all()
        )

    @with_request_id
    def get_corresponding_position_ids(
        self,
        area_id: Optional[int] = None,
        equipment_group_id: Optional[int] = None,
        model_id: Optional[int] = None,
        asset_number_id: Optional[int] = None,
        location_id: Optional[int] = None,
        session: Optional[SASession] = None,
    ) -> List[int]:
        """
        Wrapper around the model's get_corresponding_position_ids for hierarchy traversal.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(
            "[PartsPositionImageService.get_corresponding_position_ids] "
            f"area_id={area_id}, eg_id={equipment_group_id}, model_id={model_id}, "
            f"asset_id={asset_number_id}, location_id={location_id}",
            rid,
        )

        try:
            with log_timed_operation(
                "PartsPositionImageService.get_corresponding_position_ids", rid
            ):
                ids = PartsPositionImageAssociation.get_corresponding_position_ids(
                    session=sess,
                    area_id=area_id,
                    equipment_group_id=equipment_group_id,
                    model_id=model_id,
                    asset_number_id=asset_number_id,
                    location_id=location_id,
                )
                debug_id(f"Got {len(ids)} corresponding position IDs", rid)
                return ids
        except SQLAlchemyError as e:
            error_id(f"Error in get_corresponding_position_ids: {e}", rid, exc_info=True)
            return []
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # Convenience: get Images / Positions from associations
    # ---------------------------------------------------------
    @with_request_id
    def get_images_for_position(
        self,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> List[Image]:
        """
        Get all Image rows linked to a given Position via PartsPositionImageAssociation.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(f"[PartsPositionImageService.get_images_for_position] position_id={position_id}", rid)

        try:
            with log_timed_operation("PartsPositionImageService.get_images_for_position", rid):
                q = (
                    sess.query(Image)
                    .join(PartsPositionImageAssociation, Image.id == PartsPositionImageAssociation.image_id)
                    .filter(PartsPositionImageAssociation.position_id == position_id)
                    .distinct()
                )
                images = q.all()
                debug_id(f"Found {len(images)} images for position_id={position_id}", rid)
                return images

        except SQLAlchemyError as e:
            error_id(f"Error in get_images_for_position: {e}", rid, exc_info=True)
            return []
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_images_for_part(
            self,
            part_id: int,
            session: Optional[SASession] = None,
    ) -> List[Image]:

        rid = get_request_id()
        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation("PartsPositionImageService.get_images_for_part", rid):

                # STEP 1: get distinct image IDs (safe, integer only)
                subq = (
                    sess.query(PartsPositionImageAssociation.image_id)
                    .filter(PartsPositionImageAssociation.part_id == part_id)
                    .distinct()
                    .subquery()
                )

                # STEP 2: load full Image rows
                images = (
                    sess.query(Image)
                    .join(subq, Image.id == subq.c.image_id)
                    .all()
                )

                debug_id(
                    f"Found {len(images)} images for part_id={part_id}",
                    rid,
                )

                return images

        except SQLAlchemyError as e:
            error_id(
                f"Error in get_images_for_part: {e}",
                rid,
                exc_info=True,
            )
            return []

        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # LIGHTWEIGHT COUNTS (USED BY CHUNK GRAPH / PROBES)
    # ---------------------------------------------------------
    @with_request_id
    def count_parts(
            self,
            position_ids: list[int],
            session: Optional[SASession] = None,
    ) -> int:
        """
        Count unique Parts associated with given Positions.

        Lightweight helper for relationship summaries.
        """
        rid = get_request_id()

        if not position_ids:
            debug_id(
                "[PartsPositionImageService.count_parts] No position_ids provided",
                rid,
            )
            return 0

        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                    "PartsPositionImageService.count_parts", rid
            ):
                count = (
                    sess.query(func.count(func.distinct(PartsPositionImageAssociation.part_id)))
                    .filter(
                        PartsPositionImageAssociation.position_id.in_(position_ids)
                    )
                    .scalar()
                )

                debug_id(
                    f"Counted {count} unique parts for {len(position_ids)} positions",
                    rid,
                )

                return count or 0

        except Exception as e:
            error_id(
                f"Error counting parts for positions {position_ids}: {e}",
                rid,
                exc_info=True,
            )
            return 0
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_images_for_positions(
            self,
            position_ids: List[int],
            session: Optional[SASession] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all Images linked to multiple Positions via PartsPositionImageAssociation.

        Returns UI-safe dictionaries (NOT ORM objects).
        """
        rid = get_request_id()

        if not position_ids:
            debug_id(
                "[PartsPositionImageService.get_images_for_positions] No position_ids provided",
                rid,
            )
            return []

        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                    "PartsPositionImageService.get_images_for_positions", rid
            ):
                q = (
                    sess.query(
                        Image.id,
                        Image.file_path,
                        Image.description,
                    )
                    .join(
                        PartsPositionImageAssociation,
                        Image.id == PartsPositionImageAssociation.image_id,
                    )
                    .filter(
                        PartsPositionImageAssociation.position_id.in_(position_ids)
                    )
                    .distinct()
                )

                rows = q.all()

                debug_id(
                    f"Resolved {len(rows)} images for {len(position_ids)} positions",
                    rid,
                )

                return [
                    {
                        "id": r.id,
                        "file_path": r.file_path,
                        "description": r.description,
                        "source": "position_part",
                    }
                    for r in rows
                ]

        except SQLAlchemyError as e:
            error_id(
                f"Error in get_images_for_positions: {e}",
                rid,
                exc_info=True,
            )
            return []
        finally:
            if created_here:
                sess.close()

    def search_by_positions(
            self,
            *,
            session: Session,
            position_ids: List[int],
    ) -> List[PartsPositionImageAssociation]:
        """
        Return Part–Position–Image associations for multiple positions.

        Used by:
        - DocumentUIPayload.enrich_with_parts
        - UI document enrichment
        """

        if not position_ids:
            return []

        return (
            session.query(PartsPositionImageAssociation)
            .filter(
                PartsPositionImageAssociation.position_id.in_(position_ids)
            )
            .all()
        )
