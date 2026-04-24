from __future__ import annotations

from typing import Optional

from modules.configuration.log_config import logger, with_request_id
from modules.emtacdb.emtacdb_fts import PartsPositionImageAssociation


class BillOfMaterialsPartImageAssociationService:
    """
    Domain service for Part <-> Position <-> Image associations.
    """

    @with_request_id
    def create_association(
        self,
        *,
        session,
        part_id: int,
        position_id: int | None,
        image_id: int,
    ) -> PartsPositionImageAssociation:
        association = PartsPositionImageAssociation(
            part_id=part_id,
            position_id=position_id,
            image_id=image_id,
        )
        session.add(association)

        logger.info(
            "Created PartsPositionImageAssociation part_id=%s position_id=%s image_id=%s",
            part_id,
            position_id,
            image_id,
        )
        return association

    @with_request_id
    def get_association(
        self,
        *,
        session,
        part_id: int,
        image_id: int,
    ) -> Optional[PartsPositionImageAssociation]:
        association = (
            session.query(PartsPositionImageAssociation)
            .filter_by(part_id=part_id, image_id=image_id)
            .first()
        )
        logger.debug(
            "Fetched association for part_id=%s image_id=%s found=%s",
            part_id,
            image_id,
            bool(association),
        )
        return association

    @with_request_id
    def delete_association(self, *, session, association: PartsPositionImageAssociation) -> None:
        logger.info(
            "Deleting association id=%s part_id=%s image_id=%s",
            association.id,
            association.part_id,
            association.image_id,
        )
        session.delete(association)

    @with_request_id
    def count_image_associations(self, *, session, image_id: int) -> int:
        count = (
            session.query(PartsPositionImageAssociation)
            .filter_by(image_id=image_id)
            .count()
        )
        logger.debug("Association count for image_id=%s is %s", image_id, count)
        return count