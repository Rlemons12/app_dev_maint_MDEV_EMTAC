from __future__ import annotations

from typing import List

from modules.configuration.log_config import logger, with_request_id
from modules.emtacdb.emtacdb_fts import Position


class BillOfMaterialsPositionService:
    """
    Domain service for Position lookups used by BOM edit workflows.
    """

    @with_request_id
    def get_all_positions(self, *, session) -> List[Position]:
        positions = session.query(Position).all()
        logger.debug("Retrieved %s positions for dropdown", len(positions))
        return positions