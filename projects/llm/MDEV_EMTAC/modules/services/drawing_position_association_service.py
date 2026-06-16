# modules/services/drawing_position_association_service.py

from typing import Optional, List, Tuple
from sqlalchemy import func
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
    DrawingPositionAssociation,
    Position,
    Drawing,
)


class DrawingPositionAssociationService:
    """
    Service layer for DrawingPositionAssociation.

    Responsibilities:
    - Associate / dissociate drawings and positions
    - Query positions by drawing filters
    - Query drawings by position filters
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
        drawing_id: int,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> Optional[DrawingPositionAssociation]:
        """
        Create a drawing-position association.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(
            f"[DrawingPositionAssociationService.associate] drawing_id={drawing_id}, position_id={position_id}",
            rid,
        )

        try:
            with log_timed_operation("DrawingPositionAssociationService.associate", rid):
                assoc = DrawingPositionAssociation.associate_drawing_position(
                    drawing_id=drawing_id,
                    position_id=position_id,
                    request_id=rid,
                    session=sess,
                )
                return assoc
        except Exception as e:
            error_id(f"Error in DrawingPositionAssociationService.associate: {e}", rid, exc_info=True)
            return None
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def dissociate(
        self,
        drawing_id: int,
        position_id: int,
        session: Optional[SASession] = None,
    ) -> bool:
        """
        Remove a drawing-position association.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id(
            f"[DrawingPositionAssociationService.dissociate] drawing_id={drawing_id}, position_id={position_id}",
            rid,
        )

        try:
            with log_timed_operation("DrawingPositionAssociationService.dissociate", rid):
                ok = DrawingPositionAssociation.dissociate_drawing_position(
                    drawing_id=drawing_id,
                    position_id=position_id,
                    request_id=rid,
                    session=sess,
                )
                return ok
        except Exception as e:
            error_id(f"Error in DrawingPositionAssociationService.dissociate: {e}", rid, exc_info=True)
            return False
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # Queries
    # ---------------------------------------------------------
    @with_request_id
    def get_positions_by_drawing(
        self,
        drawing_id: Optional[int] = None,
        drw_equipment_name: Optional[str] = None,
        drw_number: Optional[str] = None,
        drw_name: Optional[str] = None,
        drw_revision: Optional[str] = None,
        drw_spare_part_number: Optional[str] = None,
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
        Wrapper around DrawingPositionAssociation.get_positions_by_drawing.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id("[DrawingPositionAssociationService.get_positions_by_drawing] called", rid)

        try:
            with log_timed_operation(
                "DrawingPositionAssociationService.get_positions_by_drawing", rid
            ):
                return DrawingPositionAssociation.get_positions_by_drawing(
                    drawing_id=drawing_id,
                    drw_equipment_name=drw_equipment_name,
                    drw_number=drw_number,
                    drw_name=drw_name,
                    drw_revision=drw_revision,
                    drw_spare_part_number=drw_spare_part_number,
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
            error_id(f"Error in get_positions_by_drawing: {e}", rid, exc_info=True)
            return []
        finally:
            if created_here:
                sess.close()

    @with_request_id
    def get_drawings_by_position(
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
        drawing_id: Optional[int] = None,
        drw_equipment_name: Optional[str] = None,
        drw_number: Optional[str] = None,
        drw_name: Optional[str] = None,
        drw_revision: Optional[str] = None,
        drw_spare_part_number: Optional[str] = None,
        file_path: Optional[str] = None,
        exact_match: bool = False,
        limit: int = 100,
        session: Optional[SASession] = None,
    ) -> List[Drawing]:
        """
        Wrapper around DrawingPositionAssociation.get_drawings_by_position.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)
        debug_id("[DrawingPositionAssociationService.get_drawings_by_position] called", rid)

        try:
            with log_timed_operation(
                "DrawingPositionAssociationService.get_drawings_by_position", rid
            ):
                return DrawingPositionAssociation.get_drawings_by_position(
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
                    drawing_id=drawing_id,
                    drw_equipment_name=drw_equipment_name,
                    drw_number=drw_number,
                    drw_name=drw_name,
                    drw_revision=drw_revision,
                    drw_spare_part_number=drw_spare_part_number,
                    file_path=file_path,
                    exact_match=exact_match,
                    limit=limit,
                    request_id=rid,
                    session=sess,
                )
        except Exception as e:
            error_id(f"Error in get_drawings_by_position: {e}", rid, exc_info=True)
            return []
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # Multi-position helper (USED BY CHUNK SEARCH)
    # ---------------------------------------------------------
    def get_drawings_for_positions(
            self,
            *,
            position_ids: list[int],
            session: SASession,
            request_id: Optional[str] = None,
    ) -> list[Drawing]:

        """
        Return drawings associated with ANY of the given position_ids.

        This is a convenience wrapper used by ChunkAssociationSearch
        to support 2nd-tier enrichment.
        """
        rid = get_request_id()
        sess, created_here = self._get_session(session)

        if not position_ids:
            debug_id(
                "[DrawingPositionAssociationService.get_drawings_for_positions] No position_ids provided",
                rid,
            )
            return []

        # De-duplicate & normalize
        position_ids = sorted({int(pid) for pid in position_ids if pid is not None})

        debug_id(
            f"[DrawingPositionAssociationService.get_drawings_for_positions] "
            f"Fetching drawings for {len(position_ids)} positions",
            rid,
        )

        try:
            with log_timed_operation(
                    "DrawingPositionAssociationService.get_drawings_for_positions", rid
            ):
                drawings_map = {}

                for position_id in position_ids:
                    drawings = self.get_drawings_by_position(
                        position_id=position_id,
                        session=sess,
                    )

                    for drw in drawings:
                        drawings_map[drw.id] = drw  # dedupe by PK

                results = list(drawings_map.values())

                debug_id(
                    f"[DrawingPositionAssociationService.get_drawings_for_positions] "
                    f"Found {len(results)} unique drawings",
                    rid,
                )

                return results

        except Exception as e:
            error_id(
                f"Error in get_drawings_for_positions: {e}",
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
    def count_drawings(
            self,
            position_ids: list[int],
            session: Optional[SASession] = None,
    ) -> int:
        """
        Count unique Drawings associated with given Positions.

        Lightweight helper for relationship summaries.
        """
        rid = get_request_id()

        if not position_ids:
            debug_id(
                "[DrawingPositionAssociationService.count_drawings] No position_ids provided",
                rid,
            )
            return 0

        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                    "DrawingPositionAssociationService.count_drawings", rid
            ):
                count = (
                    sess.query(func.count(func.distinct(DrawingPositionAssociation.drawing_id)))
                    .filter(
                        DrawingPositionAssociation.position_id.in_(position_ids)
                    )
                    .scalar()
                )

                debug_id(
                    f"Counted {count} unique drawings for {len(position_ids)} positions",
                    rid,
                )

                return count or 0

        except Exception as e:
            error_id(
                f"Error counting drawings for positions {position_ids}: {e}",
                rid,
                exc_info=True,
            )
            return 0
        finally:
            if created_here:
                sess.close()

    # ---------------------------------------------------------
    # LIGHTWEIGHT COUNTS (USED BY CHUNK GRAPH / PROBES)
    # ---------------------------------------------------------
    @with_request_id
    def count_drawings(
            self,
            position_ids: list[int],
            session: Optional[SASession] = None,
    ) -> int:
        """
        Count unique Drawings associated with given Positions.

        Lightweight helper for relationship summaries.
        """
        rid = get_request_id()

        if not position_ids:
            debug_id(
                "[DrawingPositionAssociationService.count_drawings] No position_ids provided",
                rid,
            )
            return 0

        sess, created_here = self._get_session(session)

        try:
            with log_timed_operation(
                    "DrawingPositionAssociationService.count_drawings", rid
            ):
                count = (
                    sess.query(func.count(func.distinct(DrawingPositionAssociation.drawing_id)))
                    .filter(
                        DrawingPositionAssociation.position_id.in_(position_ids)
                    )
                    .scalar()
                )

                debug_id(
                    f"Counted {count} unique drawings for {len(position_ids)} positions",
                    rid,
                )

                return count or 0

        except Exception as e:
            error_id(
                f"Error counting drawings for positions {position_ids}: {e}",
                rid,
                exc_info=True,
            )
            return 0
        finally:
            if created_here:
                sess.close()
