from __future__ import annotations
from typing import List, Optional, Sequence

from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    with_request_id, get_request_id, log_timed_operation, debug_id
)
from .base_repository import BaseRepository

# Robust imports
try:
    from emtacdb_fts import Drawing, DrawingPositionAssociation  # type: ignore
except Exception:
    try:
        from modules.emtacdb.emtacdb_fts import Drawing, DrawingPositionAssociation  # type: ignore
    except Exception:
        from emtacdb_fts import Drawing  # type: ignore
        DrawingPositionAssociation = None  # type: ignore


class DrawingRepository(BaseRepository):
    @with_request_id
    def search(
        self,
        *,
        drw_number: Optional[str] = None,
        drw_name: Optional[str] = None,
        drw_revision: Optional[str] = None,
        drw_spare_part_number: Optional[str] = None,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> List[Drawing]:
        """
        Simple parameterized search. Extend as needed (FTS, ILIKE, etc.).
        """
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("DrawingRepository.search", rid):
                q = sess.query(Drawing)
                if drw_number:
                    q = q.filter(Drawing.drw_number.ilike(f"%{drw_number}%"))
                if drw_name:
                    q = q.filter(Drawing.drw_name.ilike(f"%{drw_name}%"))
                if drw_revision:
                    q = q.filter(Drawing.drw_revision.ilike(f"%{drw_revision}%"))
                if drw_spare_part_number:
                    q = q.filter(Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))
                q = q.limit(limit)
                rows = q.all()
                debug_id(f"drawing search -> {len(rows)}", rid)
                return rows
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def drawings_by_position_ids(
        self,
        position_ids: Sequence[int],
        limit: int = 200,
        request_id: Optional[str] = None,
    ) -> List[Drawing]:
        """
        Return drawings linked to position IDs via DrawingPositionAssociation.
        """
        rid = request_id or get_request_id()
        if not position_ids:
            return []
        if DrawingPositionAssociation is None:
            raise RuntimeError(
                "DrawingPositionAssociation is not importable. "
                "Ensure the class exists and is importable by this repository."
            )

        sess: Session = self._session()
        try:
            with log_timed_operation("DrawingRepository.drawings_by_position_ids", rid):
                q = (
                    sess.query(Drawing)
                    .join(DrawingPositionAssociation, DrawingPositionAssociation.drawing_id == Drawing.id)
                    .filter(DrawingPositionAssociation.position_id.in_(list(position_ids)))
                    .distinct(Drawing.id)
                    .limit(limit)
                )
                rows = q.all()
                debug_id(f"drawings_by_position_ids -> {len(rows)}", rid)
                return rows
        finally:
            if self._owns_session():
                sess.close()
