from __future__ import annotations
from typing import List, Optional, Tuple, Sequence

from sqlalchemy import distinct
from sqlalchemy.orm import Session

from modules.configuration.log_config import (
    with_request_id, get_request_id, log_timed_operation, debug_id
)
from .base_repository import BaseRepository

# ORM imports (robust)
try:
    from emtacdb_fts import Part, PartsPositionImageAssociation  # type: ignore
except Exception:
    try:
        from modules.emtacdb.emtacdb_fts import Part, PartsPositionImageAssociation  # type: ignore
    except Exception:
        # Fallback: Part available, association may be absent
        from emtacdb_fts import Part  # type: ignore
        PartsPositionImageAssociation = None  # type: ignore


class PartRepository(BaseRepository):
    """
    Thin repository over Part model. Wraps:
      - Part.search(...)
      - Part.fts_search(...)
      - Part.get_by_id(...)
      - parts_by_position_ids(...) (via PartsPositionImageAssociation)
    """

    @with_request_id
    def search(
        self,
        *,
        search_text: Optional[str] = None,
        fields: Optional[List[str]] = None,
        exact_match: bool = False,
        use_fts: bool = True,
        part_id: Optional[int] = None,
        part_number: Optional[str] = None,
        name: Optional[str] = None,
        oem_mfg: Optional[str] = None,
        model: Optional[str] = None,
        class_flag: Optional[str] = None,
        ud6: Optional[str] = None,
        type_: Optional[str] = None,
        notes: Optional[str] = None,
        documentation: Optional[str] = None,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> List[Part]:
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("PartRepository.search", rid):
                return Part.search(
                    search_text=search_text,
                    fields=fields,
                    exact_match=exact_match,
                    use_fts=use_fts,
                    part_id=part_id,
                    part_number=part_number,
                    name=name,
                    oem_mfg=oem_mfg,
                    model=model,
                    class_flag=class_flag,
                    ud6=ud6,
                    type_=type_,
                    notes=notes,
                    documentation=documentation,
                    limit=limit,
                    request_id=rid,
                    session=sess,
                )
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def fts_search(
        self,
        search_text: str,
        limit: int = 100,
        request_id: Optional[str] = None,
    ) -> List[Tuple[Part, float]]:
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("PartRepository.fts_search", rid):
                return Part.fts_search(search_text=search_text, limit=limit, request_id=rid, session=sess)
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def get_by_id(self, part_id: int, request_id: Optional[str] = None) -> Optional[Part]:
        rid = request_id or get_request_id()
        sess: Session = self._session()
        try:
            with log_timed_operation("PartRepository.get_by_id", rid):
                return Part.get_by_id(part_id=part_id, request_id=rid, session=sess)
        finally:
            if self._owns_session():
                sess.close()

    @with_request_id
    def parts_by_position_ids(
        self,
        position_ids: Sequence[int],
        limit: int = 200,
        request_id: Optional[str] = None,
    ) -> List[Part]:
        """
        Return distinct parts linked to any of the given position IDs via PartsPositionImageAssociation.
        Requires the association model. If missing, raises a descriptive RuntimeError.
        """
        rid = request_id or get_request_id()
        if not position_ids:
            return []
        if PartsPositionImageAssociation is None:
            raise RuntimeError(
                "PartsPositionImageAssociation is not importable. "
                "Ensure the class exists and is importable by this repository."
            )

        sess: Session = self._session()
        try:
            with log_timed_operation("PartRepository.parts_by_position_ids", rid):
                q = (
                    sess.query(Part)
                    .join(
                        PartsPositionImageAssociation,
                        PartsPositionImageAssociation.part_id == Part.id,
                    )
                    .filter(PartsPositionImageAssociation.position_id.in_(list(position_ids)))
                    .distinct(Part.id)
                    .limit(limit)
                )
                rows = q.all()
                debug_id(f"parts_by_position_ids -> {len(rows)} parts for {len(position_ids)} positions", rid)
                return rows
        finally:
            if self._owns_session():
                sess.close()
