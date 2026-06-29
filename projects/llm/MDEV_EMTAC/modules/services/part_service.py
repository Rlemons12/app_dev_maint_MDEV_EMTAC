# services/part_service.py
from __future__ import annotations

from typing import Optional, List, Dict, Any

from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from modules.emtacdb.emtacdb_fts import (
    Part,
    PartsPositionImageAssociation,
)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class PartService:
    """
    Service layer for managing Part entities.

    Notes:
    - Legacy wrapper methods remain for compatibility.
    - New session-aware methods are provided so orchestrators can own
      transaction boundaries cleanly.

    Performance notes:
    - get_parts_for_positions() is kept as a legacy-compatible method that
      returns all matching distinct parts.
    - search_distinct_parts_for_positions() is the optimized BOM-search method
      that pushes DISTINCT, COUNT, LIMIT, and OFFSET into SQL.
    """

    DEFAULT_INDEX = 0
    DEFAULT_PER_PAGE = 25
    MAX_PER_PAGE = 100

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ============================================================
    # SESSION-AWARE METHODS FOR ORCHESTRATORS
    # ============================================================

    def get_part_by_id(self, *, session: Session, part_id: int) -> Optional[Part]:
        return session.query(Part).filter_by(id=part_id).first()

    def get_part_by_number(self, *, session: Session, part_number: str) -> Optional[Part]:
        return (
            session.query(Part)
            .filter(Part.part_number == part_number)
            .first()
        )

    def get_existing_part_numbers(self, *, session: Session) -> set[str]:
        rows = (
            session.query(Part.part_number)
            .filter(Part.part_number.isnot(None))
            .all()
        )
        return {row[0] for row in rows if row and row[0]}

    def create_part(
        self,
        *,
        session: Session,
        part_number: str,
        name: Optional[str] = None,
        oem_mfg: Optional[str] = None,
        model: Optional[str] = None,
        class_flag: Optional[str] = None,
        ud6: Optional[str] = None,
        type_: Optional[str] = None,
        notes: Optional[str] = None,
        documentation: Optional[str] = None,
    ) -> Part:
        part = Part(
            part_number=part_number,
            name=name,
            oem_mfg=oem_mfg,
            model=model,
            class_flag=class_flag,
            ud6=ud6,
            type=type_,
            notes=notes,
            documentation=documentation,
        )
        session.add(part)
        return part

    def create_parts_bulk(
        self,
        *,
        session: Session,
        part_rows: List[Dict[str, Any]],
    ) -> None:
        """
        Bulk insert parts without committing.
        Orchestrator owns commit.
        """
        if not part_rows:
            return

        mappings = []
        for row in part_rows:
            mappings.append({
                "part_number": row.get("part_number"),
                "name": row.get("name"),
                "oem_mfg": row.get("oem_mfg"),
                "model": row.get("model"),
                "class_flag": row.get("class_flag"),
                "ud6": row.get("ud6"),
                "type": row.get("type"),
                "notes": row.get("notes"),
                "documentation": row.get("documentation"),
            })

        session.bulk_insert_mappings(Part, mappings)

    def fetch_parts_by_numbers(
        self,
        *,
        session: Session,
        part_numbers: List[str],
    ) -> List[Part]:
        if not part_numbers:
            return []

        normalized_part_numbers = sorted({
            str(part_number).strip()
            for part_number in part_numbers
            if part_number is not None and str(part_number).strip()
        })

        if not normalized_part_numbers:
            return []

        return (
            session.query(Part)
            .filter(Part.part_number.in_(normalized_part_numbers))
            .all()
        )

    def update_part_fields(
        self,
        *,
        part: Part,
        part_fields: Dict[str, Any],
    ) -> Part:
        for field_name, value in part_fields.items():
            if field_name == "type_":
                setattr(part, "type", value)
            elif hasattr(part, field_name):
                setattr(part, field_name, value)
        return part

    def search_parts(
        self,
        *,
        session: Session,
        search_text: str,
        limit: int = 100,
        use_fts: bool = False,
    ) -> List[Part]:
        query = (search_text or "").strip()
        if not query:
            return []

        limit = self._normalize_limit(limit)

        if use_fts and hasattr(Part, "fts_search"):
            try:
                results = Part.fts_search(search_text=query, limit=limit)
                if results and isinstance(results[0], tuple):
                    return [row[0] for row in results]
                return results
            except Exception:
                pass

        like_query = f"%{query}%"
        return (
            session.query(Part)
            .filter(
                (Part.part_number.ilike(like_query)) |
                (Part.name.ilike(like_query)) |
                (Part.oem_mfg.ilike(like_query)) |
                (Part.model.ilike(like_query))
            )
            .order_by(Part.part_number.asc().nullslast(), Part.id.asc())
            .limit(limit)
            .all()
        )

    def delete_part(
        self,
        *,
        session: Session,
        part: Part,
    ) -> None:
        session.delete(part)

    def get_parts_for_positions(
        self,
        position_ids: List[int],
        session: Session,
    ) -> List[Part]:
        """
        Return all distinct Parts associated with the given Position IDs.

        Compatibility method.

        IMPORTANT:
        - This method returns all matching parts.
        - For paged BOM search, prefer search_distinct_parts_for_positions().
        """
        normalized_position_ids = self._normalize_int_list(position_ids)

        if not normalized_position_ids:
            return []

        stmt = (
            select(Part)
            .join(
                PartsPositionImageAssociation,
                PartsPositionImageAssociation.part_id == Part.id,
            )
            .where(
                PartsPositionImageAssociation.position_id.in_(normalized_position_ids),
                PartsPositionImageAssociation.part_id.isnot(None),
            )
            .distinct()
            .order_by(Part.part_number.asc().nullslast(), Part.id.asc())
        )

        return session.execute(stmt).scalars().all()

    def search_distinct_parts_for_positions(
        self,
        *,
        position_ids: List[int],
        session: Session,
        index: int = DEFAULT_INDEX,
        per_page: int = DEFAULT_PER_PAGE,
    ) -> Dict[str, Any]:
        """
        Return a paged list of distinct Parts associated with the given Position IDs.

        This is the optimized service method for BOM search.

        Why this is faster:
        - Counts distinct part IDs in SQL.
        - Applies LIMIT/OFFSET in SQL.
        - Loads only the Part rows needed for the current page.
        - Avoids loading all matching parts into Python just to paginate.

        Returns:
            {
                "parts": List[Part],
                "total_count": int,
                "index": int,
                "per_page": int,
            }
        """
        normalized_position_ids = self._normalize_int_list(position_ids)
        index = self._normalize_index(index)
        per_page = self._normalize_limit(per_page)

        if not normalized_position_ids:
            return {
                "parts": [],
                "total_count": 0,
                "index": index,
                "per_page": per_page,
            }

        total_count = (
            session.query(
                func.count(
                    func.distinct(PartsPositionImageAssociation.part_id)
                )
            )
            .filter(
                PartsPositionImageAssociation.position_id.in_(normalized_position_ids),
                PartsPositionImageAssociation.part_id.isnot(None),
            )
            .scalar()
        ) or 0

        if total_count == 0:
            return {
                "parts": [],
                "total_count": 0,
                "index": index,
                "per_page": per_page,
            }

        paged_part_ids_subquery = (
            session.query(
                Part.id.label("part_id")
            )
            .join(
                PartsPositionImageAssociation,
                PartsPositionImageAssociation.part_id == Part.id,
            )
            .filter(
                PartsPositionImageAssociation.position_id.in_(normalized_position_ids),
                PartsPositionImageAssociation.part_id.isnot(None),
            )
            .group_by(Part.id, Part.part_number)
            .order_by(Part.part_number.asc().nullslast(), Part.id.asc())
            .offset(index)
            .limit(per_page)
            .subquery()
        )

        parts = (
            session.query(Part)
            .join(
                paged_part_ids_subquery,
                Part.id == paged_part_ids_subquery.c.part_id,
            )
            .order_by(Part.part_number.asc().nullslast(), Part.id.asc())
            .all()
        )

        return {
            "parts": parts,
            "total_count": total_count,
            "index": index,
            "per_page": per_page,
        }

    # ============================================================
    # INTERNAL NORMALIZATION HELPERS
    # ============================================================

    @classmethod
    def _normalize_index(cls, value: Any) -> int:
        try:
            index = int(value)
        except (TypeError, ValueError):
            return cls.DEFAULT_INDEX

        return max(index, 0)

    @classmethod
    def _normalize_limit(cls, value: Any) -> int:
        try:
            limit = int(value)
        except (TypeError, ValueError):
            return cls.DEFAULT_PER_PAGE

        limit = max(limit, 1)
        limit = min(limit, cls.MAX_PER_PAGE)

        return limit

    @staticmethod
    def _normalize_int_list(values: Optional[List[int]]) -> List[int]:
        if not values:
            return []

        normalized = set()

        for value in values:
            if value in (None, "", "None"):
                continue

            try:
                normalized.add(int(value))
            except (TypeError, ValueError):
                continue

        return sorted(normalized)

    # ============================================================
    # LEGACY WRAPPERS
    # ============================================================

    @with_request_id
    def add_to_db(
        self,
        part_number: str,
        name: Optional[str] = None,
        oem_mfg: Optional[str] = None,
        model: Optional[str] = None,
        class_flag: Optional[str] = None,
        ud6: Optional[str] = None,
        type_: Optional[str] = None,
        notes: Optional[str] = None,
        documentation: Optional[str] = None,
    ) -> int:
        with self.db_config.main_session() as session:
            try:
                return Part.add_part_to_db(
                    session,
                    part_number=part_number,
                    name=name,
                    oem_mfg=oem_mfg,
                    model=model,
                    class_flag=class_flag,
                    ud6=ud6,
                    type_=type_,
                    notes=notes,
                    documentation=documentation,
                )
            except SQLAlchemyError as e:
                error_id(f"PartService.add_to_db failed: {e}", None)
                raise

    @with_request_id
    def find(self, **filters) -> List[Part]:
        try:
            return Part.search(**filters, session=None)
        except SQLAlchemyError as e:
            error_id(f"PartService.find failed: {e}", None)
            raise

    @with_request_id
    def get(self, part_id: int) -> Optional[Part]:
        try:
            return Part.get_by_id(part_id)
        except SQLAlchemyError as e:
            error_id(f"PartService.get failed: {e}", None)
            raise

    @with_request_id
    def fts_search(self, search_text: str, limit: int = 100) -> List[tuple]:
        try:
            limit = self._normalize_limit(limit)
            return Part.fts_search(search_text=search_text, limit=limit)
        except SQLAlchemyError as e:
            error_id(f"PartService.fts_search failed: {e}", None)
            raise

    @with_request_id
    def save(
        self,
        part_number: str,
        name: Optional[str] = None,
        oem_mfg: Optional[str] = None,
        model: Optional[str] = None,
        class_flag: Optional[str] = None,
        ud6: Optional[str] = None,
        type_: Optional[str] = None,
        notes: Optional[str] = None,
        documentation: Optional[str] = None,
        part_id: Optional[int] = None,
    ) -> Part:
        with self.db_config.main_session() as session:
            try:
                if part_id:
                    part = session.query(Part).filter_by(id=part_id).first()
                    if not part:
                        raise ValueError(f"Part with id {part_id} not found")
                    part.part_number = part_number
                    part.name = name
                    part.oem_mfg = oem_mfg
                    part.model = model
                    part.class_flag = class_flag
                    part.ud6 = ud6
                    part.type = type_
                    part.notes = notes
                    part.documentation = documentation
                    info_id(f"Updated Part id={part_id}", None)
                else:
                    part = Part(
                        part_number=part_number,
                        name=name,
                        oem_mfg=oem_mfg,
                        model=model,
                        class_flag=class_flag,
                        ud6=ud6,
                        type=type_,
                        notes=notes,
                        documentation=documentation,
                    )
                    session.add(part)
                    info_id(f"Created Part '{part_number}'", None)
                return part
            except SQLAlchemyError as e:
                error_id(f"PartService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, part_id: int) -> bool:
        with self.db_config.main_session() as session:
            try:
                part = session.query(Part).filter_by(id=part_id).first()
                if part:
                    session.delete(part)
                    info_id(f"Deleted Part id={part_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"PartService.remove failed: {e}", None)
                raise

    @with_request_id
    def find_or_create(
        self,
        part_number: str,
        name: Optional[str] = None,
        oem_mfg: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Part:
        with self.db_config.main_session() as session:
            try:
                part = session.query(Part).filter_by(part_number=part_number).first()
                if part:
                    info_id(f"Found existing Part '{part_number}'", None)
                else:
                    part = Part(
                        part_number=part_number,
                        name=name,
                        oem_mfg=oem_mfg,
                        model=model,
                    )
                    session.add(part)
                    session.commit()
                    info_id(f"Created new Part '{part_number}'", None)
                return part
            except SQLAlchemyError as e:
                error_id(f"PartService.find_or_create failed: {e}", None)
                raise

    @with_request_id
    def find_related(self, part_id: int) -> Optional[Dict[str, Any]]:
        with self.db_config.main_session() as session:
            try:
                part = session.query(Part).filter_by(id=part_id).first()
                if not part:
                    return None
                return {
                    "part": part,
                    "downward": {
                        "positions": part.part_position_image,
                        "problems": part.part_problem,
                        "tasks": part.part_task,
                        "drawings": part.drawing_part,
                    }
                }
            except SQLAlchemyError as e:
                error_id(f"PartService.find_related failed: {e}", None)
                raise