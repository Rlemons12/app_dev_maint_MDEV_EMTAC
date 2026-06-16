from __future__ import annotations

from typing import Optional

from sqlalchemy import and_, func, or_, text
from sqlalchemy.orm import Session

from modules.emtacdb.emtacdb_fts import Part
from ._service_utils import apply_non_none_updates, get_by_id, delete_by_id


class PartService:
    """
    Service layer for Part.

    Rules:
    - accepts an active SQLAlchemy session
    - does not open/close sessions
    - does not commit/rollback
    - returns ORM objects or primitive IDs/tuples
    """

    DEFAULT_SEARCH_FIELDS = ("part_number", "name", "oem_mfg", "model")

    # -------------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------------
    @staticmethod
    def create(
        session: Session,
        *,
        part_number: str,
        name: str | None = None,
        oem_mfg: str | None = None,
        model: str | None = None,
        class_flag: str | None = None,
        ud6: str | None = None,
        type_: str | None = None,
        notes: str | None = None,
        documentation: str | None = None,
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
        session.flush()
        return part

    @staticmethod
    def get_by_id(session: Session, part_id: int) -> Optional[Part]:
        return get_by_id(session, Part, part_id)

    @staticmethod
    def get_by_part_number(session: Session, part_number: str) -> Optional[Part]:
        return session.query(Part).filter(Part.part_number == part_number).first()

    @staticmethod
    def update(
        session: Session,
        part_id: int,
        *,
        part_number: str | None = None,
        name: str | None = None,
        oem_mfg: str | None = None,
        model: str | None = None,
        class_flag: str | None = None,
        ud6: str | None = None,
        type_: str | None = None,
        notes: str | None = None,
        documentation: str | None = None,
    ) -> Optional[Part]:
        part = PartService.get_by_id(session, part_id)
        if not part:
            return None

        apply_non_none_updates(
            part,
            {
                "part_number": part_number,
                "name": name,
                "oem_mfg": oem_mfg,
                "model": model,
                "class_flag": class_flag,
                "ud6": ud6,
                "type": type_,
                "notes": notes,
                "documentation": documentation,
            },
        )
        session.flush()
        return part

    @staticmethod
    def delete(session: Session, part_id: int) -> bool:
        return delete_by_id(session, Part, part_id)

    @staticmethod
    def find_or_create(
        session: Session,
        *,
        part_number: str,
        name: str | None = None,
        oem_mfg: str | None = None,
        model: str | None = None,
        class_flag: str | None = None,
        ud6: str | None = None,
        type_: str | None = None,
        notes: str | None = None,
        documentation: str | None = None,
    ) -> Part:
        existing = PartService.get_by_part_number(session, part_number)
        if existing:
            return existing

        return PartService.create(
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

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------
    @staticmethod
    def search(
        session: Session,
        *,
        search_text: str | None = None,
        fields: list[str] | tuple[str, ...] | None = None,
        exact_match: bool = False,
        use_fts: bool = True,
        part_id: int | None = None,
        part_number: str | None = None,
        name: str | None = None,
        oem_mfg: str | None = None,
        model: str | None = None,
        class_flag: str | None = None,
        ud6: str | None = None,
        type_: str | None = None,
        notes: str | None = None,
        documentation: str | None = None,
        limit: int = 100,
    ) -> list[Part]:
        """
        Comprehensive search for Part objects.

        Supports:
        - exact field filtering
        - partial ILIKE text filtering
        - PostgreSQL FTS when enabled and available
        """
        query = session.query(Part)
        filters = []
        use_fts_ranking = False

        # ------------------------------------------------------------------
        # search_text
        # ------------------------------------------------------------------
        if search_text:
            search_text = search_text.strip()
            if search_text:
                if use_fts and hasattr(Part, "search_vector"):
                    try:
                        ts_query = func.plainto_tsquery("english", search_text)
                        filters.append(Part.search_vector.op("@@")(ts_query))
                        query = query.add_columns(
                            func.ts_rank(Part.search_vector, ts_query).label("rank")
                        )
                        use_fts_ranking = True
                    except Exception:
                        use_fts = False

                if not use_fts:
                    search_fields = fields or PartService.DEFAULT_SEARCH_FIELDS
                    text_filters = []

                    for field_name in search_fields:
                        if hasattr(Part, field_name):
                            field = getattr(Part, field_name)
                            if exact_match:
                                text_filters.append(field == search_text)
                            else:
                                text_filters.append(field.ilike(f"%{search_text}%"))

                    if text_filters:
                        filters.append(or_(*text_filters))

        # ------------------------------------------------------------------
        # specific field filters
        # ------------------------------------------------------------------
        if part_id is not None:
            filters.append(Part.id == part_id)

        if part_number is not None:
            filters.append(
                Part.part_number == part_number
                if exact_match
                else Part.part_number.ilike(f"%{part_number}%")
            )

        if name is not None:
            filters.append(
                Part.name == name if exact_match else Part.name.ilike(f"%{name}%")
            )

        if oem_mfg is not None:
            filters.append(
                Part.oem_mfg == oem_mfg
                if exact_match
                else Part.oem_mfg.ilike(f"%{oem_mfg}%")
            )

        if model is not None:
            filters.append(
                Part.model == model if exact_match else Part.model.ilike(f"%{model}%")
            )

        if class_flag is not None:
            filters.append(
                Part.class_flag == class_flag
                if exact_match
                else Part.class_flag.ilike(f"%{class_flag}%")
            )

        if ud6 is not None:
            filters.append(
                Part.ud6 == ud6 if exact_match else Part.ud6.ilike(f"%{ud6}%")
            )

        if type_ is not None:
            filters.append(
                Part.type == type_ if exact_match else Part.type.ilike(f"%{type_}%")
            )

        if notes is not None:
            filters.append(
                Part.notes == notes if exact_match else Part.notes.ilike(f"%{notes}%")
            )

        if documentation is not None:
            filters.append(
                Part.documentation == documentation
                if exact_match
                else Part.documentation.ilike(f"%{documentation}%")
            )

        # ------------------------------------------------------------------
        # apply filters / ordering / limit
        # ------------------------------------------------------------------
        if filters:
            query = query.filter(and_(*filters))

        if use_fts_ranking:
            query = query.order_by(text("rank DESC"))
        else:
            query = query.order_by(Part.part_number)

        query = query.limit(limit)

        if use_fts_ranking:
            raw_results = query.all()
            return [result[0] for result in raw_results]

        return query.all()

    @staticmethod
    def fts_search(
        session: Session,
        *,
        search_text: str,
        limit: int = 100,
    ) -> list[tuple[Part, float]]:
        """
        Dedicated PostgreSQL Full Text Search returning (Part, relevance_score).
        """
        ts_query = func.plainto_tsquery("english", search_text)
        rank = func.ts_rank(Part.search_vector, ts_query)

        results = (
            session.query(Part, rank.label("relevance"))
            .filter(Part.search_vector.op("@@")(ts_query))
            .order_by(rank.desc())
            .limit(limit)
            .all()
        )

        return [(row[0], float(row[1])) for row in results]

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def exists_by_part_number(session: Session, part_number: str) -> bool:
        return (
            session.query(Part.id)
            .filter(Part.part_number == part_number)
            .first()
            is not None
        )

    @staticmethod
    def list_all(session: Session, limit: int = 1000) -> list[Part]:
        return (
            session.query(Part)
            .order_by(Part.part_number, Part.id)
            .limit(limit)
            .all()
        )