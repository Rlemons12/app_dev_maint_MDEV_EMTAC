from __future__ import annotations

from typing import Optional, Any

from sqlalchemy.orm import Session, joinedload

from modules.emtacdb.emtacdb_fts import (
    Position,
    AssetNumber,
    Model,
    EquipmentGroup,
)
from ._service_utils import apply_non_none_updates, get_by_id, delete_by_id


class PositionService:
    FILTER_KEYS = (
        "area_id",
        "equipment_group_id",
        "model_id",
        "asset_number_id",
        "location_id",
        "subassembly_id",
        "component_assembly_id",
        "assembly_view_id",
        "site_location_id",
        "campus_id",
        "building_id",
    )

    @staticmethod
    def _filters(**kwargs) -> dict:
        return {k: kwargs.get(k) for k in PositionService.FILTER_KEYS}

    @staticmethod
    def create(session: Session, **kwargs) -> Position:
        obj = Position(**PositionService._filters(**kwargs))
        session.add(obj)
        session.flush()
        return obj

    @staticmethod
    def get_by_id(session: Session, entity_id: int) -> Optional[Position]:
        return get_by_id(session, Position, entity_id)

    @staticmethod
    def search(session: Session, **filters) -> list[Position]:
        return (
            session.query(Position)
            .filter_by(**PositionService._filters(**filters))
            .order_by(Position.id)
            .all()
        )

    @staticmethod
    def update(session: Session, entity_id: int, **kwargs) -> Optional[Position]:
        obj = PositionService.get_by_id(session, entity_id)
        if not obj:
            return None
        apply_non_none_updates(obj, PositionService._filters(**kwargs))
        session.flush()
        return obj

    @staticmethod
    def delete(session: Session, entity_id: int) -> bool:
        return delete_by_id(session, Position, entity_id)

    @staticmethod
    def find_or_create(session: Session, **kwargs) -> Position:
        filters = PositionService._filters(**kwargs)
        obj = session.query(Position).filter_by(**filters).first()
        if obj:
            return obj
        return PositionService.create(session, **filters)

    @staticmethod
    def get_corresponding_position_ids(session: Session, **filters) -> list[int]:
        rows = session.query(Position.id).filter_by(**PositionService._filters(**filters)).all()
        return [x for (x,) in rows]

    # ------------------------------------------------------------------
    # Bulk position build logic
    # ------------------------------------------------------------------
    @staticmethod
    def _payload(
        *,
        area_id: Optional[int] = None,
        equipment_group_id: Optional[int] = None,
        model_id: Optional[int] = None,
        asset_number_id: Optional[int] = None,
        location_id: Optional[int] = None,
        subassembly_id: Optional[int] = None,
        component_assembly_id: Optional[int] = None,
        assembly_view_id: Optional[int] = None,
        site_location_id: Optional[int] = None,
        campus_id: Optional[int] = None,
        building_id: Optional[int] = None,
    ) -> dict[str, Any]:
        return {
            "area_id": area_id,
            "equipment_group_id": equipment_group_id,
            "model_id": model_id,
            "asset_number_id": asset_number_id,
            "location_id": location_id,
            "subassembly_id": subassembly_id,
            "component_assembly_id": component_assembly_id,
            "assembly_view_id": assembly_view_id,
            "site_location_id": site_location_id,
            "campus_id": campus_id,
            "building_id": building_id,
        }

    @staticmethod
    def build_positions_from_existing_data(
            session: Session,
            *,
            create_asset_only_positions: bool = True,
            include_model_level_locations: bool = True,
    ) -> dict:
        """
        Build hierarchical positions:

            Area
                -> EquipmentGroup
                    -> Model
                        -> AssetNumber

        This replaces the previous combinational logic with true hierarchy expansion.
        """

        from sqlalchemy import text
        from modules.emtacdb.emtacdb_fts import (
            Position,
            Area,
            EquipmentGroup,
            Model,
            AssetNumber,
        )

        totals = {
            "areas": 0,
            "equipment_groups": 0,
            "models": 0,
            "assets": 0,
            "positions_created": 0,
            "errors": 0,
        }

        details = []

        try:
            # Wipe existing positions before rebuild
            session.execute(text("TRUNCATE TABLE position RESTART IDENTITY CASCADE"))
            session.flush()

            areas = session.query(Area).order_by(Area.name).all()

            for area in areas:
                totals["areas"] += 1

                # LEVEL 1 -> Area
                session.add(Position(area_id=area.id))
                totals["positions_created"] += 1
                details.append({
                    "status": "created_area_position",
                    "area_id": area.id,
                    "area_name": area.name,
                })

                equipment_groups = (
                    session.query(EquipmentGroup)
                    .filter(EquipmentGroup.area_id == area.id)
                    .order_by(EquipmentGroup.name)
                    .all()
                )

                for eg in equipment_groups:
                    totals["equipment_groups"] += 1

                    # LEVEL 2 -> Area + EquipmentGroup
                    session.add(
                        Position(
                            area_id=area.id,
                            equipment_group_id=eg.id,
                        )
                    )
                    totals["positions_created"] += 1
                    details.append({
                        "status": "created_equipment_group_position",
                        "area_id": area.id,
                        "area_name": area.name,
                        "equipment_group_id": eg.id,
                        "equipment_group_name": eg.name,
                    })

                    models = (
                        session.query(Model)
                        .filter(Model.equipment_group_id == eg.id)
                        .order_by(Model.name)
                        .all()
                    )

                    for model in models:
                        totals["models"] += 1

                        # LEVEL 3 -> Area + EquipmentGroup + Model
                        session.add(
                            Position(
                                area_id=area.id,
                                equipment_group_id=eg.id,
                                model_id=model.id,
                            )
                        )
                        totals["positions_created"] += 1
                        details.append({
                            "status": "created_model_position",
                            "area_id": area.id,
                            "area_name": area.name,
                            "equipment_group_id": eg.id,
                            "equipment_group_name": eg.name,
                            "model_id": model.id,
                            "model_name": model.name,
                        })

                        assets = (
                            session.query(AssetNumber)
                            .filter(AssetNumber.model_id == model.id)
                            .order_by(AssetNumber.number)
                            .all()
                        )

                        for asset in assets:
                            totals["assets"] += 1

                            # LEVEL 4 -> Area + EquipmentGroup + Model + Asset
                            session.add(
                                Position(
                                    area_id=area.id,
                                    equipment_group_id=eg.id,
                                    model_id=model.id,
                                    asset_number_id=asset.id,
                                )
                            )
                            totals["positions_created"] += 1
                            details.append({
                                "status": "created_asset_position",
                                "area_id": area.id,
                                "area_name": area.name,
                                "equipment_group_id": eg.id,
                                "equipment_group_name": eg.name,
                                "model_id": model.id,
                                "model_name": model.name,
                                "asset_number_id": asset.id,
                                "asset_number": asset.number,
                            })

            return {
                "success": True,
                "message": "Hierarchical position build complete",
                "data": {
                    "totals": totals,
                    "details": details,
                },
                "errors": [],
            }

        except Exception as exc:
            return {
                "success": False,
                "message": "Failed to build hierarchical positions",
                "data": {
                    "totals": totals,
                    "details": details,
                },
                "errors": [str(exc)],
            }