import logging
from pathlib import Path
from typing import Optional, Any

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import get_db_config
from modules.emtacdb.emtacdb_fts import (
    Area,
    EquipmentGroup,
    Model,
    AssetNumber,
    Position,
)

logger = logging.getLogger("position_hierarchy_loader")


def clean(value: Any) -> Optional[Any]:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    return value


def to_int(value: Any) -> Optional[int]:
    value = clean(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> Optional[str]:
    value = clean(value)
    if value is None:
        return None
    return str(value).strip()


def upsert_area(session, area_id: int, area_name: Optional[str]) -> Area:
    area = session.query(Area).filter(Area.id == area_id).first()
    if area:
        changed = False
        if area_name and area.name != area_name:
            area.name = area_name
            changed = True
        if changed:
            logger.info("Updated Area id=%s name=%s", area_id, area_name)
        return area

    area = Area(id=area_id, name=area_name or f"AREA_{area_id}")
    session.add(area)
    logger.info("Created Area id=%s name=%s", area_id, area.name)
    return area


def upsert_equipment_group(
    session,
    equipment_group_id: int,
    area_id: Optional[int],
    name: Optional[str],
    description: Optional[str],
) -> EquipmentGroup:
    row = session.query(EquipmentGroup).filter(
        EquipmentGroup.id == equipment_group_id
    ).first()

    if row:
        changed = False
        if name and row.name != name:
            row.name = name
            changed = True
        if area_id is not None and row.area_id != area_id:
            row.area_id = area_id
            changed = True
        if description is not None and row.description != description:
            row.description = description
            changed = True
        if changed:
            logger.info("Updated EquipmentGroup id=%s", equipment_group_id)
        return row

    row = EquipmentGroup(
        id=equipment_group_id,
        name=name or f"EQUIPMENT_GROUP_{equipment_group_id}",
        area_id=area_id,
        description=description,
    )
    session.add(row)
    logger.info("Created EquipmentGroup id=%s name=%s", equipment_group_id, row.name)
    return row


def upsert_model(
    session,
    model_id: int,
    equipment_group_id: Optional[int],
    name: Optional[str],
    description: Optional[str] = None,
) -> Model:
    row = session.query(Model).filter(Model.id == model_id).first()

    if row:
        changed = False
        if name and row.name != name:
            row.name = name
            changed = True
        if equipment_group_id is not None and row.equipment_group_id != equipment_group_id:
            row.equipment_group_id = equipment_group_id
            changed = True
        if description is not None and row.description != description:
            row.description = description
            changed = True
        if changed:
            logger.info("Updated Model id=%s", model_id)
        return row

    row = Model(
        id=model_id,
        name=name or f"MODEL_{model_id}",
        equipment_group_id=equipment_group_id,
        description=description,
    )
    session.add(row)
    logger.info("Created Model id=%s name=%s", model_id, row.name)
    return row


def upsert_asset_number(
    session,
    asset_number_id: int,
    model_id: Optional[int],
    number: Optional[str],
    description: Optional[str],
) -> AssetNumber:
    row = session.query(AssetNumber).filter(AssetNumber.id == asset_number_id).first()

    if row:
        changed = False
        if number and row.number != number:
            row.number = number
            changed = True
        if model_id is not None and row.model_id != model_id:
            row.model_id = model_id
            changed = True
        if description is not None and row.description != description:
            row.description = description
            changed = True
        if changed:
            logger.info("Updated AssetNumber id=%s", asset_number_id)
        return row

    row = AssetNumber(
        id=asset_number_id,
        number=number or f"ASSET_{asset_number_id}",
        model_id=model_id,
        description=description,
    )
    session.add(row)
    logger.info("Created AssetNumber id=%s number=%s", asset_number_id, row.number)
    return row


def resolve_position_hierarchy_from_asset(session, asset_number_id: Optional[int]):
    if not asset_number_id:
        return None, None, None

    model_id = AssetNumber.get_model_id_by_asset_number_id(session, asset_number_id)
    equipment_group_id = AssetNumber.get_equipment_group_id_by_asset_number_id(session, asset_number_id)
    area_id = AssetNumber.get_area_id_by_asset_number_id(session, asset_number_id)

    return area_id, equipment_group_id, model_id


def get_or_create_position(
    session,
    area_id: Optional[int],
    equipment_group_id: Optional[int],
    model_id: Optional[int],
    asset_number_id: Optional[int],
    location_id: Optional[int],
    subassembly_id: Optional[int],
    component_assembly_id: Optional[int],
    assembly_view_id: Optional[int],
    site_location_id: Optional[int],
    campus_id: Optional[int],
    building_id: Optional[int],
) -> int:
    filters = {
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

    existing = session.query(Position).filter_by(**filters).first()
    if existing:
        return existing.id

    row = Position(**filters)
    session.add(row)
    session.flush()
    return row.id


def load_hierarchy_and_positions(workbook_path: str):
    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    logger.info("Loading workbook: %s", workbook_path)

    area_df = pd.read_excel(workbook_path, sheet_name="area")
    equipment_group_df = pd.read_excel(workbook_path, sheet_name="equipment_group")
    model_df = pd.read_excel(workbook_path, sheet_name="model")
    asset_df = pd.read_excel(workbook_path, sheet_name="asset")
    position_df = pd.read_excel(workbook_path, sheet_name="position")

    db_config = get_db_config()
    session = db_config.get_main_session()

    stats = {
        "areas_created_or_updated": 0,
        "equipment_groups_created_or_updated": 0,
        "models_created_or_updated": 0,
        "assets_created_or_updated": 0,
        "positions_created_or_found": 0,
        "position_rows_skipped": 0,
        "position_rows_failed": 0,
    }

    try:
        # ------------------------------------------------------------------
        # 1. Area
        # ------------------------------------------------------------------
        for _, row in area_df.iterrows():
            area_id = to_int(row.get("area_id"))
            area_name = normalize_text(row.get("area"))

            if area_id is None:
                continue

            upsert_area(session, area_id=area_id, area_name=area_name)
            stats["areas_created_or_updated"] += 1

        session.commit()
        logger.info("Area load complete: %s", stats["areas_created_or_updated"])

        # ------------------------------------------------------------------
        # 2. Equipment Group
        # ------------------------------------------------------------------
        for _, row in equipment_group_df.iterrows():
            equipment_group_id = to_int(row.get("equipment_group_id"))
            area_id = to_int(row.get("area_id"))
            name = normalize_text(row.get("equipment_group"))
            description = normalize_text(row.get("equipment_group_description"))

            if equipment_group_id is None:
                continue

            upsert_equipment_group(
                session=session,
                equipment_group_id=equipment_group_id,
                area_id=area_id,
                name=name,
                description=description,
            )
            stats["equipment_groups_created_or_updated"] += 1

        session.commit()
        logger.info(
            "Equipment group load complete: %s",
            stats["equipment_groups_created_or_updated"],
        )

        # ------------------------------------------------------------------
        # 3. Model
        # ------------------------------------------------------------------
        for _, row in model_df.iterrows():
            model_id = to_int(row.get("model_id"))
            equipment_group_id = to_int(row.get("equipment_group_id"))
            name = normalize_text(row.get("model"))

            if model_id is None:
                continue

            upsert_model(
                session=session,
                model_id=model_id,
                equipment_group_id=equipment_group_id,
                name=name,
            )
            stats["models_created_or_updated"] += 1

        session.commit()
        logger.info("Model load complete: %s", stats["models_created_or_updated"])

        # ------------------------------------------------------------------
        # 4. Asset Number
        # ------------------------------------------------------------------
        for _, row in asset_df.iterrows():
            asset_number_id = to_int(row.get("asset_number_id"))
            model_id = to_int(row.get("model_id"))
            number = normalize_text(row.get("asset_number"))
            description = normalize_text(row.get("asset_description"))

            if asset_number_id is None:
                continue

            upsert_asset_number(
                session=session,
                asset_number_id=asset_number_id,
                model_id=model_id,
                number=number,
                description=description,
            )
            stats["assets_created_or_updated"] += 1

        session.commit()
        logger.info("Asset load complete: %s", stats["assets_created_or_updated"])

        # ------------------------------------------------------------------
        # 5. Position
        # ------------------------------------------------------------------
        for excel_row_num, row in enumerate(position_df.itertuples(index=False), start=2):
            try:
                area_id = to_int(getattr(row, "area_id", None))
                equipment_group_id = to_int(getattr(row, "equipment_group_id", None))
                model_id = to_int(getattr(row, "model_id", None))
                asset_number_id = to_int(getattr(row, "asset_number_id", None))
                location_id = to_int(getattr(row, "location_id", None))
                subassembly_id = to_int(getattr(row, "subassembly_id", None))
                component_assembly_id = to_int(getattr(row, "component_assembly_id", None))
                assembly_view_id = to_int(getattr(row, "assembly_view_id", None))
                site_location_id = to_int(getattr(row, "site_location_id", None))
                campus_id = to_int(getattr(row, "campus_id", None))
                building_id = to_int(getattr(row, "building_id", None))

                if not any([
                    area_id,
                    equipment_group_id,
                    model_id,
                    asset_number_id,
                    location_id,
                    subassembly_id,
                    component_assembly_id,
                    assembly_view_id,
                    site_location_id,
                    campus_id,
                    building_id,
                ]):
                    stats["position_rows_skipped"] += 1
                    continue

                if asset_number_id:
                    resolved_area_id, resolved_equipment_group_id, resolved_model_id = (
                        resolve_position_hierarchy_from_asset(session, asset_number_id)
                    )

                    if area_id is None:
                        area_id = resolved_area_id
                    if equipment_group_id is None:
                        equipment_group_id = resolved_equipment_group_id
                    if model_id is None:
                        model_id = resolved_model_id

                position_id = get_or_create_position(
                    session=session,
                    area_id=area_id,
                    equipment_group_id=equipment_group_id,
                    model_id=model_id,
                    asset_number_id=asset_number_id,
                    location_id=location_id,
                    subassembly_id=subassembly_id,
                    component_assembly_id=component_assembly_id,
                    assembly_view_id=assembly_view_id,
                    site_location_id=site_location_id,
                    campus_id=campus_id,
                    building_id=building_id,
                )

                stats["positions_created_or_found"] += 1

                if stats["positions_created_or_found"] % 250 == 0:
                    session.commit()
                    logger.info(
                        "Position progress: %s rows processed",
                        stats["positions_created_or_found"],
                    )

            except Exception as exc:
                session.rollback()
                stats["position_rows_failed"] += 1
                logger.exception(
                    "Failed processing position row %s: %s",
                    excel_row_num,
                    exc,
                )

        session.commit()

        logger.info("Load complete: %s", stats)
        return stats

    except SQLAlchemyError:
        session.rollback()
        logger.exception("Hierarchy/position load failed")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    workbook = r"E:\emtac\Database\DB_LOADSHEETS\position_load_template_with_drawing.xlsx"
    result = load_hierarchy_and_positions(workbook)
    print(result)