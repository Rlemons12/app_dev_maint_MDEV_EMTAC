from flask import Blueprint, jsonify
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    with_request_id,
    info_id,
    error_id,
    get_request_id,
)
from modules.emtacdb.emtacdb_fts import Position, Area, SiteLocation


get_image_list_data_bp = Blueprint("get_image_list_data_bp", __name__)


@get_image_list_data_bp.route("/get_image_list_data")
@with_request_id
def get_list_data():
    """
    Fetch hierarchical data from database for image list interface.

    Keeps the existing response structure but avoids per-node logging noise.
    """
    request_id = get_request_id()

    info_id(
        "[IMAGE-LIST-DATA] Fetching complete hierarchical image list data",
        request_id,
    )

    db_config = DatabaseConfig()
    session = db_config.get_main_session()

    data = {
        "areas": [],
        "equipment_groups": [],
        "models": [],
        "asset_numbers": [],
        "locations": [],
        "subassemblies": [],
        "component_assemblies": [],
        "assembly_views": [],
        "site_locations": [],
    }

    try:
        site_locations = session.query(SiteLocation).all()
        data["site_locations"] = [
            {
                "id": site_location.id,
                "name": site_location.title,
            }
            for site_location in site_locations
        ]

        areas = session.query(Area).all()
        data["areas"] = [
            {
                "id": area.id,
                "name": area.name,
            }
            for area in areas
        ]

        for area in areas:
            equipment_groups = Position.get_dependent_items(
                session,
                "area",
                area.id,
            )

            data["equipment_groups"].extend(
                [
                    {
                        "id": equipment_group.id,
                        "name": equipment_group.name,
                        "area_id": area.id,
                    }
                    for equipment_group in equipment_groups
                ]
            )

            for equipment_group in equipment_groups:
                models = Position.get_dependent_items(
                    session,
                    "equipment_group",
                    equipment_group.id,
                )

                data["models"].extend(
                    [
                        {
                            "id": model.id,
                            "name": model.name,
                            "equipment_group_id": equipment_group.id,
                        }
                        for model in models
                    ]
                )

                for model in models:
                    asset_numbers = Position.get_dependent_items(
                        session,
                        "model",
                        model.id,
                        "asset_number",
                    )

                    data["asset_numbers"].extend(
                        [
                            {
                                "id": asset_number.id,
                                "number": asset_number.number,
                                "model_id": model.id,
                            }
                            for asset_number in asset_numbers
                        ]
                    )

                    locations = Position.get_dependent_items(
                        session,
                        "model",
                        model.id,
                        "location",
                    )

                    data["locations"].extend(
                        [
                            {
                                "id": location.id,
                                "name": location.name,
                                "model_id": model.id,
                            }
                            for location in locations
                        ]
                    )

                    for location in locations:
                        subassemblies = Position.get_dependent_items(
                            session,
                            "location",
                            location.id,
                        )

                        data["subassemblies"].extend(
                            [
                                {
                                    "id": subassembly.id,
                                    "name": subassembly.name,
                                    "location_id": location.id,
                                }
                                for subassembly in subassemblies
                            ]
                        )

                        for subassembly in subassemblies:
                            component_assemblies = Position.get_dependent_items(
                                session,
                                "subassembly",
                                subassembly.id,
                            )

                            data["component_assemblies"].extend(
                                [
                                    {
                                        "id": component_assembly.id,
                                        "name": component_assembly.name,
                                        "subassembly_id": subassembly.id,
                                    }
                                    for component_assembly in component_assemblies
                                ]
                            )

                            for component_assembly in component_assemblies:
                                assembly_views = Position.get_dependent_items(
                                    session,
                                    "component_assembly",
                                    component_assembly.id,
                                )

                                data["assembly_views"].extend(
                                    [
                                        {
                                            "id": assembly_view.id,
                                            "name": assembly_view.name,
                                            "component_assembly_id": component_assembly.id,
                                        }
                                        for assembly_view in assembly_views
                                    ]
                                )

        info_id(
            f"[IMAGE-LIST-DATA] Loaded hierarchy data | "
            f"areas={len(data['areas'])} | "
            f"equipment_groups={len(data['equipment_groups'])} | "
            f"models={len(data['models'])} | "
            f"asset_numbers={len(data['asset_numbers'])} | "
            f"locations={len(data['locations'])} | "
            f"subassemblies={len(data['subassemblies'])} | "
            f"component_assemblies={len(data['component_assemblies'])} | "
            f"assembly_views={len(data['assembly_views'])} | "
            f"site_locations={len(data['site_locations'])}",
            request_id,
        )

        return jsonify(data), 200

    except SQLAlchemyError as e:
        session.rollback()
        error_id(
            f"[IMAGE-LIST-DATA] Database error while fetching image list data: {e}",
            request_id,
            exc_info=True,
        )
        return jsonify(
            {
                "error": "Database error",
                "message": str(e),
            }
        ), 500

    except Exception as e:
        session.rollback()
        error_id(
            f"[IMAGE-LIST-DATA] Unexpected error while fetching image list data: {e}",
            request_id,
            exc_info=True,
        )
        return jsonify(
            {
                "error": "Server error",
                "message": str(e),
            }
        ), 500

    finally:
        session.close()