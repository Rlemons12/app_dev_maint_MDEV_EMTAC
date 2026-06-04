from flask import Blueprint, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

from modules.configuration.config import DATABASE_URL
from modules.configuration.log_config import (
    with_request_id,
    get_request_id,
    info_id,
    error_id,
)
from modules.emtacdb.emtacdb_fts import (
    Area,
    EquipmentGroup,
    Model,
    AssetNumber,
    Location,
)

get_list_data_bp = Blueprint("get_list_data_bp", __name__)

engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))


@get_list_data_bp.route("/get_list_data")
@with_request_id
def get_list_data():
    request_id = get_request_id()
    session = Session()

    try:
        areas = session.query(Area).all()
        equipment_groups = session.query(EquipmentGroup).all()
        models = session.query(Model).all()
        asset_numbers = session.query(AssetNumber).all()
        locations = session.query(Location).all()

        data = {
            "areas": [{"id": area.id, "name": area.name} for area in areas],
            "equipment_groups": [
                {
                    "id": group.id,
                    "name": group.name,
                    "area_id": group.area_id,
                }
                for group in equipment_groups
            ],
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "equipment_group_id": model.equipment_group_id,
                }
                for model in models
            ],
            "asset_numbers": [
                {
                    "id": number.id,
                    "number": number.number,
                    "model_id": number.model_id,
                }
                for number in asset_numbers
            ],
            "locations": [
                {
                    "id": location.id,
                    "name": location.name,
                    "model_id": location.model_id,
                }
                for location in locations
            ],
        }

        info_id(
            f"[LIST-DATA] Loaded dropdown data | "
            f"areas={len(data['areas'])} | "
            f"equipment_groups={len(data['equipment_groups'])} | "
            f"models={len(data['models'])} | "
            f"asset_numbers={len(data['asset_numbers'])} | "
            f"locations={len(data['locations'])}",
            request_id,
        )

        return jsonify(data), 200

    except Exception as e:
        session.rollback()
        error_id(
            f"[LIST-DATA] Failed loading dropdown data: {e}",
            request_id,
            exc_info=True,
        )
        return jsonify(
            {
                "error": "Failed to load dropdown data",
                "detail": str(e),
            }
        ), 500

    finally:
        Session.remove()