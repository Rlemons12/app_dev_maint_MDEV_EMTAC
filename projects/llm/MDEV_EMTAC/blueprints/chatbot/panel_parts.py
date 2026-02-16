from flask import Blueprint, jsonify
from modules.configuration.config_env import DatabaseConfig
from modules.services.parts_position_image_service import PartsPositionImageService
from modules.services.image_service import ImageService
from modules.emtacdb.emtacdb_fts import Drawing, DrawingPartAssociation
from modules.configuration.log_config import (
    info_id,
    debug_id,
    error_id,
)
from modules.services.drawing_part_association_service import DrawingPartAssociationService

panel_parts_bp = Blueprint("panel_parts", __name__)

@panel_parts_bp.route("/parts/<int:part_id>/images")
def get_part_images(part_id):
    session = DatabaseConfig().get_main_session()

    try:
        parts_position_image_service = PartsPositionImageService()
        image_service = ImageService()

        images = parts_position_image_service.get_images_for_part(
            session=session,
            part_id=part_id,
        )

        return jsonify({
            "part_id": part_id,
            "images": [image_service.serialize(i) for i in images],
        })

    finally:
        # 🔑 REQUIRED — prevents cursor + transaction leaks
        session.close()

@panel_parts_bp.route("/parts/<int:part_id>/drawings")
def get_part_drawings(part_id):
    session = DatabaseConfig().get_main_session()

    info_id(
        f"[GET PART DRAWINGS] request received for part_id={part_id}"
    )

    try:
        drawing_part_service = DrawingPartAssociationService()

        drawings = drawing_part_service.get_drawings_for_part(
            part_id=part_id,
            session=session,
        )

        debug_id(
            f"[GET PART DRAWINGS] drawings fetched for part_id={part_id}: "
            f"count={len(drawings)}"
        )

        # Optional: log each drawing (safe, low volume)
        for d in drawings:
            debug_id(
                f"[GET PART DRAWINGS] drawing: "
                f"id={d.id}, "
                f"number={d.drw_number}, "
                f"name={d.drw_name}, "
                f"rev={d.drw_revision}, "
                f"path={d.file_path}"
            )

        response = {
            "part_id": part_id,
            "drawings": [
                {
                    "id": d.id,
                    "drw_number": d.drw_number,
                    "drw_name": d.drw_name,
                    "drw_revision": d.drw_revision,
                    "file_path": d.file_path,
                }
                for d in drawings
            ],
        }

        info_id(
            f"[GET PART DRAWINGS] response ready for part_id={part_id}, "
            f"drawings_returned={len(response['drawings'])}"
        )

        return jsonify(response)

    except Exception as e:
        error_id(
            f"[GET PART DRAWINGS] error for part_id={part_id}: {e}",
            exc_info=True
        )
        raise

    finally:
        session.close()
        debug_id(
            f"[GET PART DRAWINGS] session closed for part_id={part_id}"
        )

