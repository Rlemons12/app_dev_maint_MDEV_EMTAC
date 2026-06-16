# blueprints/media_routes.py

import os
from pathlib import Path
from flask import Blueprint, send_file

from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id
from modules.emtacdb.emtacdb_fts import Image

media_bp = Blueprint("media_bp", __name__)


@media_bp.route("/images/<int:image_id>")
@with_request_id
def serve_image(image_id, request_id=None):
    """
    Universal image-serving endpoint.

    Used by:
      - Chatbot UI
      - Image panels
      - Compare tools
      - Any future consumer

    Contract:
      URL        : /images/<image_id>
      Storage    : DATABASE_PATH_IMAGES_FOLDER
      DB column  : Image.file_path (filename or legacy full path)
    """

    rid = request_id
    info_id(f"[MEDIA] Request to serve image id={image_id}", rid)

    db_config = DatabaseConfig()

    try:
        with db_config.get_main_session() as session:
            image = session.query(Image).filter_by(id=image_id).first()

            if not image:
                error_id(f"[MEDIA] Image id={image_id} not found", rid)
                return "Image not found", 404

            # --------------------------------------------------
            # CONFIG-DRIVEN PATH RESOLUTION (CRITICAL)
            # --------------------------------------------------
            # Always strip to filename to tolerate legacy DB rows
            filename = Path(image.file_path).name
            file_path = os.path.normpath(
                os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
            )

            info_id(f"[MEDIA] Resolved image path: {file_path}", rid)

            if not os.path.exists(file_path):
                error_id(f"[MEDIA] Image file missing: {file_path}", rid)
                return "Image file not found", 404

            # --------------------------------------------------
            # MIME TYPE RESOLUTION
            # --------------------------------------------------
            ext = Path(file_path).suffix.lower()
            mimetype_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
                ".webp": "image/webp",
                ".tif": "image/tiff",
                ".tiff": "image/tiff",
                ".svg": "image/svg+xml",
            }
            mimetype = mimetype_map.get(ext, "image/jpeg")

            return send_file(
                file_path,
                mimetype=mimetype,
                as_attachment=False,
                conditional=True,  # enables browser caching
            )

    except Exception as e:
        error_id(f"[MEDIA] Error serving image id={image_id}: {e}", rid, exc_info=True)
        return "Internal Server Error", 500
