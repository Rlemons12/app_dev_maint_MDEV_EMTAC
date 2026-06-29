# blueprints/image_compare_bp.py

from __future__ import annotations

import os

from flask import Blueprint, jsonify, request, send_from_directory, send_file, flash

from modules.emtacdb.emtacdb_fts import Image
from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER, DATABASE_DIR
from modules.configuration.log_config import logger
from modules.configuration.config_env import DatabaseConfig
from modules.coordinators.image_compare_coordinator import ImageCompareCoordinator


db_config = DatabaseConfig()

image_compare_bp = Blueprint("image_compare_bp", __name__)

image_compare_coordinator = ImageCompareCoordinator()


@image_compare_bp.route("/upload_and_compare", methods=["POST"])
def upload_and_compare():
    """
    Upload a temporary query image and compare it against stored image embeddings.

    Existing frontend contract is preserved:
      Success:
        {
            "image_similarity_search": [...]
        }

      Error:
        {
            "error": "...",
            "image_similarity_search": []
        }
    """
    logger.info("Received request to upload and compare an image.")

    success, response_body, http_status = image_compare_coordinator.compare_uploaded_image(
        files=request.files,
        field_name="query_image",
        similarity_threshold=0.3,
        limit=10,
        cleanup_query_file=True,
    )

    if not success:
        logger.warning(
            "Image comparison request failed. "
            f"status={response_body.get('status')} "
            f"error={response_body.get('error')}"
        )

    return jsonify(response_body), http_status


@image_compare_bp.route("/uploads/<filename>")
def uploaded_file(filename):
    logger.info(f"Serving file {filename} from uploads.")
    return send_from_directory(DATABASE_PATH_IMAGES_FOLDER, filename)


@image_compare_bp.route("/serve_image/<int:image_id>")
def serve_image_route(image_id):
    logger.info(f"Received request to serve image with ID: {image_id}")

    with db_config.get_main_session() as session:
        try:
            return serve_image(session, image_id)
        except Exception:
            logger.exception(f"Error serving image {image_id}:")
            flash(f"Error serving image {image_id}", "error")
            return "Image not found", 404


def serve_image(session, image_id):
    logger.info(f"Attempting to retrieve image with ID: {image_id}")

    try:
        image = session.query(Image).filter_by(id=image_id).first()

        if image:
            file_path = os.path.join(DATABASE_DIR, image.file_path)

            if os.path.exists(file_path):
                logger.info(f"Serving file: {file_path}")
                return send_file(file_path, mimetype="image/jpeg", as_attachment=False)

            logger.error(f"File not found on disk: {file_path}")
            return "Image file not found", 404

        logger.error(f"No image found in database with ID: {image_id}")
        return "Image not found", 404

    except Exception:
        logger.exception("Unhandled error while serving the image:")
        return "Internal Server Error", 500