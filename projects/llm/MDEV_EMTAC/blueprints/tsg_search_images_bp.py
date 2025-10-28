import os
import logging
from flask import Blueprint, request, flash, jsonify, render_template
from modules.emtacdb.emtacdb_fts import Image, ImagePositionAssociation, Position
from modules.emtacdb.utlity.main_database.database import create_thumbnail
from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import SQLAlchemyError
from PIL import Image as PILImage
from io import BytesIO
import base64
from modules.configuration.config_env import DatabaseConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Change to DEBUG for more verbose logs
logger = logging.getLogger(__name__)

# Initialize the DatabaseConfig instance
db_config = DatabaseConfig()

# Blueprint setup
tsg_search_images_bp = Blueprint('tsg_search_images_bp', __name__)

@tsg_search_images_bp.route('/')
def search_images():
    # Create a session from the main database
    session = db_config.get_main_session()

    # Retrieve parameters from the request
    description = request.args.get('description', '')
    title = request.args.get('title', '')
    area_id = request.args.get('tsg_searchimage_area', None)
    equipment_group_id = request.args.get('tsg_searchimage_equipment_group', None)
    model_id = request.args.get('tsg_searchimage_model', None)
    asset_number_id = request.args.get('tsg_searchimage_asset_number', None)
    location_id = request.args.get('tsg_searchimage_location', None)

    # Logging parameters
    logger.debug(f"Search parameters - Description: {description}, Title: {title}, "
                 f"Area ID: {area_id}, Equipment Group ID: {equipment_group_id}, Model ID: {model_id}, "
                 f"Asset Number ID: {asset_number_id}, Location ID: {location_id}")

    page = request.args.get('page', 1, type=int)  # Default to page 1 if not provided
    per_page = 5  # Number of images per page
    offset = (page - 1) * per_page

    # Initialize an empty query and filter list
    query = None
    filters = []

    # Apply filters for Image attributes
    if description:
        filters.append(Image.description.ilike(f"%{description}%"))
    if title:
        filters.append(Image.title.ilike(f"%{title}%"))

    # If there are filters for Image attributes, create a query
    if filters:
        query = session.query(Image).filter(*filters)
    else:
        # Apply filters for Position attributes if no Image attributes filters are given
        query = session.query(Image).join(ImagePositionAssociation).join(Position).options(
            joinedload(Image.image_position_association).joinedload(ImagePositionAssociation.position)
        )
        if area_id:
            filters.append(Position.area_id == int(area_id))
        if equipment_group_id:
            filters.append(Position.equipment_group_id == int(equipment_group_id))
        if model_id:
            filters.append(Position.model_id == int(model_id))
        if asset_number_id:
            filters.append(Position.asset_number_id == int(asset_number_id))
        if location_id:
            filters.append(Position.location_id == int(location_id))
        if filters:
            query = query.filter(*filters)

    # If no filters are provided, show a flash message and redirect to the upload image page
    if not description and not title and not area_id and not equipment_group_id and not model_id and not asset_number_id and not location_id:
        flash("No search criteria provided", "error")
        logger.info("No search criteria provided. Redirecting to upload_search_database.html.")
        session.close()
        return render_template('upload_search_database/upload_search_database.html')

    try:
        # Query the images with pagination
        images = query.offset(offset).limit(per_page).all()

        if not images:
            flash("No images found", "error")
            logger.info("No images found with the provided criteria.")
            return jsonify(thumbnails=[])

        # Extract necessary attributes for each image and construct a list of dictionaries
        thumbnails = []
        for image in images:
            try:
                # Construct full image path before any exception handling
                image_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, image.file_path)
                logger.debug(f"Constructed image path: {image_path}")

                # Log DATABASE_PATH_IMAGES_FOLDER and image.file_path
                logger.debug(f"DATABASE_PATH_IMAGES_FOLDER: {DATABASE_PATH_IMAGES_FOLDER}")
                logger.debug(f"image.file_path: {image.file_path}")

                if not os.path.exists(image_path):
                    logger.error(f"File not found: {image_path}")
                    continue

                with open(image_path, 'rb') as img_file:
                    img = PILImage.open(img_file)
                    thumbnail = create_thumbnail(img)

                    if thumbnail is None:
                        logger.error("Error creating thumbnail: Thumbnail is None.")
                        continue

                    thumbnail_bytes_io = BytesIO()
                    thumbnail.save(thumbnail_bytes_io, format='JPEG')
                    thumbnail_src = f"data:image/jpeg;base64,{base64.b64encode(thumbnail_bytes_io.getvalue()).decode()}"

                    # Construct dictionary with image data including title, description, and thumbnail source
                    image_info = {
                        'id': image.id,
                        'title': image.title,
                        'description': image.description,  # Add description to the response
                        'src': f'/serve_image/{image.id}',  # Corrected line
                        'thumbnail_src': thumbnail_src
                    }
                    thumbnails.append(image_info)
            except Exception as e:
                logger.error(f"An error occurred while processing the image {image.file_path}: {e}")
                continue

        session.close()

        # Log the successful retrieval of images
        logger.info(f"Successfully retrieved {len(thumbnails)} images.")

        # Return thumbnails as JSON
        return jsonify(thumbnails=thumbnails)

    except SQLAlchemyError as e:
        # Handle any SQLAlchemy errors
        logger.error(f"An SQLAlchemy error occurred while retrieving images: {e}")
        flash("An error occurred while retrieving images", "error")
        session.close()
        return jsonify(thumbnails=[])

    except Exception as e:
        # Handle any other exceptions
        logger.error(f"An unexpected error occurred: {e}")
        flash("An unexpected error occurred while retrieving images", "error")
        session.close()
        return jsonify(thumbnails=[])
