from flask import Blueprint, request, render_template, flash
from modules.emtacdb.emtacdb_fts import Image, Position, ImagePositionAssociation
from modules.emtacdb.utlity.main_database.database import serve_image
from modules.configuration.config import DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, joinedload
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a SQLAlchemy engine using the DATABASE_URL from your config
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = sessionmaker(bind=engine)

search_images_bp = Blueprint('search_images_bp', __name__)

@search_images_bp.route('/serve_image/<int:image_id>')
def serve_image_route(image_id):
    logger.debug(f"Request to serve image with ID: {image_id}")
    with Session() as session:
        try:
            return serve_image(session, image_id)
        except Exception as e:
            logger.error(f"Error serving image {image_id}: {e}")
            flash(f"Error serving image {image_id}", "error")
            return "Image not found", 404

@search_images_bp.route('/')
def search_images():
    session = Session()

    # Retrieve parameters from the request
    description = request.args.get('description', '')
    title = request.args.get('title', '')
    area_id = request.args.get('searchimage_area', None)
    equipment_group_id = request.args.get('searchimage_equipment_group', None)
    model_id = request.args.get('searchimage_model', None)
    asset_number_id = request.args.get('searchimage_asset_number', None)
    location_id = request.args.get('searchimage_location', None)

    # Logging parameters
    logger.debug(f"Search parameters - Description: {description}, Title: {title}, Area ID: {area_id}, Equipment Group ID: {equipment_group_id}, Model ID: {model_id}, Asset Number ID: {asset_number_id}, Location ID: {location_id}")

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

    # Apply filters for Position attributes if no Image attributes filters are given
    if not query:
        query = session.query(Image).join(ImagePositionAssociation).join(Position).options(
            joinedload(Image.image_position_association).joinedload(ImagePositionAssociation.position)
        )
        filters = []
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
    if not filters:
        flash("No search criteria provided", "error")
        logger.info("No search criteria provided. Redirecting to upload_search_database.html.")
        session.close()
        return render_template('upload_search_database/upload_search_database.html')

    # Paginate the results
    images = query.offset(offset).limit(per_page).all()
    total_images = query.count()
    total_pages = (total_images + per_page - 1) // per_page

    session.close()

    logger.debug(f"Total images found: {total_images}")
    logger.debug(f"Total pages: {total_pages}")

    if not images:
        flash("No images found matching the criteria", "error")
        logger.info("No images found. Redirecting to upload_search_database.html.")
        return render_template('upload_search_database/upload_search_database.html')

    logger.info(f"Returning {len(images)} images for display on page {page}.")
    return render_template('image_results.html', images=images, description=description, title=title, page=page, total_pages=total_pages)
