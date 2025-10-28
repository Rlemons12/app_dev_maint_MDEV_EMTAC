from flask import Blueprint, request, flash, jsonify
from modules.emtacdb.emtacdb_fts import Image
from modules.emtacdb.utlity.main_database.database import create_thumbnail
from modules.configuration.config import DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from io import BytesIO
import base64

# Create a SQLAlchemy engine using the DATABASE_URL from your config
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = sessionmaker(bind=engine)

tsg_search_images_parts_bp = Blueprint('tsg_search_images_parts_bp', __name__)

@tsg_search_images_parts_bp.route('/tsg_search_images_parts')
def search_images_parts():
    session = Session()

    # Retrieve parameters from the request
    description = request.args.get('description', '')
    part_id = request.args.get('tsg_searchimage_part', None)  # Add part_id parameter

    # Debug statements
    print("Description:", description)
    print("Part ID:", part_id)

    # Start the query with the Image model
    query = session.query(Image)

    # Apply filters based on provided parameters
    if any([description, part_id]):
        # Create a list to hold all filter conditions
        filters = []

        # Add individual filters to the list
        if description:
            filters.append(Image.description.ilike(f"%{description}%"))
        if part_id:  # Add filter for part_id
            # Assuming Image has a relationship with Parts called 'part'
            filters.append(Image.part_id == int(part_id))

        # Apply all filters using AND logic
        query = query.filter(*filters)

    try:
        # Query the images based on the provided criteria
        images = query.all()

        if not images:
            # Flash message indicating no images found
            flash("No images found", "error")
            return jsonify(thumbnails=[])

        # Extract necessary attributes for each image and construct a list of dictionaries
        thumbnails = []
        for image in images:
            # Generate thumbnail for each image
            thumbnail = create_thumbnail(image.image_blob)  # Assuming image_blob is the binary image data
            thumbnail_bytes_io = BytesIO()
            thumbnail.save(thumbnail_bytes_io, format='JPEG')
            thumbnail_src = f"data:image/jpeg;base64,{base64.b64encode(thumbnail_bytes_io.getvalue()).decode()}"
            
            # Construct dictionary with image data including thumbnail source
            image_info = {
                'id': image.id,
                'title': image.title,
                'src': f'/serve_image/{image.id}',  # Corrected line
                'thumbnail_src': thumbnail_src
            }
            thumbnails.append(image_info)

        session.close()

        # Return thumbnails as JSON
        return jsonify(thumbnails=thumbnails)

    except SQLAlchemyError as e:
        # Handle any SQLAlchemy errors
        print(f"An error occurred while retrieving images: {e}")
        flash("An error occurred while retrieving images", "error")
        session.close()
        return jsonify(thumbnails=[])
