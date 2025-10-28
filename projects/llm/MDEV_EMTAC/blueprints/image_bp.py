import sys
import os
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, event, inspect
from datetime import datetime
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Blueprint, request, jsonify, redirect, url_for, render_template
from werkzeug.utils import secure_filename

# Updated imports to align with load_image_folder.py patterns
from modules.emtacdb.emtacdb_fts import (
    Session, Image, Position, ImagePositionAssociation
)
from modules.configuration.log_config import (
    logger, with_request_id, get_request_id, set_request_id,
    info_id, debug_id, warning_id, error_id, log_timed_operation
)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.config import DATABASE_DIR, DATABASE_URL, DATABASE_PATH_IMAGES_FOLDER, ALLOWED_EXTENSIONS
from plugins.ai_modules.ai_models import ModelsConfig
from modules.emtacdb.utlity.revision_database.auditlog import commit_audit_logs

# Initialize database configuration
db_config = DatabaseConfig()

# Create a blueprint for the image routes
image_bp = Blueprint('image_bp', __name__)

# Constants - aligned with load_image_folder.py
MINIMUM_SIZE = (100, 100)  # Define the minimum width and height for the image


# Helper Functions
def allowed_file(filename):
    """Check if file extension is allowed - aligned with load_image_folder.py"""
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in [ext.lower() for ext in ALLOWED_EXTENSIONS]


def clean_image_title(title, original_filename=None):
    """Clean image title using the same logic as load_image_folder.py"""
    import re

    # Common image extensions to remove from title
    COMMON_EXTENSIONS = [
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp',
        '.svg', '.ico', '.heic', '.raw', '.cr2', '.nef', '.arw'
    ]

    if not title and original_filename:
        title = os.path.splitext(original_filename)[0]

    if not title:
        return "image"

    original_title = title

    # Remove any image extensions embedded in the title
    pattern = '|'.join([re.escape(ext) for ext in COMMON_EXTENSIONS])
    regex = re.compile(f"({pattern})", re.IGNORECASE)

    # Iteratively remove extensions until none are found
    while True:
        new_title = regex.sub('', title)
        if new_title == title:
            break
        title = new_title

    # Clean up any dots that might be left at the end and other cleanup
    title = title.rstrip('.')
    title = re.sub(r'[^\w\s-]', '', title).strip()

    # If we've removed everything, use a default name
    if not title:
        title = "image"

    return title


# Route to serve the upload_search_database.html page
@image_bp.route('/upload_search_database', methods=['GET'])
@with_request_id
def upload_image_page(request_id=None):
    """Serve upload page with request ID tracking"""
    filename = request.args.get('filename', '')
    info_id(f"Serving upload page for filename: {filename}", request_id)
    return render_template('upload_search_database/upload_search_database.html', filename=filename)


@image_bp.route('/upload_image', methods=['GET', 'POST'])
@with_request_id
def upload_image(request_id=None):
    """Main image upload route - aligned with Image.add_to_db file handling"""
    try:
        if request.method == 'GET':
            return render_template('upload.html')

        info_id("Processing image upload request", request_id)

        # Use the same session management pattern as load_image_folder.py
        with db_config.main_session() as session:
            with log_timed_operation("upload_image_processing", request_id):
                # Validate image file
                image_file = request.files.get('image')
                if not image_file or image_file.filename == '':
                    error_id("No image file provided", request_id)
                    return jsonify({'message': 'No image file provided'}), 400

                if not allowed_file(image_file.filename):
                    error_id(f"Unsupported file format: {image_file.filename}", request_id)
                    return jsonify({'message': 'Unsupported file format'}), 400

                # Get and clean title using the same logic as load_image_folder.py
                title = request.form.get('title', '').strip()
                filename = secure_filename(image_file.filename)
                clean_title = clean_image_title(title, filename)

                info_id(f"Processing image with title: '{clean_title}'", request_id)

                # Collect metadata - same pattern as before but with better logging
                area_id = _safe_convert_to_int(request.form.get('area'), 'area', request_id)
                equipment_group_id = _safe_convert_to_int(request.form.get('equipment_group'), 'equipment_group',
                                                          request_id)
                model_id = _safe_convert_to_int(request.form.get('model'), 'model', request_id)
                asset_number_id = _safe_convert_to_int(request.form.get('asset_number'), 'asset_number', request_id)
                location_id = _safe_convert_to_int(request.form.get('location'), 'location', request_id)
                site_location_id = _safe_convert_to_int(request.form.get('site_location'), 'site_location', request_id)

                debug_id(f"Metadata: area_id={area_id}, equipment_group_id={equipment_group_id}, "
                         f"model_id={model_id}, asset_number_id={asset_number_id}, "
                         f"location_id={location_id}, site_location_id={site_location_id}", request_id)

                # Create or get position using the same method as load_image_folder.py
                position_id = Position.add_to_db(
                    session=session,
                    area_id=area_id,
                    equipment_group_id=equipment_group_id,
                    model_id=model_id,
                    asset_number_id=asset_number_id,
                    location_id=location_id,
                    site_location_id=site_location_id
                )

                if not position_id:
                    error_id("Failed to create or retrieve position", request_id)
                    return jsonify({'message': 'Failed to create or retrieve position'}), 500

                info_id(f"Using position ID: {position_id}", request_id)

                # Save file temporarily and let Image.add_to_db handle the final copying
                temp_file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
                image_file.save(temp_file_path)
                debug_id(f"Saved temporary file: {temp_file_path}", request_id)

                try:
                    # Get description
                    description = request.form.get('description', '').strip()
                    if not description:
                        description = 'Auto-imported image'

                    # Let Image.add_to_db handle all the file copying and processing
                    # Pass the temporary file path as the source
                    new_image = Image._add_to_db_internal(
                        session=session,
                        title=clean_title,
                        file_path=temp_file_path,  # Pass the temp file path as source
                        description=description,
                        position_id=position_id,
                        request_id=request_id
                    )

                    if not new_image:
                        error_id("Failed to add image to database", request_id)
                        return jsonify({'message': 'Failed to add image to database'}), 500

                    if isinstance(new_image, int):
                        info_id(f"Successfully created image with ID: {new_image}", request_id)
                    else:
                        info_id(f"Successfully created image with ID: {new_image.id}", request_id)

                    # Clean up temporary file if it still exists (Image.add_to_db should have copied it)
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            debug_id(f"Cleaned up temporary file: {temp_file_path}", request_id)
                        except Exception as cleanup_e:
                            warning_id(f"Could not clean up temp file: {cleanup_e}", request_id)

                    # Commit audit logs using the same pattern
                    try:
                        commit_audit_logs()
                        debug_id("Committed audit logs", request_id)
                    except Exception as audit_e:
                        warning_id(f"Failed to commit audit logs: {audit_e}", request_id)

                    return redirect(url_for('image_bp.upload_image_page', filename=filename))

                except Exception as e:
                    # Clean up temp file if something goes wrong
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            debug_id(f"Cleaned up temp file after error: {temp_file_path}", request_id)
                        except:
                            pass
                    raise

    except Exception as e:
        error_id(f"Error in upload_image: {str(e)}", request_id, exc_info=True)

        # Attempt to commit audit logs even on error
        try:
            commit_audit_logs()
        except Exception as audit_e:
            error_id(f"Failed to commit audit logs after error: {audit_e}", request_id)

        return jsonify({'message': 'An internal error occurred', 'error': str(e)}), 500


@image_bp.route('/add_image', methods=['POST'])
@with_request_id
def add_image(request_id=None):
    """Add image route - let Image.add_to_db handle file operations"""
    try:
        info_id("Processing add_image request", request_id)

        with db_config.main_session() as session:
            with log_timed_operation("add_image_processing", request_id):
                # Validate inputs
                title = request.form.get('title', '').strip()
                complete_document_id = _safe_convert_to_int(
                    request.form.get('complete_document_id'), 'complete_document_id', request_id
                )
                position_id = _safe_convert_to_int(
                    request.form.get('position_id'), 'position_id', request_id
                )

                debug_id(f"Form data - title: {title}, complete_document_id: {complete_document_id}, "
                         f"position_id: {position_id}", request_id)

                # Validate image file
                if 'image' not in request.files:
                    error_id("No image file in request", request_id)
                    return jsonify({'message': 'No image file provided'}), 400

                file = request.files['image']
                if file.filename == '':
                    error_id("No file selected", request_id)
                    return jsonify({'message': 'No selected file'}), 400

                if not allowed_file(file.filename):
                    error_id(f"Unsupported file type: {file.filename}", request_id)
                    return jsonify({'message': 'Unsupported file type'}), 400

                # Process filename and title
                filename = secure_filename(file.filename)
                clean_title = clean_image_title(title, filename)

                info_id(f"Processing file: {filename} with title: '{clean_title}'", request_id)

                # Save file temporarily - let Image.add_to_db handle the rest
                temp_file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
                file.save(temp_file_path)
                debug_id(f"Saved temporary file: {temp_file_path}", request_id)

                try:
                    # Get or generate description
                    description = request.form.get('description', '').strip()
                    if not description:
                        description = 'Auto-imported image'
                        info_id("Using default description", request_id)

                    # Let Image.add_to_db handle all file operations
                    new_image = Image.add_to_db(
                        session=session,
                        title=clean_title,
                        file_path=temp_file_path,  # Pass temp file as source
                        description=description,
                        position_id=position_id,
                        complete_document_id=complete_document_id,
                        request_id=request_id
                    )

                    if not new_image:
                        error_id("Failed to add image to database", request_id)
                        return jsonify({'message': 'Failed to add image to database'}), 500

                    info_id(f"Successfully created image with ID: {new_image.id}", request_id)

                    # Clean up temporary file if it still exists
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            debug_id(f"Cleaned up temporary file: {temp_file_path}", request_id)
                        except Exception as cleanup_e:
                            warning_id(f"Could not clean up temp file: {cleanup_e}", request_id)

                    # Commit audit logs
                    try:
                        commit_audit_logs()
                        debug_id("Committed audit logs", request_id)
                    except Exception as audit_e:
                        warning_id(f"Failed to commit audit logs: {audit_e}", request_id)

                    return redirect(url_for('image_bp.upload_image_page', filename=filename))

                except Exception as e:
                    # Clean up temp file if something goes wrong
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            debug_id(f"Cleaned up temp file after error: {temp_file_path}", request_id)
                        except:
                            pass
                    raise

    except Exception as e:
        error_id(f"Error in add_image: {str(e)}", request_id, exc_info=True)
        return jsonify({'message': 'An internal error occurred', 'error': str(e)}), 500


# Helper function for safe integer conversion with logging
def _safe_convert_to_int(value, field_name, request_id):
    """Safely convert form data to integers with proper logging"""
    try:
        if value and str(value).strip():
            result = int(value)
            debug_id(f"Converted {field_name}: {value} -> {result}", request_id)
            return result
        return None
    except (ValueError, TypeError) as e:
        warning_id(f"Invalid value for {field_name}: {value} - {str(e)}", request_id)
        return None