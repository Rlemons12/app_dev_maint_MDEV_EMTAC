import logging
from flask import jsonify
import os
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file
from modules.emtacdb.emtacdb_fts import Part
from sqlalchemy.exc import IntegrityError
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import PartsPositionImageAssociation, Image, Position
from modules.configuration.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS,DATABASE_DIR, BASE_DIR
from blueprints.bill_of_materials import update_part_bp
from modules.configuration.log_config import logger, with_request_id
from sqlalchemy import and_


@update_part_bp.route('/edit_part/<int:part_id>', methods=['GET', 'POST'])
@with_request_id
def edit_part(part_id):
    logger.info(f'Starting edit_part function for part_id: {part_id}')
    db_session = DatabaseConfig().get_main_session()
    # Check if this is an AJAX request
    is_ajax = request.form.get('ajax') == 'true' or request.args.get('ajax') == 'true'
    logger.debug(f'Database session obtained for editing part: {part_id}, is_ajax: {is_ajax}')

    # Use the Part.get_by_id class method to fetch the part
    part = Part.get_by_id(part_id=part_id, session=db_session)

    if part:
        logger.info(f'Found part for editing: {part.part_number} (ID: {part_id})')
    else:
        logger.warning(f'Part with ID {part_id} not found')
        if is_ajax:
            return jsonify({'success': False, 'message': 'Part not found.'}), 404
        else:
            flash("Part not found.", "error")
            return redirect(url_for('update_part_bp.search_part'))

    # Get existing images associated with this part
    try:
        part_images = db_session.query(Image).join(
            PartsPositionImageAssociation,
            PartsPositionImageAssociation.image_id == Image.id
        ).filter(
            PartsPositionImageAssociation.part_id == part_id
        ).all()
        logger.info(f'Retrieved {len(part_images)} images associated with part {part_id}')
    except Exception as e:
        logger.error(f'Error retrieving images for part {part_id}: {str(e)}')
        part_images = []

    # Get all positions for dropdown
    try:
        positions = db_session.query(Position).all()
        logger.debug(f'Retrieved {len(positions)} positions for dropdown')
    except Exception as e:
        logger.error(f'Error retrieving positions: {str(e)}')
        positions = []

    if request.method == 'POST':
        logger.info(f'Processing POST request for part {part_id}')

        # Log form data (excluding file contents for security)
        form_data = {k: v for k, v in request.form.items() if k != 'part_image'}
        logger.debug(f'Form data received: {form_data}')

        # Update part attributes from form input
        old_values = {
            'part_number': part.part_number,
            'name': part.name,
            'oem_mfg': part.oem_mfg,
            'model': part.model,
            'class_flag': part.class_flag,
            'ud6': part.ud6,
            'type': part.type
        }

        part.part_number = request.form.get('part_number')
        part.name = request.form.get('name')
        part.oem_mfg = request.form.get('oem_mfg')
        part.model = request.form.get('model')
        part.class_flag = request.form.get('class_flag')
        part.ud6 = request.form.get('ud6')
        part.type = request.form.get('type')
        part.notes = request.form.get('notes')
        part.documentation = request.form.get('documentation')

        # Log changes to part attributes
        new_values = {
            'part_number': part.part_number,
            'name': part.name,
            'oem_mfg': part.oem_mfg,
            'model': part.model,
            'class_flag': part.class_flag,
            'ud6': part.ud6,
            'type': part.type
        }

        for key, old_value in old_values.items():
            new_value = new_values[key]
            if old_value != new_value:
                logger.info(f'Updated part {part_id} {key}: "{old_value}" -> "{new_value}"')

        try:
            # Handle image upload if a file was submitted
            if 'part_image' in request.files and request.files['part_image'].filename != '':
                uploaded_file = request.files['part_image']
                logger.info(f'Image upload detected for part {part_id}: {uploaded_file.filename}')

                # Check if the file extension is allowed
                if not '.' in uploaded_file.filename or \
                        uploaded_file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
                    logger.warning(f'Invalid file type attempted: {uploaded_file.filename}')
                    if is_ajax:
                        return jsonify({'success': False,
                                        'message': 'File type not allowed. Please upload jpg, jpeg, png, or gif files only.'}), 400
                    else:
                        flash("File type not allowed. Please upload jpg, jpeg, png, or gif files only.", "error")
                        return render_template('bill_of_materials/bom_partials/edit_part.html', part=part,
                                               part_images=part_images,
                                               positions=positions)

                # Ensure the filename is secure
                filename = secure_filename(uploaded_file.filename)
                logger.debug(f'Secured filename: {filename}')

                # UPDATED: Use unified storage location (DATABASE_PATH_IMAGES_FOLDER)
                from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER

                logger.debug(f'Using unified storage path: {DATABASE_PATH_IMAGES_FOLDER}')

                # Create the directory if it doesn't exist
                if not os.path.exists(DATABASE_PATH_IMAGES_FOLDER):
                    logger.info(f'Creating unified image directory: {DATABASE_PATH_IMAGES_FOLDER}')
                    os.makedirs(DATABASE_PATH_IMAGES_FOLDER)

                # Create unique filename to avoid conflicts
                base_name, ext = os.path.splitext(filename)
                unique_filename = f"{part.part_number}_{base_name}{ext}".replace(' ', '_')

                # Save the file to unified location
                abs_file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, unique_filename)
                uploaded_file.save(abs_file_path)
                logger.info(f'Image saved to unified location: {abs_file_path}')

                # UPDATED: Store unified relative path in database (consistent with tools)
                rel_file_path = os.path.join("DB_IMAGES", unique_filename)
                logger.debug(f'Relative file path for database: {rel_file_path}')

                # Create a new Image record with unified relative path
                image_title = request.form.get('image_title', f"Image for {part.part_number}")
                image_description = request.form.get('image_description', f"Image for part {part.part_number}")

                # Check if an image with the same title and description already exists
                existing_image = db_session.query(Image).filter(
                    and_(Image.title == image_title, Image.description == image_description)
                ).first()

                if existing_image is not None and existing_image.file_path == rel_file_path:
                    logger.info(f"Image with the same title, description, and file path already exists: {image_title}")
                    new_image = existing_image
                else:
                    # Create new image with unified storage path
                    new_image = Image(
                        title=image_title,
                        description=image_description,
                        file_path=rel_file_path  # Store unified relative path: DB_IMAGES/filename.png
                    )
                    db_session.add(new_image)
                    db_session.flush()  # Flush to get the image ID
                    logger.info(f'Created new image record with ID: {new_image.id} using unified storage')

                # Get the position ID from the form and handle empty values properly
                position_id = request.form.get('position_id')

                # FIXED: Convert empty string to None for PostgreSQL compatibility
                if position_id and position_id.strip() and position_id not in ('', '__None', 'None'):
                    try:
                        position_id_int = int(position_id)
                        logger.debug(f'Position selected for image: {position_id_int}')
                    except (ValueError, TypeError):
                        position_id_int = None
                        logger.warning(f'Invalid position_id "{position_id}", using None')
                else:
                    position_id_int = None
                    logger.debug('No position selected for image')

                # Create the association with proper position_id handling
                association = PartsPositionImageAssociation(
                    part_id=part_id,
                    position_id=position_id_int,
                    image_id=new_image.id
                )

                db_session.add(association)

                # Update the log message to reflect the actual values
                if position_id_int:
                    logger.info(
                        f'Created association between part {part_id}, position {position_id_int}, and image {new_image.id}')
                else:
                    logger.info(f'Created association between part {part_id} and image {new_image.id} (no position)')

            # Handle image removal if requested
            if 'remove_image' in request.form:
                image_ids_to_remove = request.form.getlist('remove_image')
                logger.info(f'Request to remove {len(image_ids_to_remove)} images: {image_ids_to_remove}')

                for image_id in image_ids_to_remove:
                    # Find the association
                    association = db_session.query(PartsPositionImageAssociation).filter_by(
                        part_id=part_id,
                        image_id=image_id
                    ).first()

                    if association:
                        logger.debug(f'Found association for part {part_id} and image {image_id}')
                        db_session.delete(association)
                        logger.info(f'Deleted association for part {part_id} and image {image_id}')

                        # Optionally, also delete the image if it's not associated with any other parts
                        image_associations = db_session.query(PartsPositionImageAssociation).filter_by(
                            image_id=image_id
                        ).count()

                        if image_associations <= 1:  # 1 because we haven't committed the deletion yet
                            image = db_session.query(Image).filter_by(id=image_id).first()
                            if image:
                                logger.debug(f'Image {image_id} has no other associations, preparing to delete')

                                # UPDATED: Handle both old and new storage paths when deleting files
                                if image.file_path.startswith('DB_IMAGES'):
                                    # New unified storage path
                                    abs_file_path = os.path.join(DATABASE_DIR, image.file_path)
                                else:
                                    # Legacy path (static/uploads/parts)
                                    abs_file_path = os.path.join(BASE_DIR, image.file_path)

                                if os.path.exists(abs_file_path):
                                    logger.debug(f'Deleting image file: {abs_file_path}')
                                    os.remove(abs_file_path)
                                else:
                                    logger.warning(f'Image file not found for deletion: {abs_file_path}')

                                db_session.delete(image)
                                logger.info(f'Deleted image record with ID: {image_id}')
                    else:
                        logger.warning(f'No association found for part {part_id} and image {image_id}')

            db_session.commit()
            logger.info(f'Successfully committed all changes for part {part_id}')

            if is_ajax:
                return jsonify({
                    'success': True,
                    'message': 'Part updated successfully!'
                })
            else:
                flash("Part updated successfully!", "success")
                # After successful update, redirect to the same edit page to show the changes
                return redirect(url_for('update_part_bp.edit_part', part_id=part_id))

        except IntegrityError as ie:
            db_session.rollback()
            logger.error(f'IntegrityError during part update: {str(ie)}')
            if is_ajax:
                return jsonify({'success': False, 'message': 'Part number must be unique.'}), 400
            else:
                flash("Part number must be unique.", "error")
        except Exception as e:
            db_session.rollback()
            logger.error(f'Unexpected error during part update: {str(e)}', exc_info=True)
            if is_ajax:
                return jsonify({'success': False, 'message': f'An error occurred: {str(e)}'}), 500
            else:
                flash(f"An error occurred: {str(e)}", "error")

    # For GET requests
    if is_ajax:
        # For AJAX GET requests, redirect to edit_part_ajax endpoint
        return redirect(url_for('update_part_bp.edit_part_ajax', part_id=part_id))

    # Regular rendering for non-AJAX requests
    logger.debug(f'Rendering edit_part template for part {part_id}')
    return render_template('bill_of_materials/bom_partials/edit_part.html',
                           part=part,
                           part_images=part_images,
                           positions=positions)

# Add route to serve images directly if needed
@update_part_bp.route('/part_image/<int:image_id>')
@with_request_id
def serve_part_image(image_id):
    logger.info(f"Attempting to serve image with ID: {image_id}")
    db_session = DatabaseConfig().get_main_session()

    try:
        image = db_session.query(Image).filter_by(id=image_id).first()
        if image:
            logger.debug(f"Image found: {image.title}, File path: {image.file_path}")
            file_path = os.path.join(BASE_DIR, image.file_path)
            if os.path.exists(file_path):
                logger.info(f"Serving file: {file_path}")
                return send_file(file_path, mimetype='image/jpeg')
            else:
                logger.error(f"File not found: {file_path}")
                return "Image file not found", 404
        else:
            logger.error(f"Image not found with ID: {image_id}")
            return "Image not found", 404
    except Exception as e:
        logger.error(f"An error occurred while serving the image: {e}")
        return "Internal Server Error", 500

@update_part_bp.route('/search_part', methods=['GET'])
@with_request_id
def search_part():
    logger.info(f'Starting search_part function')
    db_session = DatabaseConfig().get_main_session()
    search_query = request.args.get('search_query', '')
    is_ajax = request.args.get('ajax', 'false') == 'true'

    # Get positions for dropdown regardless of search
    try:
        positions = db_session.query(Position).all()
        logger.debug(f'Retrieved {len(positions)} positions for dropdown')
    except Exception as e:
        logger.error(f'Error retrieving positions: {str(e)}')
        positions = []

    if search_query:
        logger.info(f'Searching for parts with query: {search_query}')

        try:
            # Use the Part.search class method
            parts = Part.search(
                search_text=search_query,
                session=db_session,
                limit=10  # Get more results for selection
            )

            # Check if any parts were found
            if parts and len(parts) > 0:
                logger.info(f'Found {len(parts)} parts matching query: {search_query}')

                if len(parts) == 1 and not is_ajax:
                    # For non-AJAX requests, if only one part found, redirect directly to edit_part
                    return redirect(url_for('update_part_bp.edit_part', part_id=parts[0].id))
                else:
                    # Return the search results partial template for AJAX requests
                    if is_ajax:
                        return render_template('bill_of_materials/bom_partials/search_parts_results.html',
                                               parts=parts,
                                               search_query=search_query)
                    else:
                        # For non-AJAX requests, return the full template
                        return render_template('bill_of_materials/bom_partials/search_parts_results.html',
                                               parts=parts,
                                               search_query=search_query,
                                               positions=positions)
            else:
                logger.info(f'No parts found matching query: {search_query}')
                if is_ajax:
                    return '<div class="alert alert-info">No parts found matching your search criteria.</div>'
                else:
                    flash("No parts found matching your search criteria.", "info")
        except Exception as e:
            logger.error(f'Error in search_part: {str(e)}', exc_info=True)
            if is_ajax:
                return f'<div class="alert alert-danger">An error occurred: {str(e)}</div>'
            else:
                flash(f"An error occurred during search: {str(e)}", "error")

    # If no parts found or no search query
    if is_ajax:
        return '<div class="alert alert-info">Please enter a search query.</div>'
    else:
        return render_template('bill_of_materials/bom_partials/edit_part.html',
                               part=None,
                               part_images=[],
                               positions=positions,
                               search_query=search_query)

@update_part_bp.route('/search_part_ajax', methods=['GET'])
@with_request_id
def search_part_ajax():
    logger.info(f'Starting search_part_ajax function')
    db_session = DatabaseConfig().get_main_session()
    search_query = request.args.get('search_query', '')

    try:
        if search_query:
            logger.info(f'Searching for parts with query: {search_query}')

            # Use the Part.search class method
            parts = Part.search(
                search_text=search_query,
                session=db_session,
                limit=10  # Get up to 10 matching parts
            )

            # Check if any parts were found
            if parts and len(parts) > 0:
                logger.info(f'Found {len(parts)} parts matching query: {search_query}')

                # Render just the search results section
                return render_template('bill_of_materials/bom_partials/search_parts_results.html',
                                       parts=parts,
                                       search_query=search_query)
            else:
                logger.info(f'No parts found matching query: {search_query}')
                return '<div class="alert alert-info">No parts found matching your search criteria.</div>'

    except Exception as e:
        logger.error(f'Error in search_part_ajax: {str(e)}', exc_info=True)
        return f'<div class="alert alert-danger">An error occurred during search: {str(e)}</div>'

    return '<div class="alert alert-info">Please enter a search query.</div>'

@update_part_bp.route('/edit_part_ajax/<int:part_id>', methods=['GET'])
@with_request_id
def edit_part_ajax(part_id):
    logger.info(f'Starting edit_part_ajax function for part_id: {part_id}')
    db_session = DatabaseConfig().get_main_session()

    # Get the part
    part = Part.get_by_id(part_id=part_id, session=db_session)

    if not part:
        logger.warning(f'Part with ID {part_id} not found')
        return '<div class="alert alert-danger">Part not found.</div>'

    logger.debug(f'Found part: {part.part_number} (ID: {part.id})')

    # Get existing images associated with this part
    try:
        part_images = db_session.query(Image).join(
            PartsPositionImageAssociation,
            PartsPositionImageAssociation.image_id == Image.id
        ).filter(
            PartsPositionImageAssociation.part_id == part_id
        ).all()
        logger.info(f'Retrieved {len(part_images)} images associated with part {part_id}')
    except Exception as e:
        logger.error(f'Error retrieving images for part {part_id}: {str(e)}')
        part_images = []

    # Get all positions for dropdown
    try:
        positions = db_session.query(Position).all()
        logger.debug(f'Retrieved {len(positions)} positions for dropdown')
    except Exception as e:
        logger.error(f'Error retrieving positions: {str(e)}')
        positions = []

    # Render just the edit part form (not the full template)
    try:
        logger.debug(f'About to render edit_part template for part {part_id}')
        html = render_template('bill_of_materials/bom_partials/edit_part.html',
                               part=part,
                               part_images=part_images,
                               positions=positions)
        logger.debug(f'Successfully rendered template for part {part_id}, HTML length: {len(html)}')
        return html
    except Exception as e:
        logger.error(f'Error rendering template for part {part_id}: {str(e)}', exc_info=True)
        return f'<div class="alert alert-danger">Error rendering part form: {str(e)}</div>'