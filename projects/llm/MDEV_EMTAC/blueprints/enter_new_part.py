from flask import Blueprint, jsonify, flash, redirect, request, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import os
from sqlalchemy import and_
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, BASE_DIR,DATABASE_DIR
from modules.configuration.log_config import logger
from modules.emtacdb.emtacdb_fts import Part, Model, Image, Position, PartsPositionImageAssociation
from modules.emtacdb.utlity.main_database.database import add_image_to_db, add_parts_position_image_association

# Initialize the database configuration
db_config = DatabaseConfig()

enter_new_part_bp = Blueprint('enter_new_part_bp', __name__)


@enter_new_part_bp.route('/get_part_form_data', methods=['GET'])
def get_part_form_data():
    session = db_config.get_main_session()

    try:
        # Fetch models from the database
        models = session.query(Model).all()

        # Fetch positions for the dropdown
        positions = session.query(Position).all()

        data = {
            'models': [{'id': model.id, 'name': model.name} for model in models],
            'positions': [{'id': position.id, 'name': position.name} for position in positions]
        }

        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in get_part_form_data: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})
    finally:
        session.close()


@enter_new_part_bp.route('/enter_part', methods=['GET', 'POST'])
def enter_part():
    logger.info("Enter part route accessed")
    session = db_config.get_main_session()

    # Get all positions for dropdown
    try:
        positions = session.query(Position).all()
        logger.debug(f'Retrieved {len(positions)} positions for dropdown')
    except Exception as e:
        logger.error(f'Error retrieving positions: {str(e)}')
        positions = []

    if request.method == 'GET':
        # If it's a GET request, show the form (or redirect elsewhere if needed)
        return render_template('bill_of_materials/bill_of_materials.html', positions=positions)

    if request.method == 'POST':
        try:
            logger.info("Processing POST request for new part")

            # Fetch form data and process POST request
            part_number = request.form['part_number']
            name = request.form['name']
            oem_mfg = request.form['oem_mfg']
            model = request.form['model']
            class_flag = request.form['class_flag']
            ud6 = request.form['ud6']
            type_value = request.form['type']
            notes = request.form['notes']
            documentation = request.form['documentation']

            logger.debug(f"Form data: part_number={part_number}, name={name}, model={model}")

            # Create a new Part entry
            new_part = Part(
                part_number=part_number,
                name=name,
                oem_mfg=oem_mfg,
                model=model,
                class_flag=class_flag,
                ud6=ud6,
                type=type_value,
                notes=notes,
                documentation=documentation
            )

            session.add(new_part)
            session.flush()  # Flush to get the part ID before committing
            part_id = new_part.id
            logger.info(f"Created new part with ID: {part_id}")

            # Handle image upload if a file was submitted
            if 'part_image' in request.files and request.files['part_image'].filename != '':
                uploaded_file = request.files['part_image']
                logger.info(f'Image upload detected for new part {part_id}: {uploaded_file.filename}')

                # Check if the file extension is allowed
                if not '.' in uploaded_file.filename or \
                        uploaded_file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
                    logger.warning(f'Invalid file type attempted: {uploaded_file.filename}')
                    flash("File type not allowed. Please upload jpg, jpeg, png, or gif files only.", "error")
                    session.rollback()
                    return render_template('bill_of_materials/bill_of_materials.html', positions=positions)

                # Ensure the filename is secure
                filename = secure_filename(uploaded_file.filename)
                logger.debug(f'Secured filename: {filename}')

                # Create upload folder for parts
                upload_folder = os.path.join(UPLOAD_FOLDER, 'parts')
                logger.debug(f'Upload folder path: {upload_folder}')

                # Create the directory if it doesn't exist
                if not os.path.exists(upload_folder):
                    logger.info(f'Creating upload directory: {upload_folder}')
                    os.makedirs(upload_folder)

                # Save the file with absolute path
                abs_file_path = os.path.join(upload_folder, filename)
                uploaded_file.save(abs_file_path)
                logger.info(f'Image saved to: {abs_file_path}')

                # Get image title and description from form
                image_title = request.form.get('image_title', f"Image for {part_number}")
                image_description = request.form.get('image_description', f"Image for part {part_number}")
                position_id = request.form.get('position_id')

                # Option 1: Use the add_image_to_db utility function if available
                try:
                    # Calculate relative path for storage in database
                    rel_file_path = os.path.relpath(abs_file_path, BASE_DIR)
                    logger.debug(f'Relative file path for database: {rel_file_path}')

                    # Add image to database using utility function
                    image_id = add_image_to_db(
                        title=image_title,
                        file_path=rel_file_path,
                        position_id=position_id,
                        description=image_description
                    )

                    logger.info(f"Image added to database with ID: {image_id}")

                    # Create association between part and image
                    if image_id:
                        add_parts_position_image_association(
                            part_id=part_id,
                            position_id=position_id,
                            image_id=image_id
                        )
                        logger.info(
                            f"Created association between part {part_id}, position {position_id}, and image {image_id}")
                except Exception as e:
                    logger.error(f"Error using utility functions to add image: {str(e)}", exc_info=True)

                    # Option 2: Fall back to direct database operations if utility functions fail
                    try:
                        logger.info("Falling back to direct database operations for image handling")
                        # Calculate relative path for storage in database
                        rel_file_path = os.path.relpath(abs_file_path, BASE_DIR)

                        # Check if image already exists
                        existing_image = session.query(Image).filter(
                            and_(Image.title == image_title, Image.description == image_description)
                        ).first()

                        if existing_image is not None and existing_image.file_path == rel_file_path:
                            logger.info(f"Image already exists: {image_title}")
                            new_image = existing_image
                        else:
                            # Create new image record
                            new_image = Image(
                                title=image_title,
                                description=image_description,
                                file_path=rel_file_path
                            )
                            session.add(new_image)
                            session.flush()
                            logger.info(f"Created new image record with ID: {new_image.id}")

                        # Create association
                        association = PartsPositionImageAssociation(
                            part_id=part_id,
                            position_id=position_id,
                            image_id=new_image.id
                        )
                        session.add(association)
                        logger.info(
                            f"Created association between part {part_id}, position {position_id}, and image {new_image.id}")
                    except Exception as nested_e:
                        logger.error(f"Error in fallback image handling: {str(nested_e)}", exc_info=True)
                        # Continue execution - we'll still create the part even if image handling fails

            # Commit all changes
            session.commit()
            logger.info(f"Successfully committed all changes for new part {part_id}")
            flash('Part successfully entered!', 'success')
            return redirect(url_for('enter_new_part_bp.enter_part'))

        except Exception as e:
            session.rollback()
            logger.error(f"Error entering part: {str(e)}", exc_info=True)
            flash(f'Error entering part: {str(e)}', 'error')
            return redirect(url_for('enter_new_part_bp.enter_part'))
        finally:
            session.close()


# Add route to serve part images
@enter_new_part_bp.route('/part_image/<int:image_id>')
def serve_part_image(image_id):
    """
    Flask route that serves image files using the Image class method.
    """
    # Call the class method
    success, response, status_code = Image.serve_file(image_id)

    if success:
        # Return the Flask response object directly
        return response
    else:
        # Return error message with appropriate status code
        return response, status_code