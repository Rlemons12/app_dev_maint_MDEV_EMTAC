from flask import Blueprint, jsonify, flash, redirect, request, url_for, render_template
from werkzeug.utils import secure_filename
import os
from sqlalchemy import and_

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, BASE_DIR
from modules.configuration.log_config import logger, with_request_id
from modules.emtacdb.emtacdb_fts import (
    Part,
    Image,
    Position,
    PartsPositionImageAssociation,
)
from modules.emtacdb.utlity.main_database.database import (
    add_image_to_db,
    add_parts_position_image_association,
)

from modules.coordinators.bill_of_materials_coordinator import BillOfMaterialsCoordinator

enter_new_part_bp = Blueprint("enter_new_part_bp", __name__)

coordinator = BillOfMaterialsCoordinator()


@enter_new_part_bp.route("/get_part_form_data", methods=["GET"])
# @login_required
@with_request_id
def get_part_form_data():
    """
    Route layer ONLY handles HTTP concerns.
    """
    logger.info("Route hit: /get_part_form_data")

    result = coordinator.get_part_form_data()

    status_code = result.pop("status_code", 200)
    return jsonify(result), status_code

@enter_new_part_bp.route('/enter_part', methods=['GET', 'POST'])
@with_request_id
def enter_part():
    logger.info("Enter part route accessed")
    session = db_config.get_main_session()

    # Get all positions for dropdown
    try:
        positions = session.query(Position).all()
        logger.debug(f"Retrieved {len(positions)} positions for dropdown")
    except Exception as e:
        logger.error(f"Error retrieving positions: {str(e)}", exc_info=True)
        positions = []

    if request.method == 'GET':
        return render_template(
            'bill_of_materials/bill_of_materials.html',
            positions=positions
        )

    if request.method == 'POST':
        try:
            logger.info("Processing POST request for new part")

            part_number = request.form['part_number']
            name = request.form['name']
            oem_mfg = request.form['oem_mfg']
            model = request.form['model']
            class_flag = request.form['class_flag']
            ud6 = request.form['ud6']
            type_value = request.form['type']
            notes = request.form['notes']
            documentation = request.form['documentation']

            logger.debug(
                f"Form data: part_number={part_number}, name={name}, model={model}"
            )

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
            session.flush()
            part_id = new_part.id
            logger.info(f"Created new part with ID: {part_id}")

            if 'part_image' in request.files and request.files['part_image'].filename != '':
                uploaded_file = request.files['part_image']
                logger.info(
                    f"Image upload detected for new part {part_id}: {uploaded_file.filename}"
                )

                if (
                    '.' not in uploaded_file.filename or
                    uploaded_file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS
                ):
                    logger.warning(
                        f"Invalid file type attempted: {uploaded_file.filename}"
                    )
                    flash(
                        "File type not allowed. Please upload jpg, jpeg, png, or gif files only.",
                        "error"
                    )
                    session.rollback()
                    return render_template(
                        'bill_of_materials/bill_of_materials.html',
                        positions=positions
                    )

                filename = secure_filename(uploaded_file.filename)
                logger.debug(f"Secured filename: {filename}")

                upload_folder = os.path.join(UPLOAD_FOLDER, 'parts')
                logger.debug(f"Upload folder path: {upload_folder}")

                if not os.path.exists(upload_folder):
                    logger.info(f"Creating upload directory: {upload_folder}")
                    os.makedirs(upload_folder)

                abs_file_path = os.path.join(upload_folder, filename)
                uploaded_file.save(abs_file_path)
                logger.info(f"Image saved to: {abs_file_path}")

                image_title = request.form.get('image_title', f"Image for {part_number}")
                image_description = request.form.get(
                    'image_description',
                    f"Image for part {part_number}"
                )
                position_id = request.form.get('position_id')

                try:
                    rel_file_path = os.path.relpath(abs_file_path, BASE_DIR)
                    logger.debug(f"Relative file path for database: {rel_file_path}")

                    image_id = add_image_to_db(
                        title=image_title,
                        file_path=rel_file_path,
                        position_id=position_id,
                        description=image_description
                    )

                    logger.info(f"Image added to database with ID: {image_id}")

                    if image_id:
                        add_parts_position_image_association(
                            part_id=part_id,
                            position_id=position_id,
                            image_id=image_id
                        )
                        logger.info(
                            f"Created association between part {part_id}, "
                            f"position {position_id}, and image {image_id}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error using utility functions to add image: {str(e)}",
                        exc_info=True
                    )

                    try:
                        logger.info(
                            "Falling back to direct database operations for image handling"
                        )

                        rel_file_path = os.path.relpath(abs_file_path, BASE_DIR)

                        existing_image = session.query(Image).filter(
                            and_(
                                Image.title == image_title,
                                Image.description == image_description
                            )
                        ).first()

                        if existing_image is not None and existing_image.file_path == rel_file_path:
                            logger.info(f"Image already exists: {image_title}")
                            new_image = existing_image
                        else:
                            new_image = Image(
                                title=image_title,
                                description=image_description,
                                file_path=rel_file_path
                            )
                            session.add(new_image)
                            session.flush()
                            logger.info(
                                f"Created new image record with ID: {new_image.id}"
                            )

                        association = PartsPositionImageAssociation(
                            part_id=part_id,
                            position_id=position_id,
                            image_id=new_image.id
                        )
                        session.add(association)
                        logger.info(
                            f"Created association between part {part_id}, "
                            f"position {position_id}, and image {new_image.id}"
                        )

                    except Exception as nested_e:
                        logger.error(
                            f"Error in fallback image handling: {str(nested_e)}",
                            exc_info=True
                        )

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

@enter_new_part_bp.route("/part_image/<int:image_id>", methods=["GET"])
@with_request_id
def serve_part_image(image_id: int):
    logger.info(f"Route hit: /part_image/{image_id}")

    result = coordinator.get_part_image(image_id=image_id)

    if result.get("success", False):
        return result["response"]

    return result.get("message", "Image not found"), result.get("status_code", 404)