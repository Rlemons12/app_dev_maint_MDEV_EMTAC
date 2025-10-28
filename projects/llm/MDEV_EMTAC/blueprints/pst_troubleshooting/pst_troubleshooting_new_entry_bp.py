# blueprints/pst_troubleshoot_new_entry_bp.py

from flask import Blueprint, render_template, request, jsonify, flash
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import with_request_id
from modules.emtacdb.emtacdb_fts import (
    Problem, Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation,
    Part, Position, ProblemPositionAssociation
)
from sqlalchemy.exc import SQLAlchemyError
from modules.configuration.log_config import logger,with_request_id


# Initialize Blueprint
pst_troubleshoot_new_entry_bp = Blueprint('pst_troubleshoot_new_entry_bp', __name__,)

# Initialize Database Config
db_config = DatabaseConfig()

@pst_troubleshoot_new_entry_bp.route('/', methods=['GET'])
@with_request_id
def new_entry_form():
    """
    Render the New Problem Entry Form.
    """
    session = db_config.get_main_session()
    try:
        areas = session.query(Area).all()
        parts = session.query(Part).all()
        drawings = session.query(Drawing).all()
        return render_template('pst_troubleshoot_new_entry.html', areas=areas, parts=parts, drawings=drawings)
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        flash('An error occurred while loading the new entry form.', 'danger')
        return render_template('error.html'), 500
    finally:
        session.close()


@pst_troubleshoot_new_entry_bp.route('/create_problem', methods=['POST'])
@with_request_id
def create_new_problem():
    """
    Handle the creation of a new problem via AJAX, including position handling and association.
    """
    session = db_config.get_main_session()
    try:
        # Extract form data
        problem_name = request.form.get('name')  # Make sure this matches your form field
        problem_description = request.form.get('description')
        area_id = request.form.get('area_id')
        equipment_group_id = request.form.get('equipment_group_id')
        model_id = request.form.get('model_id')

        # Get both ID and input string for asset number and location
        asset_number_id = request.form.get('asset_number_id')
        asset_number_input = request.form.get('asset_number')
        location_id = request.form.get('location_id')
        location_input = request.form.get('location')
        site_location_id = request.form.get('site_location_id')

        logger.info(f"Form Data - Name: {problem_name}, Description: {problem_description}, Area: {area_id}, "
                    f"Equipment Group: {equipment_group_id}, Model: {model_id}, Asset Number ID: {asset_number_id}, "
                    f"Asset Number Input: {asset_number_input}, Location ID: {location_id}, "
                    f"Location Input: {location_input}, Site Location: {site_location_id}")

        # Validate required fields
        if not all([problem_name, problem_description, area_id, equipment_group_id, model_id]):
            return jsonify({'success': False, 'message': 'All required fields must be filled out.'}), 400

        # Handle asset number logic
        asset_number = None
        if asset_number_id:
            asset_number = session.query(AssetNumber).filter_by(id=asset_number_id).first()
            if asset_number:
                logger.info(f"Found asset number with ID: {asset_number_id}")
        elif asset_number_input:
            # Try to interpret as ID first
            try:
                aid = int(asset_number_input)
                asset_number = session.query(AssetNumber).filter_by(id=aid).first()
                if asset_number:
                    logger.info(f"Found asset number by ID: {aid}")
            except (ValueError, TypeError):
                pass

            # If not found, try by number
            if not asset_number:
                asset_number = session.query(AssetNumber).filter_by(number=asset_number_input,
                                                                    model_id=model_id).first()
                if not asset_number:
                    asset_number = AssetNumber(number=asset_number_input, model_id=model_id)
                    session.add(asset_number)
                    session.commit()
                    logger.info(f"Created new Asset Number: {asset_number_input}")

        # Handle location logic
        location = None
        if location_id:
            location = session.query(Location).filter_by(id=location_id).first()
            if location:
                logger.info(f"Found location with ID: {location_id}")
        elif location_input:
            # Try to interpret as ID first
            try:
                lid = int(location_input)
                location = session.query(Location).filter_by(id=lid).first()
                if location:
                    logger.info(f"Found location by ID: {lid}")
            except (ValueError, TypeError):
                pass

            # If not found, try by name
            if not location:
                location = session.query(Location).filter_by(name=location_input, model_id=model_id).first()
                if not location:
                    location = Location(name=location_input, model_id=model_id)
                    session.add(location)
                    session.commit()
                    logger.info(f"Created new Location: {location_input}")

        # Handle site location
        site_location = None
        if site_location_id == 'new':
            new_site_location_title = request.form.get('new_siteLocation_title')
            new_site_location_room_number = request.form.get('new_siteLocation_room_number')
            if not all([new_site_location_title, new_site_location_room_number]):
                return jsonify(
                    {'success': False, 'message': 'New Site Location title and room number are required.'}), 400
            site_location = SiteLocation(title=new_site_location_title, room_number=new_site_location_room_number)
            session.add(site_location)
            session.commit()
            logger.info(f"Created new Site Location: {new_site_location_title}")
        elif site_location_id:
            site_location = session.query(SiteLocation).filter_by(id=site_location_id).first()
            if not site_location:
                return jsonify({'success': False, 'message': 'Selected Site Location does not exist.'}), 400

        # Create the new Problem
        new_problem = Problem(
            name=problem_name,
            description=problem_description
        )
        session.add(new_problem)
        session.commit()
        logger.info(f"Created new Problem: {problem_name} with ID {new_problem.id}")

        # Create or find Position
        position = session.query(Position).filter_by(
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number.id if asset_number else None,
            location_id=location.id if location else None,
            site_location_id=site_location.id if site_location else None
        ).first()

        if not position:
            position = Position(
                area_id=area_id,
                equipment_group_id=equipment_group_id,
                model_id=model_id,
                asset_number_id=asset_number.id if asset_number else None,
                location_id=location.id if location else None,
                site_location_id=site_location.id if site_location else None
            )
            session.add(position)
            session.commit()
            logger.info(f"Created new Position with ID {position.id}")

        # Create association
        problem_position_association = ProblemPositionAssociation(
            problem_id=new_problem.id,
            position_id=position.id
        )
        session.add(problem_position_association)
        session.commit()
        logger.info(f"Created association between Problem ID {new_problem.id} and Position ID {position.id}")

        return jsonify({'success': True, 'message': 'Problem created successfully!', 'problem_id': new_problem.id}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error during problem creation: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while creating the problem.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error during problem creation: {e}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshoot_new_entry_bp.route('/get_equipment_groups', methods=['GET'])
@with_request_id
def get_equipment_groups():
    area_id = request.args.get('area_id')
    if not area_id:
        return jsonify({"error": "area_id parameter is required"}), 400

    try:
        session = db_config.get_main_session()
        equipment_groups = session.query(EquipmentGroup).filter_by(area_id=area_id).all()
        data = [{'id': eg.id, 'name': eg.name} for eg in equipment_groups]
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error fetching equipment groups: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        session.close()

@pst_troubleshoot_new_entry_bp.route('/get_models', methods=['GET'])
@with_request_id
def get_models():
    equipment_group_id = request.args.get('equipment_group_id')
    if not equipment_group_id:
        return jsonify({"error": "equipment_group_id parameter is required"}), 400

    try:
        session = db_config.get_main_session()
        models = session.query(Model).filter_by(equipment_group_id=equipment_group_id).all()
        data = [{'id': m.id, 'name': m.name} for m in models]
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error fetching models: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        session.close()

@pst_troubleshoot_new_entry_bp.route('/get_asset_numbers', methods=['GET'])
@with_request_id
def get_asset_numbers():
    model_id = request.args.get('model_id')
    if not model_id:
        return jsonify({"error": "model_id parameter is required"}), 400

    try:
        session = db_config.get_main_session()
        asset_numbers = session.query(AssetNumber).filter_by(model_id=model_id).all()
        data = [{'id': an.id, 'number': an.number} for an in asset_numbers]
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error fetching asset numbers: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        session.close()

@pst_troubleshoot_new_entry_bp.route('/get_locations', methods=['GET'])
@with_request_id
def get_locations():
    model_id = request.args.get('model_id')
    if not model_id:
        return jsonify({"error": "model_id parameter is required"}), 400

    try:
        session = db_config.get_main_session()
        locations = session.query(Location).filter_by(model_id=model_id).all()
        data = [{'id': loc.id, 'name': loc.name} for loc in locations]
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error fetching locations: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        session.close()

@pst_troubleshoot_new_entry_bp.route('/get_site_locations', methods=['GET'])
@with_request_id
def get_site_locations():
    """
    Fetch all Site Locations with detailed logging for debugging.
    """
    session = None
    try:
        # Create a new session
        session = db_config.get_main_session()
        logger.info("Database session created successfully for /get_site_locations.")

        # Debug log: Starting to fetch site locations
        logger.info("Starting to fetch site locations from the database.")

        # Query all Site Locations
        site_locations = session.query(SiteLocation).all()

        # Debug log: Number of entries retrieved
        if site_locations:
            logger.info(f"Fetched {len(site_locations)} site locations from the database.")
        else:
            logger.warning("No site locations found in the database.")

        # Prepare data for JSON response
        data = [{'id': loc.id, 'title': loc.title, 'room_number': loc.room_number} for loc in site_locations]

        # Debug log: Data to be returned in JSON response
        logger.debug(f"Site Location data prepared for response: {data}")

        # Return JSON response
        return jsonify(data), 200

    except Exception as e:
        # Log detailed error information
        logger.error(f"Error fetching site locations: {e}", exc_info=True)

        # Return specific error message
        return jsonify({"error": "Failed to fetch site locations from database."}), 500

    finally:
        # Ensure the session is always closed
        if session:
            session.close()
            logger.info("Database session closed after fetching site locations.")

@pst_troubleshoot_new_entry_bp.route('/get_positions', methods=['GET'])
@with_request_id
def get_positions():
    try:
        # Get filter parameters from the request
        site_location_id = request.args.get('site_location_id')
        area_id = request.args.get('area_id')
        equipment_group_id = request.args.get('equipment_group_id')
        model_id = request.args.get('model_id')
        asset_number_id = request.args.get('asset_number_id')
        location_id = request.args.get('location_id')

        logger.info(
            f"Received GET request with filters: site_location_id={site_location_id}, area_id={area_id}, equipment_group_id={equipment_group_id}, model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}")

        # Build the query based on filters
        session = db_config.get_main_session()
        query = session.query(Position)

        if site_location_id:
            query = query.filter(Position.site_location_id == site_location_id)
        if area_id:
            query = query.filter(Position.area_id == area_id)
        if equipment_group_id:
            query = query.filter(Position.equipment_group_id == equipment_group_id)
        if model_id:
            query = query.filter(Position.model_id == model_id)
        if asset_number_id:
            query = query.filter(Position.asset_number_id == asset_number_id)
        if location_id:
            query = query.filter(Position.location_id == location_id)

        positions = query.all()
        logger.info(f"Found {len(positions)} positions matching the filters.")

        if not positions:
            logger.warning("No positions found with the given filters.")
            return jsonify({"message": "No positions found"}), 404

        # Prepare the response data
        result_data = []

        for position in positions:
            position_data = {
                'position_id': position.id,
                'area': {
                    'id': position.area.id if position.area else None,
                    'name': position.area.name if position.area else None,
                    'description': position.area.description if position.area else None
                },
                'equipment_group': {
                    'id': position.equipment_group.id if position.equipment_group else None,
                    'name': position.equipment_group.name if position.equipment_group else None
                },
                'model': {
                    'id': position.model.id if position.model else None,
                    'name': position.model.name if position.model else None,
                    'description': position.model.description if position.model else None
                },
                'asset_number': {
                    'id': position.asset_number.id if position.asset_number else None,
                    'number': position.asset_number.number if position.asset_number else None,
                    'description': position.asset_number.description if position.asset_number else None
                },
                'location': {
                    'id': position.location.id if position.location else None,
                    'name': position.location.name if position.location else None,
                    'description': position.location.description if position.location else None
                },
                'site_location': {
                    'id': position.site_location.id if position.site_location else None,
                    'title': position.site_location.title if position.site_location else None,
                    'room_number': position.site_location.room_number if position.site_location else None
                },
                'parts': [],
                'documents': [],
                'drawings': [],
                'images': []
            }

            logger.info(f"Processing Position ID: {position.id}")

            # Fetch parts
            parts_associations = session.query(PartsPositionImageAssociation).filter_by(
                position_id=position.id).all()
            part_ids = [assoc.part_id for assoc in parts_associations]
            if part_ids:
                parts = session.query(Part).filter(Part.id.in_(part_ids)).all()
                for part in parts:
                    position_data['parts'].append({
                        'part_id': part.id,
                        'part_number': part.part_number,
                        'name': part.name
                    })

            # Fetch drawings
            drawing_associations = session.query(DrawingPositionAssociation).filter_by(position_id=position.id).all()
            drawing_ids = [assoc.drawing_id for assoc in drawing_associations]
            if drawing_ids:
                drawings = session.query(Drawing).filter(Drawing.id.in_(drawing_ids)).all()
                for drawing in drawings:
                    position_data['drawings'].append({
                        'drawing_id': drawing.id,
                        'drw_name': drawing.drw_name,
                        'drw_number': drawing.drw_number
                        # Include other fields if necessary
                    })

            # Fetch documents
            document_associations = session.query(CompletedDocumentPositionAssociation).filter_by(
                position_id=position.id).all()
            document_ids = [assoc.complete_document_id for assoc in document_associations]
            if document_ids:
                documents = session.query(CompleteDocument).filter(CompleteDocument.id.in_(document_ids)).all()
                for doc in documents:
                    position_data['documents'].append({
                        'document_id': doc.id,
                        'title': doc.title,
                        'rev': doc.rev,
                        'file_path': doc.file_path,
                        'content': doc.content  # Include if necessary
                    })
                logger.info(f"Added {len(documents)} documents to Position ID {position.id}")

            # Fetch images
            image_associations = session.query(ImagePositionAssociation).filter_by(position_id=position.id).all()
            image_ids = [assoc.image_id for assoc in image_associations]
            if image_ids:
                images = session.query(Image).filter(Image.id.in_(image_ids)).all()
                for image in images:
                    position_data['images'].append({
                        'image_id': image.id,
                        'title': image.title,
                        'description': image.description,
                        'file_path': image.file_path,
                    })
                logger.info(f"Added {len(images)} images to Position ID {position.id}")

            # Append the position data to the result list
            result_data.append(position_data)

        logger.info(f"Returning data for {len(result_data)} positions.")
        return jsonify(result_data), 200

    except Exception as e:
        logger.error(f"Error in /get_positions: {e}", exc_info=True)
        return jsonify({"message": "Error occurred during position search", "error": str(e)}), 500

    finally:
        session.close()
        logger.info("Database session closed for /get_positions.")

@pst_troubleshoot_new_entry_bp.route('/create_equipment_group', methods=['POST'])
@with_request_id
def create_equipment_group():
    """
    Handle the creation of a new equipment group using the class method.
    """
    session = db_config.get_main_session()
    try:
        # Extract form data
        name = request.form.get('name')
        area_id = request.form.get('area_id')
        description = request.form.get('description')

        # Validate required fields
        if not all([name, area_id]):
            logger.warning("Missing required fields for equipment group creation")
            return jsonify({'success': False, 'message': 'Name and Area ID are required.'}), 400

        # Log the incoming request data
        logger.info(f"Creating equipment group - Name: {name}, Area ID: {area_id}, Description: {description}")

        # Use the class method to add the equipment group
        new_equipment_group = EquipmentGroup.add_equipment_group(
            session=session,
            name=name,
            area_id=area_id,
            description=description
        )

        logger.info(f"Created new Equipment Group: {name} with ID {new_equipment_group.id}")

        # Return success response with the new equipment group details
        return jsonify({
            'success': True,
            'message': 'Equipment Group created successfully!',
            'equipment_group': {
                'id': new_equipment_group.id,
                'name': new_equipment_group.name,
                'area_id': new_equipment_group.area_id,
                'description': new_equipment_group.description
            }
        }), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error during equipment group creation: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while creating the equipment group.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error during equipment group creation: {e}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshoot_new_entry_bp.route('/create_model', methods=['POST'])
@with_request_id
def create_model():
    """
    Handle the creation of a new model using the class method.
    """
    session = db_config.get_main_session()
    try:
        # Extract form data
        name = request.form.get('name')
        equipment_group_id = request.form.get('equipment_group_id')
        description = request.form.get('description')

        # Validate required fields
        if not all([name, equipment_group_id]):
            logger.warning("Missing required fields for model creation")
            return jsonify({'success': False, 'message': 'Name and Equipment Group ID are required.'}), 400

        # Log the incoming request data
        logger.info(
            f"Creating model - Name: {name}, Equipment Group ID: {equipment_group_id}, Description: {description}")

        # Use the class method to add the model
        new_model = Model.add_model(
            session=session,
            name=name,
            equipment_group_id=equipment_group_id,
            description=description
        )

        logger.info(f"Created new Model: {name} with ID {new_model.id}")

        # Return success response with the new model details
        return jsonify({
            'success': True,
            'message': 'Model created successfully!',
            'model': {
                'id': new_model.id,
                'name': new_model.name,
                'equipment_group_id': new_model.equipment_group_id,
                'description': new_model.description
            }
        }), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error during model creation: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while creating the model.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error during model creation: {e}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshoot_new_entry_bp.route('/create_asset_number', methods=['POST'])
@with_request_id
def create_asset_number():
    """
    Handle the creation of a new asset number using the class method.
    """
    session = db_config.get_main_session()
    try:
        # Extract form data
        number = request.form.get('number')
        model_id = request.form.get('model_id')
        description = request.form.get('description')

        # Validate required fields
        if not all([number, model_id]):
            logger.warning("Missing required fields for asset number creation")
            return jsonify({'success': False, 'message': 'Number and Model ID are required.'}), 400

        # Log the incoming request data
        logger.info(f"Creating asset number - Number: {number}, Model ID: {model_id}, Description: {description}")

        # Use the class method to add the asset number
        new_asset_number = AssetNumber.add_asset_number(
            session=session,
            number=number,
            model_id=model_id,
            description=description
        )

        # Return success response with the new asset number details
        return jsonify({
            'success': True,
            'message': 'Asset Number created successfully!',
            'asset_number': {
                'id': new_asset_number.id,
                'number': new_asset_number.number,
                'model_id': new_asset_number.model_id,
                'description': new_asset_number.description
            }
        }), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error during asset number creation: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while creating the asset number.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error during asset number creation: {e}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshoot_new_entry_bp.route('/create_location', methods=['POST'])
@with_request_id
def create_location():
    """
    Handle the creation of a new location using the class method.
    """
    session = db_config.get_main_session()
    try:
        # Extract form data
        name = request.form.get('name')
        model_id = request.form.get('model_id')
        description = request.form.get('description')

        # Validate required fields
        if not all([name, model_id]):
            logger.warning("Missing required fields for location creation")
            return jsonify({'success': False, 'message': 'Name and Model ID are required.'}), 400

        # Log the incoming request data
        logger.info(f"Creating location - Name: {name}, Model ID: {model_id}, Description: {description}")

        # Use the class method to add the location
        new_location = Location.add_location(
            session=session,
            name=name,
            model_id=model_id,
            description=description
        )

        # Return success response with the new location details
        return jsonify({
            'success': True,
            'message': 'Location created successfully!',
            'location': {
                'id': new_location.id,
                'name': new_location.name,
                'model_id': new_location.model_id,
                'description': new_location.description
            }
        }), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error during location creation: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while creating the location.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error during location creation: {e}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshoot_new_entry_bp.route('/create_site_location', methods=['POST'])
@with_request_id
def create_site_location():
    """
    Handle the creation of a new site location using the class method.
    """
    session = db_config.get_main_session()
    try:
        # Extract form data
        title = request.form.get('title')
        room_number = request.form.get('room_number')
        site_area = request.form.get('site_area', 'Default Area')  # Provide a default if not specified

        # Validate required fields
        if not all([title, room_number]):
            logger.warning("Missing required fields for site location creation")
            return jsonify({'success': False, 'message': 'Title and Room Number are required.'}), 400

        # Log the incoming request data
        logger.info(f"Creating site location - Title: {title}, Room Number: {room_number}, Site Area: {site_area}")

        # Use the class method to add the site location
        new_site_location = SiteLocation.add_site_location(
            session=session,
            title=title,
            room_number=room_number,
            site_area=site_area
        )

        # Return success response with the new site location details
        return jsonify({
            'success': True,
            'message': 'Site Location created successfully!',
            'site_location': {
                'id': new_site_location.id,
                'title': new_site_location.title,
                'room_number': new_site_location.room_number,
                'site_area': new_site_location.site_area
            }
        }), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error during site location creation: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while creating the site location.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error during site location creation: {e}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred.'}), 500
    finally:
        session.close()