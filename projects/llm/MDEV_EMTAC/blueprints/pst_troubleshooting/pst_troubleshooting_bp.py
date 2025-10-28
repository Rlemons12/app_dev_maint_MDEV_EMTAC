# blueprints/pst_troubleshooting_bp.py

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash, abort
import logging
from modules.configuration.config_env import DatabaseConfig  # Import your DatabaseConfig class
from modules.emtacdb.emtacdb_fts import (
    Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation, Position,
    Part, Drawing, CompleteDocument, Image, PartsPositionImageAssociation,
    DrawingPositionAssociation, CompletedDocumentPositionAssociation, ImagePositionAssociation,
    Problem, Solution  # Ensure all models are imported
)

# Define the blueprint once
pst_troubleshooting_bp = Blueprint('pst_troubleshooting_bp', __name__)
logger = logging.getLogger(__name__)

# Initialize DatabaseConfig
db_config = DatabaseConfig()


def allowed_file(filename, extensions={'png', 'jpg', 'jpeg', 'gif', 'pdf'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

# Helper Routes without redundant prefix

@pst_troubleshooting_bp.route('/get_equipment_groups', methods=['GET'])
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

@pst_troubleshooting_bp.route('/get_models', methods=['GET'])
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

@pst_troubleshooting_bp.route('/get_asset_numbers', methods=['GET'])
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

@pst_troubleshooting_bp.route('/get_locations', methods=['GET'])
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

@pst_troubleshooting_bp.route('/get_site_locations', methods=['GET'])
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

@pst_troubleshooting_bp.route('/get_positions', methods=['GET'])
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

@pst_troubleshooting_bp.route('/', methods=['GET', 'POST'])
@pst_troubleshooting_bp.route('/<int:problem_id>', methods=['GET', 'POST'])
def pst_troubleshooting_page(problem_id=None):
    session = db_config.get_main_session()
    try:
        if problem_id:
            # Fetch the existing problem from the database
            problem = session.query(Problem).filter_by(id=problem_id).first()
            if not problem:
                abort(404, description="Problem not found")
        else:
            # Initialize an empty problem for creation
            problem = None

        # Fetch additional data needed for the template
        areas = session.query(Area).all()  # Retrieve areas from the database
        parts = session.query(Part).all()  # Retrieve parts
        drawings = session.query(Drawing).all()  # Retrieve drawings

        if request.method == 'POST':
            try:
                # Extract form data
                problem_name = request.form.get('problem_name')
                problem_description = request.form.get('problem_description')
                area_id = request.form.get('area_id')
                equipment_group_id = request.form.get('equipment_group_id')
                model_id = request.form.get('model_id')
                asset_number_id = request.form.get('asset_number_id')
                asset_number_input = request.form.get('asset_number_input')
                location_id = request.form.get('location_id')
                location_input = request.form.get('location_input')
                site_location_id = request.form.get('site_location_id')
                part_numbers = request.form.getlist('parts[]')  # Adjust if different
                new_solution_name = request.form.get('new_solution_name')

                # Handle manual input for Asset Number
                if not asset_number_id and asset_number_input:
                    new_asset = AssetNumber(number=asset_number_input, model_id=model_id)
                    session.add(new_asset)
                    session.commit()
                    asset_number_id = new_asset.id

                # Handle manual input for Location
                if not location_id and location_input:
                    new_location = Location(name=location_input, model_id=model_id)
                    session.add(new_location)
                    session.commit()
                    location_id = new_location.id

                if problem:
                    # Update existing problem
                    problem.name = problem_name
                    problem.description = problem_description
                    problem.area_id = area_id
                    problem.equipment_group_id = equipment_group_id
                    problem.model_id = model_id
                    problem.asset_number_id = asset_number_id
                    problem.location_id = location_id
                    problem.site_location_id = site_location_id
                    # Update parts if necessary
                    problem.parts = part_numbers  # Assuming a relationship exists

                    # Handle new solution
                    if new_solution_name:
                        new_solution = Solution(name=new_solution_name, problem_id=problem.id)
                        session.add(new_solution)

                    session.commit()
                    flash('Problem updated successfully!', 'success')
                else:
                    # Create new problem
                    new_problem = Problem(
                        name=problem_name,
                        description=problem_description,
                        area_id=area_id,
                        equipment_group_id=equipment_group_id,
                        model_id=model_id,
                        asset_number_id=asset_number_id,
                        location_id=location_id,
                        site_location_id=site_location_id
                    )
                    session.add(new_problem)
                    session.commit()

                    # Handle new solution
                    if new_solution_name:
                        new_solution = Solution(name=new_solution_name, problem_id=new_problem.id)
                        session.add(new_solution)
                        session.commit()

                    flash('Problem created successfully!', 'success')
                    return redirect(
                        url_for('pst_troubleshooting_bp.pst_troubleshooting_page', problem_id=new_problem.id))

                return redirect(url_for('pst_troubleshooting_bp.pst_troubleshooting_page', problem_id=problem_id))
            except Exception as e:
                session.rollback()
                logger.error(f"Error processing PST Troubleshooting form: {e}", exc_info=True)
                flash('An error occurred while processing your request.', 'danger')
                return redirect(url_for('pst_troubleshooting_bp.pst_troubleshooting_page', problem_id=problem_id))
        else:
            # Handle GET request
            return render_template(
                'pst_troubleshooting/pst_troubleshooting.html',
                problem=problem,
                areas=areas,
                parts=parts,
                drawings=drawings
            )
    except Exception as e:
        logger.error(f"Error loading the troubleshooting page: {e}", exc_info=True)
        flash('Error loading the form.', 'danger')
        return redirect(url_for('pst_troubleshooting_bp.pst_troubleshooting_page'))
    finally:
        session.close()

