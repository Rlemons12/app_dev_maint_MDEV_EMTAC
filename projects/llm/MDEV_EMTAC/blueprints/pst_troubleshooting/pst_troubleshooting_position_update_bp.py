from flask import Blueprint, request, jsonify
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import with_request_id
from modules.emtacdb.emtacdb_fts import (Position, ProblemPositionAssociation,
                                         Problem, EquipmentGroup, Model, AssetNumber, Location, SiteLocation,
                                         Part, Solution
                                         )
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


pst_troubleshooting_position_update_bp = Blueprint('pst_troubleshooting_position_update_bp', __name__)

# Initialize Database Config
db_config = DatabaseConfig()


@pst_troubleshooting_position_update_bp.route('/get_problem/<int:problem_id>', methods=['GET'])
@with_request_id
def get_problem(problem_id):
    """
    Retrieve a single problem's details for updating.
    """
    session = db_config.get_main_session()
    try:
        problem = session.query(Problem).options(
            joinedload(Problem.area),
            joinedload(Problem.equipment_group),
            joinedload(Problem.model),
            joinedload(Problem.asset_number),
            joinedload(Problem.location),
            joinedload(Problem.site_location),
            joinedload(Problem.parts),
            joinedload(Problem.solutions)
        ).filter_by(id=problem_id).first()

        if not problem:
            return jsonify({'error': 'Problem not found.'}), 404

        problem_data = {
            'id': problem.id,
            'name': problem.name,
            'description': problem.description,
            'area_id': problem.area_id,
            'equipment_group_id': problem.equipment_group_id,
            'model_id': problem.model_id,
            'asset_number': problem.asset_number.number if problem.asset_number else '',
            'location': problem.location.name if problem.location else '',
            'site_location_id': problem.site_location_id,
            'parts': [part.id for part in problem.parts],
            'solutions': [solution.name for solution in problem.solutions]
        }

        logger.info(f"Retrieved Problem ID: {problem.id}")

        return jsonify(problem_data), 200

    except SQLAlchemyError as e:
        logger.error(f"Database error during problem retrieval: {e}")
        return jsonify({'error': 'An error occurred while retrieving the problem.'}), 500
    except Exception as e:
        logger.error(f"Unexpected error during problem retrieval: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshooting_position_update_bp.route('/update_problem', methods=['POST'])
@with_request_id
def update_problem():
    """
    Update the details of an existing problem.
    """
    session = db_config.get_main_session()
    try:
        # Extract form data
        problem_id = request.form.get('problem_id')
        problem_name = request.form.get('problem_name')
        problem_description = request.form.get('problem_description')
        area_id = request.form.get('area_id')
        equipment_group_id = request.form.get('equipment_group_id')
        model_id = request.form.get('model_id')

        # Get IDs directly from the form
        asset_number_id = request.form.get('asset_number_id')
        location_id = request.form.get('location_id')

        # Fallback to old way if IDs aren't provided
        asset_number_input = request.form.get('asset_number')
        location_input = request.form.get('location')

        site_location_id = request.form.get('site_location_id')

        # Check if we have the required data (either IDs or input strings)
        if not all([problem_id, problem_name, problem_description, area_id, equipment_group_id, model_id,
                    site_location_id]) or \
                not ((asset_number_id or asset_number_input) and (location_id or location_input)):
            return jsonify({'success': False, 'message': 'All fields are required.'}), 400

        # Retrieve the problem
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if not problem:
            return jsonify({'success': False, 'message': 'Problem not found.'}), 404

        # Update fields
        problem.name = problem_name
        problem.description = problem_description
        problem.area_id = area_id
        problem.equipment_group_id = equipment_group_id
        problem.model_id = model_id

        # Handle Asset Number: First try to use ID directly, then fall back to string input
        if asset_number_id:
            # Use the ID directly
            logger.info(f"Using provided asset_number_id: {asset_number_id}")
            problem.asset_number_id = asset_number_id
        elif asset_number_input:
            # Fall back to the old way - look up by number or create new
            asset_number = session.query(AssetNumber).filter_by(number=asset_number_input).first()
            if not asset_number:
                asset_number = AssetNumber(number=asset_number_input, model_id=model_id)
                session.add(asset_number)
                session.commit()
                logger.info(f"Created new Asset Number: {asset_number_input}")
            problem.asset_number_id = asset_number.id
            logger.info(f"Looked up asset_number_id: {asset_number.id} for number: {asset_number_input}")

        # Handle Location: First try to use ID directly, then fall back to string input
        if location_id:
            # Use the ID directly
            logger.info(f"Using provided location_id: {location_id}")
            problem.location_id = location_id
        elif location_input:
            # Fall back to the old way - look up by name or create new
            location = session.query(Location).filter_by(name=location_input).first()
            if not location:
                location = Location(name=location_input, model_id=model_id)
                session.add(location)
                session.commit()
                logger.info(f"Created new Location: {location_input}")
            problem.location_id = location.id
            logger.info(f"Looked up location_id: {location.id} for name: {location_input}")

        # Handle Site Location: If 'new', create a new Site Location
        if site_location_id == 'new':
            new_site_location_title = request.form.get('new_siteLocation_title')
            new_site_location_room_number = request.form.get('new_siteLocation_room_number')

            if not all([new_site_location_title, new_site_location_room_number]):
                return jsonify(
                    {'success': False, 'message': 'New Site Location title and room number are required.'}), 400

            site_location = SiteLocation(
                title=new_site_location_title,
                room_number=new_site_location_room_number
            )
            session.add(site_location)
            session.commit()
            logger.info(f"Created new Site Location: {new_site_location_title}")
            problem.site_location_id = site_location.id
        else:
            site_location = session.query(SiteLocation).filter_by(id=site_location_id).first()
            if not site_location:
                return jsonify({'success': False, 'message': 'Selected Site Location does not exist.'}), 400
            problem.site_location_id = site_location.id

        # Optionally, handle parts association
        parts_ids = request.form.getlist('parts[]')  # Assuming multiple parts can be selected
        if parts_ids:
            # Clear existing associations
            problem.parts = []
            # Re-associate parts
            for part_id in parts_ids:
                part = session.query(Part).filter_by(id=part_id).first()
                if part:
                    problem.parts.append(part)

        session.commit()
        logger.info(f"Updated Problem ID: {problem.id}")

        # Handle new Solution if provided
        new_solution_name = request.form.get('new_solution_name')
        if new_solution_name:
            new_solution = Solution(name=new_solution_name, problem_id=problem.id)
            session.add(new_solution)
            session.commit()
            logger.info(f"Added new Solution: {new_solution_name} to Problem ID: {problem.id}")

        return jsonify({'success': True, 'message': 'Problem updated successfully!', 'problem_id': problem.id}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error during problem update: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while updating the problem.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error during problem update: {e}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshooting_position_update_bp.route('/search_problems', methods=['GET'])
@with_request_id
def search_problems():
    """
    Search for problems by querying the Position entity with provided criteria and linking with Problem.
    """
    session = db_config.get_main_session()
    try:
        # Extract search parameters
        area_id = request.args.get('area_id')
        equipment_group_id = request.args.get('equipment_group_id')
        model_id = request.args.get('model_id')
        asset_number = request.args.get('asset_number')
        location = request.args.get('location')
        site_location_id = request.args.get('site_location_id')

        # Build the initial query on Position
        query = session.query(Position)
        if area_id:
            query = query.filter_by(area_id=area_id)
        if equipment_group_id:
            query = query.filter_by(equipment_group_id=equipment_group_id)
        if model_id:
            query = query.filter_by(model_id=model_id)
        if asset_number:
            query = query.filter(Position.asset_number.has(number=asset_number))
        if location:
            query = query.filter(Position.location.has(name=location))
        if site_location_id:
            query = query.filter_by(site_location_id=site_location_id)

        # Retrieve Position IDs that match the criteria
        positions = query.all()
        position_ids = [position.id for position in positions]

        # Use ProblemPositionAssociation to find related Problems
        problem_associations = session.query(ProblemPositionAssociation).filter(
            ProblemPositionAssociation.position_id.in_(position_ids)
        ).all()

        # Collect unique Problems related to the Position IDs
        problems = []
        for association in problem_associations:
            problem = session.query(Problem).filter_by(id=association.problem_id).first()
            if problem and problem not in problems:
                problems.append(problem)

        # Prepare data for JSON response
        problems_data = [{
            'id': problem.id,
            'name': problem.name,
            'description': problem.description
        } for problem in problems]

        logger.info(f"Found {len(problems_data)} problems matching criteria.")
        return jsonify(problems_data), 200

    except SQLAlchemyError as e:
        logger.error(f"Database error during problem search: {e}")
        return jsonify({'error': 'An error occurred while searching for problems.'}), 500
    except Exception as e:
        logger.error(f"Unexpected error during problem search: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshooting_position_update_bp.route('/get_problem_details/<int:problem_id>', methods=['GET'])
@with_request_id
def get_problem_details(problem_id):
    """
    Fetch the details of a problem along with its associated position information.
    """
    session = db_config.get_main_session()
    try:
        # Fetch the problem with the specified ID
        problem = session.query(Problem).filter_by(id=problem_id).first()

        if not problem:
            logger.error(f"Problem with ID {problem_id} not found.")
            return jsonify({'error': 'Problem not found.'}), 404

        # Find the associated position using ProblemPositionAssociation
        association = session.query(ProblemPositionAssociation).filter_by(problem_id=problem_id).first()

        if not association:
            logger.error(f"No position associated with problem ID {problem_id}.")
            return jsonify({'error': 'Position not found for this problem.'}), 404

        # Fetch the position details with eager loading
        position = session.query(Position).options(
            joinedload(Position.area),
            joinedload(Position.equipment_group),
            joinedload(Position.model),
            joinedload(Position.asset_number),
            joinedload(Position.location),
            joinedload(Position.site_location)
        ).filter_by(id=association.position_id).first()

        if not position:
            logger.error(f"Position with ID {association.position_id} not found.")
            return jsonify({'error': 'Position not found.'}), 404

        # Prepare the JSON response with both problem and position details
        response_data = {
            'problem': {
                'id': problem.id,
                'name': problem.name,
                'description': problem.description
            },
            'position': {
                'area_id': position.area_id,
                'equipment_group_id': position.equipment_group_id,
                'model_id': position.model_id,
                'asset_number': position.asset_number.number if position.asset_number else None,
                'asset_number_id': position.asset_number_id,  # Add this line
                'location': position.location.name if position.location else None,
                'location_id': position.location_id,  # Add this line
                'site_location_id': position.site_location_id
            }
        }

        logger.info(f"Fetched details for problem ID {problem_id} with associated position.")
        return jsonify(response_data), 200

    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching problem details: {e}")
        return jsonify({'error': 'An error occurred while fetching problem details.'}), 500
    except Exception as e:
        logger.error(f"Unexpected error while fetching problem details: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshooting_position_update_bp.route('/get_equipment_groups', methods=['GET'])
@with_request_id
def get_equipment_groups():
    session = db_config.get_main_session()
    area_id = request.args.get('area_id')
    equipment_groups = session.query(EquipmentGroup).filter_by(area_id=area_id).all()
    data = [{'id': eg.id, 'name': eg.name} for eg in equipment_groups]
    return jsonify(data)


@pst_troubleshooting_position_update_bp.route('/get_models', methods=['GET'])
@with_request_id
def get_models():
    session = db_config.get_main_session()
    equipment_group_id = request.args.get('equipment_group_id')
    models = session.query(Model).filter_by(equipment_group_id=equipment_group_id).all()
    data = [{'id': model.id, 'name': model.name} for model in models]
    return jsonify(data)

@pst_troubleshooting_position_update_bp.route('/get_asset_numbers', methods=['GET'])
@with_request_id
def get_asset_numbers():
    session = db_config.get_main_session()
    model_id = request.args.get('model_id')
    asset_numbers = session.query(AssetNumber).filter_by(model_id=model_id).all()
    data = [{'id': asset.id, 'number': asset.number} for asset in asset_numbers]
    return jsonify(data)

@pst_troubleshooting_position_update_bp.route('/get_locations', methods=['GET'])
@with_request_id
def get_locations():
    session = db_config.get_main_session()
    model_id = request.args.get('model_id')
    locations = session.query(Location).filter_by(model_id=model_id).all()
    data = [{'id': location.id, 'name': location.name} for location in locations]
    return jsonify(data)

@pst_troubleshooting_position_update_bp.route('/get_site_locations', methods=['GET'])
@with_request_id
def get_site_locations():
    session = db_config.get_main_session()
    model_id = request.args.get('model_id')
    asset_number_id = request.args.get('asset_number_id')
    location_id = request.args.get('location_id')
    area_id = request.args.get('area_id')  # New parameter for area
    equipment_group_id = request.args.get('equipment_group_id')  # New parameter for equipment group

    # Log the incoming request parameters
    logger.info(f"Received request to /get_site_locations with model_id: {model_id}, "
                f"asset_number_id: {asset_number_id}, location_id: {location_id}, "
                f"area_id: {area_id}, equipment_group_id: {equipment_group_id}")

    try:
        # Filter positions by all the provided filters
        positions = session.query(Position).filter_by(
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            area_id=area_id,
            equipment_group_id=equipment_group_id
        ).all()

        # Log the number of positions found
        logger.info(f"Found {len(positions)} positions matching the filters.")

        # Extract site locations from the filtered positions
        site_locations = [
            {'id': pos.site_location.id, 'title': pos.site_location.title, 'room_number': pos.site_location.room_number}
            for pos in positions if pos.site_location
        ]

        # Log the number of site locations found
        logger.info(f"Extracted {len(site_locations)} site locations.")

        # Add a default "New Site Location" option to the list
        site_locations.append({'id': 'new', 'title': 'New Site Location', 'room_number': ''})

        return jsonify(site_locations)
    except Exception as e:
        logger.error(f"Error fetching site locations: {e}")
        return jsonify({"error": "An error occurred while fetching site locations"}), 500

@pst_troubleshooting_position_update_bp.route('/search_site_locations', methods=['GET'])
@with_request_id
def search_site_locations():
    search_term = request.args.get('search', '')

    # Log the search term
    logger.info(f"Received request to /search_site_locations with search term: {search_term}")

    if search_term:
        try:
            # Search for site locations by the title (room number can also be included if needed)
            site_locations = session.query(SiteLocation).filter(SiteLocation.title.ilike(f'%{search_term}%')).all()

            # Log the number of search results found
            logger.info(f"Found {len(site_locations)} site locations matching the search term.")

            results = [
                {'id': location.id, 'title': location.title, 'room_number': location.room_number}
                for location in site_locations
            ]

            return jsonify(results)
        except Exception as e:
            logger.error(f"Error during site location search: {e}")
            return jsonify({"error": "An error occurred during site location search"}), 500

    logger.info("No search term provided, returning an empty list.")
    return jsonify([])  # Return an empty list if no search term is provided