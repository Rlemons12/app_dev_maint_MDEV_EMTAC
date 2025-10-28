from flask import Blueprint, request, jsonify
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import Problem, ProblemPositionAssociation, Position
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))

# Define the blueprint
get_troubleshooting_guide_edit_data_bp = Blueprint('get_troubleshooting_guide_edit_data_bp', __name__)

@get_troubleshooting_guide_edit_data_bp.route('/get_troubleshooting_guide_edit_data', methods=['GET'])
def get_troubleshooting_guide_edit_data():
    logger.info("get_troubleshooting_guide_edit_data route accessed")

    # Get query data from the request
    problem_name = request.args.get('query', '')
    area_id = request.args.get('area')
    equipment_group_id = request.args.get('equipment_group')
    model_id = request.args.get('model')
    asset_number_id = request.args.get('asset_number')
    location_id = request.args.get('location')

    session = Session()
    try:
        # Build filters for the position search based on the dropdown selections
        filters = []
        if area_id:
            filters.append(Position.area_id == area_id)
            logger.info("Area [{area_id}]")
        if equipment_group_id:
            filters.append(Position.equipment_group_id == equipment_group_id)
            logger.info("Equipment Group ID [{equipment_group_id}]")
        if model_id:
            filters.append(Position.model_id == model_id)
            logger.info("Model ID [{model_id}]")
        if asset_number_id:
            filters.append(Position.asset_number_id == asset_number_id)
            logger.info("Asset Number [{asset_number_id}]")
        if location_id:
            filters.append(Position.location_id == location_id)
            logger.info("Location ID [{location_id}]")

        # Query positions based on filters (if filters exist)
        position_ids = []
        if filters:
            logger.info("Searching for positions with filters.")
            position_query = session.query(Position).filter(and_(*filters)).all()
            position_ids = [position.id for position in position_query]
            logger.info(f"Found matching Position IDs: {position_ids}")

        # If no filters, select all positions
        if not position_ids:
            logger.info("No filters provided, searching all positions.")
            position_query = session.query(Position).all()
            position_ids = [position.id for position in position_query]

        # Find problems based on position IDs in ProblemPositionAssociation
        problem_ids = []
        if position_ids:
            problem_association_query = session.query(ProblemPositionAssociation).filter(
                ProblemPositionAssociation.position_id.in_(position_ids)
            ).all()
            problem_ids = [association.problem_id for association in problem_association_query]

        # Query problems based on problem_ids and the partial title match if provided
        problem_query = session.query(Problem).filter(Problem.id.in_(problem_ids))
        if problem_name:
            logger.info(f"Searching for problems with a partial title match: {problem_name}")
            problem_query = problem_query.filter(Problem.name.ilike(f"%{problem_name}%"))

        # Retrieve the problems along with their associated solutions and relations
        problems = problem_query.all()

        # Prepare the response with details from associated tables (Solution, Images, Documents, etc.)
        results = []
        for problem in problems:
            problem_data = {
                'id': problem.id,
                'name': problem.name,
                'description': problem.description,
                'solution': [{
                    'id': solution.id,
                    'description': solution.description,
                    'parts': [{'id': assoc.part_id} for assoc in solution.part_solution],
                    'drawings': [{'id': assoc.drawing_id} for assoc in solution.drawing_solution]
                } for solution in problem.solution],
                'positions': [{'id': assoc.position_id} for assoc in problem.problem_position],
                'images': [{'id': assoc.image_id} for assoc in problem.image_problem],
                'documents': [{'id': assoc.complete_document_id} for assoc in problem.complete_document_problem],
                'drawings': [{'id': assoc.drawing_id} for assoc in problem.drawing_problem],
                'parts': [{'id': assoc.part_id} for assoc in problem.part_problem]
            }
            results.append(problem_data)

        return jsonify({'problems': results})

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        session.close()

