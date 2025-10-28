import logging
from flask import Blueprint, request, jsonify, render_template
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine
from modules.emtacdb.emtacdb_fts import (
    Problem, Task, Position, Part, Drawing,
    ImageProblemAssociation, ImageTaskAssociation,
    CompleteDocumentProblemAssociation, CompleteDocumentTaskAssociation,
    PartProblemAssociation, PartTaskAssociation, DrawingProblemAssociation,
    ProblemPositionAssociation, PartsPositionImageAssociation,
    DrawingPositionAssociation
)
from modules.configuration.config import DATABASE_URL

# Create the blueprint
trouble_shooting_guide_bp = Blueprint('trouble_shooting_guide_bp', __name__)

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@trouble_shooting_guide_bp.route('/update_problem_solution', methods=['POST'])
def update_problem_solution():
    logger.info("Entered update_problem_solution route")

    # Get form data
    problem_name = request.form.get('problem_name')
    tsg_area_id = request.form.get('tsg_area')
    tsg_equipment_group_id = request.form.get('tsg_equipment_group')
    tsg_model_id = request.form.get('tsg_model')
    tsg_asset_number_id = request.form.get('tsg_asset_number')
    tsg_location_id = request.form.get('tsg_location')
    problem_description = request.form.get('problem_description')
    solution_description = request.form.get('solution_description')
    selected_document_ids = request.form.getlist('tsg_document_search')
    selected_problem_image_ids = request.form.getlist('tsg_problem_image_search')
    selected_solution_image_ids = request.form.getlist('tsg_solution_image_search')
    selected_part_ids = request.form.getlist('tsg_selected_part_search')
    selected_drawing_ids = request.form.getlist('tsg_selected_drawing_search')

    # Debug: Log all form data
    logger.info(f"Form Data: {request.form}")

    # Handle missing parts
    if not selected_part_ids:
        logger.info("No selected part IDs provided")
        selected_part_ids = []
    else:
        logger.info(f"Selected part IDs: {selected_part_ids}")
        selected_parts = Session.query(Part).filter(Part.id.in_(selected_part_ids)).all()
        selected_part_ids = [part.id for part in selected_parts]
        logger.info(f"Matching parts from database: {[part.part_number for part in selected_parts]}")
        if not selected_part_ids:
            return jsonify({'error': 'No matching part found'}), 400

    # Handle missing drawings
    if not selected_drawing_ids:
        logger.info("No selected drawing IDs provided")
        selected_drawing_ids = []
    else:
        logger.info(f"Selected drawing IDs: {selected_drawing_ids}")
        selected_drawings = Session.query(Drawing).filter(Drawing.id.in_(selected_drawing_ids)).all()
        selected_drawing_ids = [drawing.id for drawing in selected_drawings]
        logger.info(f"Matching drawings from database: {[drawing.drw_number for drawing in selected_drawings]}")
        if not selected_drawing_ids:
            return jsonify({'error': 'No matching drawing found'}), 400

    # Log the form data received
    logger.info(f"Problem Name: {problem_name}")
    logger.info(f"Model ID: {tsg_model_id}")
    logger.info(f"Asset Number ID: {tsg_asset_number_id}")
    logger.info(f"Location: {tsg_location_id}")
    logger.info(f"Problem Description: {problem_description}")
    logger.info(f"Solution Description: {solution_description}")
    logger.info(f"Selected Document IDs: {selected_document_ids}")
    logger.info(f"Selected Problem Image IDs: {selected_problem_image_ids}")
    logger.info(f"Selected Solution Image IDs: {selected_solution_image_ids}")
    logger.info(f"Selected Part IDs: {selected_part_ids}")
    logger.info(f"Selected Drawing IDs: {selected_drawing_ids}")

    # Check required fields
    if not (problem_name and tsg_area_id and tsg_equipment_group_id and problem_description and solution_description):
        return jsonify({'error': 'All required fields are not provided'}), 400

    # Convert lists to integers
    try:
        selected_document_ids = [int(doc_id) for doc_id in selected_document_ids]
        selected_problem_image_ids = [int(img_id) for img_id in selected_problem_image_ids]
        selected_solution_image_ids = [int(img_id) for img_id in selected_solution_image_ids]
        selected_drawing_ids = [int(drawing_id) for drawing_id in selected_drawing_ids]
    except ValueError as ve:
        logger.error(f"Value conversion error: {str(ve)}")
        return jsonify({'error': 'Invalid ID format in the data provided'}), 400

    try:
        session = Session()

        # Create the Problem entity
        problem = Problem(name=problem_name, description=problem_description)
        session.add(problem)
        session.commit()

        # Create the Position entity
        position = Position(
            area_id=tsg_area_id,
            equipment_group_id=tsg_equipment_group_id,
            model_id=tsg_model_id,
            asset_number_id=tsg_asset_number_id,
            location_id=tsg_location_id
        )
        session.add(position)
        session.commit()

        # Associate Problem with Position
        problem_position_association = ProblemPositionAssociation(problem_id=problem.id, position_id=position.id)
        session.add(problem_position_association)

        # Associate Documents with Problem
        for doc_id in selected_document_ids:
            document_association = CompleteDocumentProblemAssociation(
                problem_id=problem.id,
                complete_document_id=doc_id
            )
            session.add(document_association)

        # Create the Solution entity
        solution = Task(description=solution_description, problem=problem)
        session.add(solution)
        session.commit()

        # Associate Documents with Solution
        for doc_id in selected_document_ids:
            document_association = CompleteDocumentTaskAssociation(
                solution_id=solution.id,
                complete_document_id=doc_id
            )
            session.add(document_association)

        # Associate Problem with Images
        for img_id in selected_problem_image_ids:
            image_problem_association = ImageProblemAssociation(
                image_id=img_id,
                problem_id=problem.id
            )
            session.add(image_problem_association)

        # Associate Solution with Images
        for img_id in selected_solution_image_ids:
            image_solution_association = ImageTaskAssociation(
                image_id=img_id,
                solution_id=solution.id
            )
            session.add(image_solution_association)

        # Associate Parts with Problem
        for part_id in selected_part_ids:
            logger.info(f"Associating part_id {part_id} with problem_id {problem.id}")
            part_problem_association = PartProblemAssociation(
                part_id=part_id,
                problem_id=problem.id
            )
            session.add(part_problem_association)

        # Associate Parts with Solution
        for part_id in selected_part_ids:
            logger.info(f"Associating part_id {part_id} with solution_id {solution.id}")
            part_solution_association = PartTaskAssociation(
                part_id=part_id,
                solution_id=solution.id
            )
            session.add(part_solution_association)

        # Associate Parts with Position
        for part_id in selected_part_ids:
            logger.info(f"Processing part_id: {part_id}")
            part = session.query(Part).filter_by(id=part_id).first()
            if part:
                position_part_association = PartsPositionImageAssociation(
                    part_id=part_id,
                    position_id=position.id
                )
                session.add(position_part_association)
            else:
                logger.error("Error: Part not found for association")
                return jsonify({'error': 'Part not found for association'}), 400

        # Associate Drawings with Position
        for drawing_id in selected_drawing_ids:
            logger.info(f"Processing drawing_id: {drawing_id}")
            drawing = session.query(Drawing).filter_by(id=drawing_id).first()
            if drawing:
                position_drawing_association = DrawingPositionAssociation(
                    drawing_id=drawing_id,
                    position_id=position.id
                )
                session.add(position_drawing_association)
            else:
                logger.error("Error: Drawing not found for association")
                return jsonify({'error': 'Drawing not found for association'}), 400

        # Associate Drawings with Problem
        for drawing_id in selected_drawing_ids:
            logger.info(f"Associating drawing_id {drawing_id} with problem_id {problem.id}")
            drawing = session.query(Drawing).filter_by(id=drawing_id).first()
            if drawing:
                problem_drawing_association = DrawingProblemAssociation(
                    drawing_id=drawing_id,
                    problem_id=problem.id
                )
                session.add(problem_drawing_association)
            else:
                logger.error("Error: Drawing not found for association with problem")
                return jsonify({'error': 'Drawing not found for association with problem'}), 400

        # Commit the changes after associating parts, drawings, and images
        session.commit()
        logger.info("Problem, solution, parts, and drawings associated successfully")

        return render_template('troubleshooting_guide.html')
    except Exception as e:
        session.rollback()
        logger.error(f"Error: {str(e)}", exc_info=e)
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()
