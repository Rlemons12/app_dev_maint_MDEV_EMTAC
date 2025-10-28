from flask import Blueprint, request, flash, jsonify, url_for
from modules.emtacdb.emtacdb_fts import (
    ImageTaskAssociation, CompleteDocument, Image, Problem, Position, ProblemPositionAssociation,
    CompleteDocumentProblemAssociation,
    Part, Drawing, DrawingProblemAssociation, DrawingTaskAssociation,
    PartProblemAssociation, PartTaskAssociation, PartsPositionImageAssociation
)
from modules.configuration.config import DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging

# Create a SQLAlchemy engine using the DATABASE_URL from your config
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = sessionmaker(bind=engine)

search_problem_solution_bp = Blueprint('search_problem_solution_bp', __name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@search_problem_solution_bp.route('/search_problem_solution', methods=['GET'])
def search_problem_solution():
    session = Session()
    try:
        # Retrieve parameters from the request
        description = request.args.get('problem_description', '')
        location_id = request.args.get('problem_location', None)
        asset_number_id = request.args.get('problem_asset_number', None)
        model_id = request.args.get('problem_model', None)
        problem_title = request.args.get('problem_title', '')  # Retrieve problem title
        area_id = request.args.get('area_id', None)  # New area filter
        equipment_group_id = request.args.get('equipment_group_id', None)  # New equipment group filter

        # Logging the parameters
        logger.info(f"Description: {description}")
        logger.info(f"Location ID: {location_id}")
        logger.info(f"Asset Number ID: {asset_number_id}")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Problem Title: {problem_title}")
        logger.info(f"Area ID: {area_id}")
        logger.info(f"Equipment Group ID: {equipment_group_id}")

        # Start the query with the Position model
        query = session.query(Position)

        # Apply filters based on provided parameters
        if location_id:
            query = query.filter(Position.location_id == int(location_id))
        if asset_number_id:
            query = query.filter(Position.asset_number_id == int(asset_number_id))
        if model_id:
            query = query.filter(Position.model_id == int(model_id))
        if area_id:
            query = query.filter(Position.area_id == int(area_id))  # Filter by area
        if equipment_group_id:
            query = query.filter(Position.equipment_group_id == int(equipment_group_id))  # Filter by equipment group

        # Retrieve positions based on the filters
        positions = query.all()
        logger.info(f"Positions found: {positions}")

        if not positions:
            flash("No positions found", "error")
            return jsonify(problems=[])

        # Retrieve problems associated with the found positions
        problems = []
        for position in positions:
            problem_positions = session.query(ProblemPositionAssociation).filter_by(position_id=position.id).all()
            for problem_position in problem_positions:
                problem = session.query(Problem).filter_by(id=problem_position.problem_id).first()
                if problem:
                    problems.append(problem)

        logger.info(f"Problems found: {problems}")

        if not problems:
            flash("No problems found", "error")
            return jsonify(problems=[])

        # Collect all parts and drawings for display at the end
        all_parts = []
        all_drawings = []

        # Construct response containing the problems and associated Tasks
        response = []
        for problem in problems:
            position = problem.problem_position[0].position if problem.problem_position else None
            problem_info = {
                'id': problem.id,
                'name': problem.name,
                'description': problem.description,
                'location': position.location.name if position and position.location else None,
                'asset_number': position.asset_number.number if position and position.asset_number else None,
                'model': position.model.name if position and position.model else None,
                'area': position.area.name if position and position.area else None,  # Add area information
                'equipment_group': position.equipment_group.name if position and position.equipment_group else None,
                # Add equipment group info
                'solutions': [],
                'documents': [],
                'images': [],
                'drawings': [],
                'parts': []
            }

            # Collect drawings related to each problem
            drawing_problem_associations = session.query(DrawingProblemAssociation).filter_by(
                problem_id=problem.id).all()
            logger.info(f"Drawing associations for problem {problem.id}: {drawing_problem_associations}")
            for drawing_association in drawing_problem_associations:
                drawing = session.query(Drawing).filter_by(id=drawing_association.drawing_id).first()
                if drawing:
                    drawing_info = {
                        'id': drawing.id,
                        'number': drawing.drw_number,
                        'name': drawing.drw_name
                    }
                    problem_info['drawings'].append(drawing_info)
                    all_drawings.append(drawing_info)

            # Collect parts related to each problem
            part_problem_associations = session.query(PartProblemAssociation).filter_by(problem_id=problem.id).all()
            logger.info(f"Part associations for problem {problem.id}: {part_problem_associations}")
            for part_association in part_problem_associations:
                part = session.query(Part).filter_by(id=part_association.part_id).first()
                if part:
                    part_info = {
                        'id': part.id,
                        'number': part.part_number,
                        'name': part.name,
                        'images': []
                    }
                    problem_info['parts'].append(part_info)
                    all_parts.append(part_info)

            # Collect images and drawings directly related to each solution
            for solution in problem.solution:
                solution_info = {
                    'id': solution.id,
                    'description': solution.description,
                    'drawings': [],
                    'parts': []
                }

                # Fetch drawings related to this solution
                drawing_solution_associations = session.query(DrawingTaskAssociation).filter_by(
                    solution_id=solution.id).all()
                logger.info(f"Drawing associations for solution {solution.id}: {drawing_solution_associations}")
                for drawing_association in drawing_solution_associations:
                    drawing = session.query(Drawing).filter_by(id=drawing_association.drawing_id).first()
                    if drawing:
                        drawing_info = {
                            'id': drawing.id,
                            'number': drawing.drw_number,
                            'name': drawing.drw_name
                        }
                        solution_info['drawings'].append(drawing_info)
                        problem_info['drawings'].append(drawing_info)
                        all_drawings.append(drawing_info)

                # Fetch parts related to this solution
                part_solution_associations = session.query(PartTaskAssociation).filter_by(
                    solution_id=solution.id).all()
                logger.info(f"Part associations for solution {solution.id}: {part_solution_associations}")
                for part_association in part_solution_associations:
                    part = session.query(Part).filter_by(id=part_association.part_id).first()
                    if part:
                        part_info = {
                            'id': part.id,
                            'number': part.part_number,
                            'name': part.name,
                            'images': []
                        }
                        solution_info['parts'].append(part_info)
                        problem_info['parts'].append(part_info)
                        all_parts.append(part_info)

                problem_info['solutions'].append(solution_info)

                # Query and add images associated with the solution
                image_solution_associations = session.query(ImageTaskAssociation).filter_by(
                    solution_id=solution.id).all()
                logger.info(f"Image associations for solution {solution.id}: {image_solution_associations}")
                for association in image_solution_associations:
                    image = session.query(Image).get(association.image_id)
                    if image:
                        image_info = {
                            'id': image.id,
                            'title': image.title,
                            'description': image.description
                        }
                        problem_info['images'].append(image_info)

            # Query and add parts associated with the position
            if position:
                part_position_images = session.query(PartsPositionImageAssociation).filter_by(
                    position_id=position.id).all()
                logger.info(f"Parts and images for position {position.id}: {part_position_images}")

                for part_pos_image in part_position_images:
                    part = session.query(Part).get(part_pos_image.part_id)
                    if part:
                        part_info = {
                            'id': part.id,
                            'number': part.part_number,
                            'name': part.name,
                            'images': []
                        }
                        if part_pos_image.image_id:
                            image = session.query(Image).get(part_pos_image.image_id)
                            if image:
                                image_info = {
                                    'id': image.id,
                                    'title': image.title,
                                    'description': image.description
                                }
                                part_info['images'].append(image_info)

                        problem_info['parts'].append(part_info)
                        all_parts.append(part_info)

            # Retrieve associated documents using the CompleteDocument model
            documents = session.query(CompleteDocument).join(CompleteDocumentProblemAssociation).filter(
                CompleteDocumentProblemAssociation.problem_id == problem.id).all()
            logger.info(f"Documents for problem {problem.id}: {documents}")

            serialized_documents = []
            for document in documents:
                serialized_document = {
                    'id': document.id,
                    'title': document.title
                }
                serialized_documents.append(serialized_document)
            problem_info['documents'] = serialized_documents

            response.append(problem_info)

        # Build HTML content for display
        html_content = ""
        for problem_info in response:
            html_content += f"<h3>Problem:</h3><p>{problem_info['description']}</p>"
            html_content += "<h3>Solutions:</h3>"
            for solution in problem_info['solutions']:
                html_content += f"<p>{solution['description']}</p>"

            html_content += "<h3>Associated Documents:</h3><ul>"
            for document in problem_info['documents']:
                document_link = url_for('search_documents_bp.view_document', document_id=document['id'])
                html_content += f"<li><a href='{document_link}'>{document['title']}</a></li>"
            html_content += "</ul>"

            html_content += "<h3>Associated Images:</h3>"
            for image in problem_info['images']:
                image_link = url_for('serve_image_route', image_id=image['id'])
                html_content += f"""
                    <div class="image-details">
                        <a href="{image_link}">
                            <img class="thumbnail" src="{image_link}" alt="{image['title']}">
                        </a>
                        <div class="description">
                            <h2>{image['title']}</h2>
                            <p>{image['description']}</p>
                        </div>
                    </div>
                """

            html_content += "<h3>Associated Drawings:</h3>"
            for drawing in problem_info['drawings']:
                html_content += f"<p>Drawing Number: {drawing['number']}</p>"
                html_content += f"<p>Drawing Name: {drawing['name']}</p>"
            html_content += "<hr>"

            html_content += "<h3>Associated Parts:</h3>"
            for part in problem_info['parts']:
                html_content += f"<p>Part Number: {part['number']}</p>"
                if part['images']:
                    html_content += "<h4>Part Images:</h4>"
                    for part_image in part['images']:
                        part_image_link = url_for('serve_image_route', image_id=part_image['id'])
                        html_content += f"""
                            <div class="part-image-details">
                                <a href="{part_image_link}">
                                    <img class="thumbnail" src="{part_image_link}" alt="{part_image['title']}">
                                </a>
                                <div class="description">
                                    <h3>{part_image['title']}</h3>
                                    <p>{part_image['description']}</p>
                                </div>
                            </div>
                        """
            html_content += "<hr>"

        # Display all collected parts and drawings at the end
        html_content += "<h3>All Associated Parts:</h3>"
        for part in all_parts:
            html_content += f"<p>Part Number: {part['number']}, Part Name: {part['name']}</p>"

        html_content += "<h3>All Associated Drawings:</h3>"
        for drawing in all_drawings:
            html_content += f"<p>Drawing Number: {drawing['number']}, Drawing Name: {drawing['name']}</p>"

        return html_content

    except SQLAlchemyError as e:
        logger.error("An error occurred while retrieving problems: %s", e)
        flash("An error occurred while retrieving problems: {}".format(e), "error")
        return jsonify(error=str(e)), 500

    finally:
        session.close()  # Ensure the session is closed
