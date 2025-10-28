import logging
from flask import Blueprint, jsonify, request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import (Area, EquipmentGroup, Model, AssetNumber, Location, Image, CompleteDocument, Part,
                                         Drawing, Position, SiteLocation, Problem, Task,
                                         ImageProblemAssociation, ImageTaskAssociation, CompleteDocumentProblemAssociation,
                                         PartProblemAssociation, DrawingProblemAssociation)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

get_troubleshooting_guide_data_bp = Blueprint('get_troubleshooting_guide_data_bp', __name__)

engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))

@get_troubleshooting_guide_data_bp.route('/get_troubleshooting_guide_data_bp')
def get_list_data():
    session = Session()
    try:
        logger.info('Querying the database to get all areas, equipment groups, models, asset numbers, locations, and documents')
        areas = session.query(Area).all()
        equipment_groups = session.query(EquipmentGroup).all()
        models = session.query(Model).all()
        asset_numbers = session.query(AssetNumber).all()
        locations = session.query(Location).all()
        documents = session.query(CompleteDocument).all()
        images = session.query(Image).all()
        drawings = session.query(Drawing).all()
        parts = session.query(Part).all()
        positions = session.query(Position).all()
        site_locations = session.query(SiteLocation).all()

        logger.info('Data successfully queried from the database')

        areas_list = [{'id': area.id, 'name': area.name, 'description': area.description} for area in areas]
        equipment_groups_list = [{'id': equipment_group.id, 'name': equipment_group.name, 'area_id': equipment_group.area_id} for equipment_group in equipment_groups]
        models_list = [{'id': model.id, 'name': model.name, 'description': model.description, 'equipment_group_id': model.equipment_group_id} for model in models]
        asset_numbers_list = [{'id': number.id, 'number': number.number, 'description': number.description, 'model_id': number.model_id} for number in asset_numbers]
        locations_list = [{'id': location.id, 'name': location.name, 'model_id': location.model_id} for location in locations]
        documents_list = [{'id': doc.id, 'title': doc.title, 'file_path': doc.file_path} for doc in documents]
        images_list = [{'id': image.id, 'title': image.title, 'description': image.description} for image in images]
        drawings_list = [{'id': drawing.id, 'number': drawing.drw_number, 'name': drawing.drw_name} for drawing in drawings]
        parts_list = [{'id': part.id, 'part_number': part.part_number, 'name': part.name, 'documentation': part.documentation} for part in parts]
        positions_list = [{'id': position.id, 'area_id': position.area_id, 'equipment_group_id': position.equipment_group_id, 'model_id': position.model_id, 'asset_number_id': position.asset_number_id, 'location_id': position.location_id, 'site_location_id': position.site_location_id} for position in positions]
        site_locations_list = [{'id': site_location.id, 'title': site_location.title, 'room_number': site_location.room_number} for site_location in site_locations]

        logger.info('Data successfully converted to lists of dictionaries for JSON serialization')
    except Exception as e:
        logger.error("An error occurred while querying the database:", exc_info=e)
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()
        logger.info('Database session closed')

    data = {
        'areas': areas_list,
        'equipment_groups': equipment_groups_list,
        'models': models_list,
        'asset_numbers': asset_numbers_list,
        'locations': locations_list,
        'documents': documents_list,
        'images': images_list,
        'drawings': drawings_list,
        'parts': parts_list,
        'positions': positions_list,
        'site_locations': site_locations_list,
    }

    logger.info('Data combined into a single dictionary')
    """logger.info(f'Drawings List: {drawings_list}')"""
    return jsonify(data)

@get_troubleshooting_guide_data_bp.route('/get_problem_solution_data/<int:problem_id>', methods=['GET'])
def get_problem_solution_data(problem_id):
    session = Session()
    try:
        logger.info(f'Querying the database to get problem and solution data for problem_id: {problem_id}')

        # Fetch the problem and related solution
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if not problem:
            return jsonify({'error': 'Problem not found'}), 404

        solution = session.query(Task).filter_by(problem_id=problem_id).first()

        # Fetch associated problem images
        problem_images = session.query(ImageProblemAssociation).filter_by(problem_id=problem_id).all()
        problem_image_data = [{'id': img.image.id, 'title': img.image.title, 'file_path': img.image.file_path} for img in problem_images]

        # Fetch associated solution images
        solution_images = session.query(ImageTaskAssociation).filter_by(solution_id=solution.id).all() if solution else []
        solution_image_data = [{'id': img.image.id, 'title': img.image.title, 'file_path': img.image.file_path} for img in solution_images]

        # Fetch associated documents
        problem_documents = session.query(CompleteDocumentProblemAssociation).filter_by(problem_id=problem_id).all()
        document_data = [{'id': doc.complete_document_id, 'title': doc.complete_document.title} for doc in problem_documents]

        # Fetch associated parts
        problem_parts = session.query(PartProblemAssociation).filter_by(problem_id=problem_id).all()
        part_data = [{'id': part.part_id, 'name': part.part.name} for part in problem_parts]

        # Fetch associated drawings
        problem_drawings = session.query(DrawingProblemAssociation).filter_by(problem_id=problem_id).all()
        drawing_data = [{'id': drw.drawing.id, 'number': drw.drawing.drw_number} for drw in problem_drawings]  # Assuming `drw_number` is the drawing number field

        # Convert problem and solution data to dictionary
        problem_data = {
            'id': problem.id,
            'name': problem.name,
            'description': problem.description,
            'images': problem_image_data,
            'documents': document_data,
            'parts': part_data,
            'drawings': drawing_data,  # Add drawings here
        }

        solution_data = {
            'id': solution.id if solution else None,
            'description': solution.description if solution else '',
            'images': solution_image_data,
        }

        return jsonify({
            'problem': problem_data,
            'solution': solution_data,
        })
    except Exception as e:
        logger.error("An error occurred while querying the database:", exc_info=e)
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()
        logger.info('Database session closed')
# region fixme: had to change to /search_document, endpoint issues with blueprints/search_documents_bp.py
@get_troubleshooting_guide_data_bp.route('/search_document', methods=['GET'])
def search_documents():
    query = request.args.get('query', '')

    session = Session()
    try:
        # Search documents by title, description, or any other criteria
        documents = session.query(CompleteDocument).filter(CompleteDocument.title.ilike(f'%{query}%')).all()

        # Prepare the response
        document_data = [{'id': doc.id, 'title': doc.title} for doc in documents]

        return jsonify({'documents': document_data})
    except Exception as e:
        logger.error(f"An error occurred while searching for documents: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()
