from flask import Blueprint, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import Drawing
import logging

tsg_search_drawing_bp = Blueprint('tsg_search_drawing_bp', __name__)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tsg_search_drawing_bp.route('/tsg_search_drawing', methods=['GET'])
def tsg_search_drawing():
    session = Session()
    try:
        # Get search parameters from request
        equipment_name = request.args.get('drw_equipment_name')
        drawing_number = request.args.get('drw_number')
        drawing_name = request.args.get('drw_name')
        revision = request.args.get('drw_revision')
        spare_part_number = request.args.get('drw_spare_part_number')

        # Log the received search parameters
        logger.info(f'Search parameters: equipment_name={equipment_name}, drawing_number={drawing_number}, drawing_name={drawing_name}, revision={revision}, spare_part_number={spare_part_number}')

        # Build query based on search parameters
        query = session.query(Drawing)
        if equipment_name:
            query = query.filter(Drawing.drw_equipment_name.ilike(f'%{equipment_name}%'))
        if drawing_number:
            query = query.filter(Drawing.drw_number.ilike(f'%{drawing_number}%'))
        if drawing_name:
            query = query.filter(Drawing.drw_name.ilike(f'%{drawing_name}%'))
        if revision:
            query = query.filter(Drawing.drw_revision.ilike(f'%{revision}%'))
        if spare_part_number:
            query = query.filter(Drawing.drw_spare_part_number.ilike(f'%{spare_part_number}%'))

        # Execute query and fetch results
        results = query.all()
        drawings_list = [{'id': drawing.id, 'equipment_name': drawing.drw_equipment_name, 'number': drawing.drw_number, 'name': drawing.drw_name, 'revision': drawing.drw_revision, 'spare_part_number': drawing.drw_spare_part_number} for drawing in results]

        # Log the number of results found
        logger.info(f'Found {len(drawings_list)} drawings')

    except Exception as e:
        logger.error("An error occurred:", e)
        session.rollback()
        return jsonify({'error': str(e)}), 500

    finally:
        session.close()

    return jsonify(drawings_list)
