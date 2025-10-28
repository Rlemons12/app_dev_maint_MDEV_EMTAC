import logging
from flask import Blueprint, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import Part

tsg_search_parts_bp = Blueprint('tsg_search_parts_bp', __name__)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))

@tsg_search_parts_bp.route('/tsg_search_parts', methods=['GET'])
def search_parts():
    session = Session()
    query_param = request.args.get('query', '')
    try:
        print(f'trying tsg_search_parts')
        parts = session.query(Part).filter(Part.part_number.ilike(f'%{query_param}%')).all()
        parts_list = [{'id': part.id, 'part_number': part.part_number, 'name': part.name} for part in parts]
        return jsonify(parts_list)
    except Exception as e:
        logging.error("An error occurred while querying the database:", exc_info=e)
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()
