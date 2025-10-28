import logging
from flask import Blueprint, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import Drawing
from modules.configuration.log_config import get_request_id
from flask import Blueprint, request, jsonify
from typing import List, Optional
from sqlalchemy.orm import Session
search_drawing_by_number_bp = Blueprint('search_drawing_by_number_bp', __name__)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))

@search_drawing_by_number_bp.route('/search_drawing_by_number', methods=['GET'])
def search_drawing_by_number():
    session = Session()
    query_param = request.args.get('query', '')
    try:
        drawings = session.query(Drawing).filter(Drawing.drw_number.ilike(f'%{query_param}%')).all()
        drawings_list = [{'id': drawing.id, 'number': drawing.drw_number, 'name': drawing.drw_name} for drawing in drawings]
        return jsonify(drawings_list)
    except Exception as e:
        logging.error("An error occurred while querying the database:", exc_info=e)
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()



