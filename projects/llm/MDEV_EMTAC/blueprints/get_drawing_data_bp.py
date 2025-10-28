from flask import Blueprint, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import Drawing

get_drawing_data_bp = Blueprint('get_drawing_data_bp', __name__)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))  # Use scoped_session here

@get_drawing_data_bp.route('/get_drawing_data')
def get_list_data():
    # Create a session
    session = Session()

    try:
        drawings_list = session.query(Drawing).all()
        parts_list = sessio.query(Parts).all()
        
        # Convert queried data to a list of dictionaries for JSON serialization
        drawing_list = [{'id': drawing.id, 'Equipment Name': drawing.drw_equipment_name, 'Number': drawing.drw_number, 'Name': drawing.drw_name, 'Revision': drawing.drw_revision, 'Spare Part Number': drawing.drw_spare_part_number} for drawing in drawings]
        

        
    except Exception as e:
        print("An error occurred:", e)
        session.rollback()
        raise e

    finally:
        # Close the session
        session.close()

    # Combine all the lists into a single dictionary
    data = {
        'drawing' : drawing_list,
        
    }

    # Print out the documents list
    print("Documents List:")
    for document in documents_list:
        print(document)

    # Print out Image list
    print("Image List:")
    for image in images_list:
        print(image)

    
    # Return the data as JSON
    return jsonify(data)

