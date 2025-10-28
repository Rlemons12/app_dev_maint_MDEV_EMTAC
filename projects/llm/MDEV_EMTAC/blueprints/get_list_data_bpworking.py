from flask import Blueprint, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import Area, EquipmentGroup, Model, AssetNumber, Location

get_list_data_bp = Blueprint('get_list_data_bp', __name__)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))  # Use scoped_session here

@get_list_data_bp.route('/get_list_data')
def get_list_data():
    # Create a session
    session = Session()

    try:
        # Query the database to get all areas
        areas = session.query(Area).all()
        equipment_groups = session.query(EquipmentGroup).all()
        models = session.query(Model).all()
        asset_numbers = session.query(AssetNumber).all()
        locations = session.query(Location).all()

        # Convert queried data to a list of dictionaries for JSON serialization
        areas_list = [{'id': area.id, 'name': area.name} for area in areas]
        equipment_groups_list = [{'id': group.id, 'name': group.name} for group in equipment_groups]
        models_list = [{'id': model.id, 'name': model.name} for model in models]
        asset_numbers_list = [{'id': number.id, 'number': number.number} for number in asset_numbers]
        locations_list = [{'id': location.id, 'name': location.name} for location in locations]
    except Exception as e:
        print("An error occurred:", e)
        session.rollback()
        raise e

    finally:
        # Close the session
        session.close()

    # Combine all the lists into a single dictionary
    data = {
        'areas': areas_list,
        'equipment_groups': equipment_groups_list,
        'models': models_list,
        'asset_numbers': asset_numbers_list,
        'locations': locations_list
    }

    # Return the data as JSON
    return jsonify(data)

if __name__ == '__main__':
    get_list_data_bp.run(debug=True)
