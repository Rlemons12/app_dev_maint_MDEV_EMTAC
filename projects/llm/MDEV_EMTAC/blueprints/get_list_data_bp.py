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
        print(f'data # Query the database to get all areas, equipment groups, models, asset numbers, and locations')
        areas = session.query(Area).all()
        equipment_groups = session.query(EquipmentGroup).all()
        models = session.query(Model).all()
        asset_numbers = session.query(AssetNumber).all()
        locations = session.query(Location).all()

        # Convert queried data to a list of dictionaries for JSON serialization
        areas_list = [{'id': area.id, 'name': area.name} for area in areas]
        equipment_groups_list = [{'id': equipment_group.id, 'name': equipment_group.name, 'area_id': equipment_group.area_id} for equipment_group in equipment_groups]
        models_list = [{'id': model.id, 'name': model.name, 'equipment_group_id': model.equipment_group_id} for model in models]
        asset_numbers_list = [{'id': number.id, 'number': number.number, 'model_id': number.model_id} for number in asset_numbers]
        locations_list = [{'id': location.id, 'name': location.name, 'model_id': location.model_id} for location in locations]

               
        
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