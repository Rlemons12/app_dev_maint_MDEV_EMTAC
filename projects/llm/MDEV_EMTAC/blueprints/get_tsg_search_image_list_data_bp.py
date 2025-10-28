from flask import Blueprint, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import Area, EquipmentGroup, Model, AssetNumber, Location, Position, SiteLocation

get_tsg_search_image_list_data_bp = Blueprint('get_tsg_search_image_list_data_bp', __name__)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))  # Use scoped_session here

@get_tsg_search_image_list_data_bp.route('/get_tsg_search_image_list_data_bp')
def get_list_data():
    # Create a session
    session = Session()

    try:
        areas = session.query(Area).all()
        equipment_groups = session.query(EquipmentGroup).all()
        models = session.query(Model).all()
        asset_numbers = session.query(AssetNumber).all()
        locations = session.query(Location).all()
        positions = session.query(Position).all()
        site_locations = session.query(SiteLocation).all()

        # Convert queried data to a list of dictionaries for JSON serialization
        areas_list = [{'id': area.id, 'name': area.name} for area in areas]
        equipment_groups_list = [{'id': equipment_group.id, 'name': equipment_group.name, 'area_id': equipment_group.area_id} for equipment_group in equipment_groups]
        models_list = [{'id': model.id, 'name': model.name, 'equipment_group_id': model.equipment_group_id} for model in models]
        asset_numbers_list = [{'id': number.id, 'number': number.number, 'model_id': number.model_id} for number in asset_numbers]
        locations_list = [{'id': location.id, 'name': location.name, 'model_id': location.model_id} for location in locations]
        positions_list = [{'id': position.id, 'area_id': position.area_id, 'equipment_group_id': position.equipment_group_id, 'model_id': position.model_id, 'asset_number_id': position.asset_number_id, 'location_id': position.location_id, 'site_location_id': position.site_location_id} for position in positions]
        site_locations_list = [{'id': site_location.id, 'title': site_location.title, 'room_number': site_location.room_number} for site_location in site_locations]
       
    except Exception as e:
        logger.error("An error occurred while querying the database: %s", e)
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
        'locations': locations_list,
        'positions': positions_list,
        'site_locations': site_locations_list
    }

    # Return the data as JSON
    return jsonify(data)