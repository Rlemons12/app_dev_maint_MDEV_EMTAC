import logging
from flask import Blueprint, jsonify, render_template
from modules.emtacdb.emtacdb_fts import Area, EquipmentGroup, Model, AssetNumber, Location
from modules.configuration.config_env import DatabaseConfig  # Assuming this is your config file

# Initialize the database configuration
db_config = DatabaseConfig()

get_bill_of_material_query_data_bp = Blueprint('get_bill_of_material_query_data_bp', __name__)

@get_bill_of_material_query_data_bp.route('/get_parts_position_data', methods=['GET'])
def get_parts_position_data():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Received request to fetch Parts Position data")
    session = db_config.get_main_session()  # Use the main database session

    try:
        logger.debug("Querying areas from the database")
        areas = session.query(Area).all()
        logger.debug(f"Fetched {len(areas)} areas")

        logger.debug("Querying equipment groups from the database")
        equipment_groups = session.query(EquipmentGroup).all()
        logger.debug(f"Fetched {len(equipment_groups)} equipment groups")

        logger.debug("Querying models from the database")
        models = session.query(Model).all()
        logger.debug(f"Fetched {len(models)} models")

        logger.debug("Querying asset numbers from the database")
        asset_numbers = session.query(AssetNumber).all()
        logger.debug(f"Fetched {len(asset_numbers)} asset numbers")

        logger.debug("Querying locations from the database")
        locations = session.query(Location).all()
        logger.debug(f"Fetched {len(locations)} locations")

        data = {
            'areas': [{'id': area.id, 'name': area.name} for area in areas],
            'equipment_groups': [{'id': group.id, 'name': group.name, 'area_id': group.area_id} for group in equipment_groups],
            'models': [{'id': model.id, 'name': model.name, 'equipment_group_id': model.equipment_group_id} for model in models],
            'asset_numbers': [{'id': asset_number.id, 'number': asset_number.number, 'model_id': asset_number.model_id} for asset_number in asset_numbers],
            'locations': [{'id': location.id, 'name': location.name, 'model_id': location.model_id} for location in locations],
        }

        logger.info("Successfully fetched Parts Position data")
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error occurred while fetching Parts Position data: {e}")
        return jsonify({'error': str(e)})
    finally:
        logger.debug("Closing the database session")
        session.close()


@get_bill_of_material_query_data_bp.route('/search_bill_of_material', methods=['GET'])
def filter_parts_position():
    return render_template('search_bill_of_material.html')