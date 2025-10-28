import logging
from flask import Blueprint, jsonify
from modules.emtacdb.emtacdb_fts import Area, EquipmentGroup, Model, AssetNumber, Location
from modules.configuration.config_env import DatabaseConfig  # Assuming this is your config file
from modules.configuration.log_config import logger

# Initialize the database configuration
db_config = DatabaseConfig()

bill_of_materials_data_bp = Blueprint('bill_of_materials_data_bp', __name__)

@bill_of_materials_data_bp.route('/get_bom_list_data', methods=['GET'])
def get_bom_list_data():
   
    logger.info("Received request to fetch BOM list data")
    session = db_config.get_main_session()  # Use the main database session

    try:
        logger.info("Querying areas from the database")
        areas = session.query(Area).all()
        logger.info(f"Fetched {len(areas)} areas")

        logger.info("Querying equipment groups from the database")
        equipment_groups = session.query(EquipmentGroup).all()
        logger.info(f"Fetched {len(equipment_groups)} equipment groups")

        logger.info("Querying models from the database")
        models = session.query(Model).all()
        logger.info(f"Fetched {len(models)} models")

        logger.info("Querying asset numbers from the database")
        asset_numbers = session.query(AssetNumber).all()
        logger.info(f"Fetched {len(asset_numbers)} asset numbers")

        logger.info("Querying locations from the database")
        locations = session.query(Location).all()
        logger.info(f"Fetched {len(locations)} locations")

        data = {
            'areas': [{'id': area.id, 'name': area.name} for area in areas],
            'equipment_groups': [{'id': group.id, 'name': group.name, 'area_id': group.area_id} for group in equipment_groups],
            'models': [{'id': model.id, 'name': model.name, 'equipment_group_id': model.equipment_group_id} for model in models],
            'asset_numbers': [{'id': asset_number.id, 'number': asset_number.number, 'model_id': asset_number.model_id} for asset_number in asset_numbers],
            'locations': [{'id': location.id, 'name': location.name, 'model_id': location.model_id} for location in locations],
        }

        logger.info("Successfully fetched BOM list data")
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error occurred while fetching BOM list data: {e}")
        return jsonify({'error': str(e)})
    finally:
        logger.info("Closing the database session")
        session.close()
