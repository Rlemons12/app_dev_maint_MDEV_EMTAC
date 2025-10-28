from flask import Blueprint, jsonify
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Area, EquipmentGroup, Model, AssetNumber, Location
from modules.configuration.log_config import logger

# Create the blueprint
get_upload_document_list_data_bp = Blueprint('get_upload_document_list_data_bp', __name__)

# Initialize database config
db_config = DatabaseConfig()


@get_upload_document_list_data_bp.route('/get_upload_document_list_data')
def get_upload_document_list_data():
    """Get data for upload document form dropdowns"""

    try:
        logger.info('Fetching data for upload document form dropdowns')

        # Use the proper database session context manager
        with db_config.main_session() as session:
            # Query the database
            areas = session.query(Area).all()
            equipment_groups = session.query(EquipmentGroup).all()
            models = session.query(Model).all()
            asset_numbers = session.query(AssetNumber).all()
            locations = session.query(Location).all()

            # Convert to lists of dictionaries
            areas_list = [{'id': area.id, 'name': area.name} for area in areas]
            equipment_groups_list = [{'id': eg.id, 'name': eg.name, 'area_id': eg.area_id} for eg in equipment_groups]
            models_list = [{'id': model.id, 'name': model.name, 'equipment_group_id': model.equipment_group_id} for
                           model in models]
            asset_numbers_list = [{'id': an.id, 'number': an.number, 'model_id': an.model_id} for an in asset_numbers]
            locations_list = [{'id': loc.id, 'name': loc.name, 'model_id': loc.model_id} for loc in locations]

            # Prepare response data
            data = {
                'areas': areas_list,
                'equipment_groups': equipment_groups_list,
                'models': models_list,
                'asset_numbers': asset_numbers_list,
                'locations': locations_list
            }

            logger.info(
                f'Upload document data: {len(areas_list)} areas, {len(equipment_groups_list)} equipment groups, {len(models_list)} models')

            return jsonify(data)

    except Exception as e:
        logger.error(f"Error fetching upload document data: {e}")
        return jsonify({
            'error': 'Failed to fetch upload document data',
            'message': str(e)
        }), 500