import logging
from flask import Blueprint, jsonify, g
from sqlalchemy.exc import SQLAlchemyError
from modules.configuration.log_config import with_request_id, debug_id, info_id, error_id, get_request_id
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Position, Area, SiteLocation

# Set up blueprint
get_image_list_data_bp = Blueprint('get_image_list_data_bp', __name__)


@get_image_list_data_bp.route('/get_image_list_data')
@with_request_id
def get_list_data():
    """
    Fetch hierarchical data from database for image list interface.
    Returns all levels of the hierarchy including site location data.
    """
    info_id("Fetching complete hierarchical image list data.")

    # Get database session using the DatabaseConfig class
    db_config = DatabaseConfig()
    session = db_config.get_main_session()

    try:
        # Initialize results dictionary with all hierarchy levels
        data = {
            'areas': [],
            'equipment_groups': [],
            'models': [],
            'asset_numbers': [],
            'locations': [],
            'subassemblies': [],
            'component_assemblies': [],
            'assembly_views': [],
            'site_locations': []
        }

        # Fetch site locations (independent of hierarchy)
        debug_id("Fetching site locations")
        site_locations = session.query(SiteLocation).all()
        site_locations_list = [{'id': sl.id, 'name': sl.title} for sl in site_locations]
        data['site_locations'] = site_locations_list

        # Fetch areas (top level)
        debug_id("Fetching areas")
        areas = session.query(Area).all()
        areas_list = [{'id': area.id, 'name': area.name} for area in areas]
        data['areas'] = areas_list

        # Traverse the complete hierarchy
        for area in areas:
            # Get equipment groups for this area
            debug_id(f"Fetching equipment groups for area ID: {area.id}")
            equipment_groups = Position.get_dependent_items(session, 'area', area.id)
            equipment_groups_list = [{'id': eg.id, 'name': eg.name, 'area_id': area.id} for eg in equipment_groups]
            data['equipment_groups'].extend(equipment_groups_list)

            for eg in equipment_groups:
                # Get models for this equipment group
                debug_id(f"Fetching models for equipment group ID: {eg.id}")
                models = Position.get_dependent_items(session, 'equipment_group', eg.id)
                models_list = [{'id': model.id, 'name': model.name, 'equipment_group_id': eg.id} for model in models]
                data['models'].extend(models_list)

                for model in models:
                    # Get asset numbers for this model
                    debug_id(f"Fetching asset numbers for model ID: {model.id}")
                    asset_numbers = Position.get_dependent_items(session, 'model', model.id, 'asset_number')
                    asset_numbers_list = [{'id': an.id, 'number': an.number, 'model_id': model.id} for an in
                                          asset_numbers]
                    data['asset_numbers'].extend(asset_numbers_list)

                    # Get locations for this model
                    debug_id(f"Fetching locations for model ID: {model.id}")
                    locations = Position.get_dependent_items(session, 'model', model.id, 'location')
                    locations_list = [{'id': loc.id, 'name': loc.name, 'model_id': model.id} for loc in locations]
                    data['locations'].extend(locations_list)

                    # Continue down the hierarchy for each location
                    for location in locations:
                        # Get subassemblies for this location
                        debug_id(f"Fetching subassemblies for location ID: {location.id}")
                        subassemblies = Position.get_dependent_items(session, 'location', location.id)
                        subassemblies_list = [{'id': sa.id, 'name': sa.name, 'location_id': location.id}
                                              for sa in subassemblies]
                        data['subassemblies'].extend(subassemblies_list)

                        # Continue down the hierarchy for each subassembly
                        for subassembly in subassemblies:
                            # Get component assemblies for this subassembly
                            debug_id(f"Fetching component assemblies for subassembly ID: {subassembly.id}")
                            comp_assemblies = Position.get_dependent_items(session, 'subassembly', subassembly.id)
                            comp_assemblies_list = [{'id': ca.id, 'name': ca.name, 'subassembly_id': subassembly.id}
                                                    for ca in comp_assemblies]
                            data['component_assemblies'].extend(comp_assemblies_list)

                            # Get the lowest level: assembly views for each component assembly
                            for comp_assembly in comp_assemblies:
                                debug_id(f"Fetching assembly views for component assembly ID: {comp_assembly.id}")
                                assembly_views = Position.get_dependent_items(session, 'component_assembly',
                                                                              comp_assembly.id)
                                assembly_views_list = [
                                    {'id': av.id, 'name': av.name, 'component_assembly_id': comp_assembly.id}
                                    for av in assembly_views]
                                data['assembly_views'].extend(assembly_views_list)

        # Log counts for tracking and debugging
        for key, items in data.items():
            info_id(f"Retrieved {len(items)} {key}")

    except SQLAlchemyError as e:
        error_id(f"Database error while fetching image list data: {str(e)}", exc_info=True)
        session.rollback()
        return jsonify({'error': 'Database error', 'message': str(e)}), 500
    except Exception as e:
        error_id(f"Unexpected error while fetching image list data: {str(e)}", exc_info=True)
        session.rollback()
        return jsonify({'error': 'Server error', 'message': str(e)}), 500
    finally:
        session.close()
        info_id("Database session closed.")

    info_id("Returning JSON response with complete image list data.")
    return jsonify(data)