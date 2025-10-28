from flask import Blueprint, jsonify
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import (Area, EquipmentGroup, Model, AssetNumber, Location,
                                         Subassembly, AssemblyView, ComponentAssembly)

# Blueprint for the assembly model
assembly_model_bp = Blueprint('assembly_model', __name__)

db_config = DatabaseConfig()

@assembly_model_bp.route('/get_list_data')
def get_list_data():
    """
    Fetch and return data for all entities: Area, EquipmentGroup, Model, AssetNumber,
    Location, Subassembly, AssemblyView, and ComponentAssembly.
    """
    try:
        with db_config.get_main_session() as session:
            # Query the database for each entity
            areas = session.query(Area).all()
            equipment_groups = session.query(EquipmentGroup).all()
            models = session.query(Model).all()
            asset_numbers = session.query(AssetNumber).all()
            locations = session.query(Location).all()
            assemblies = session.query(Subassembly).all()
            assembly_views = session.query(AssemblyView).all()
            subassemblies = session.query(ComponentAssembly).all()

            # Convert queried data to a list of dictionaries
            areas_list = [{'id': area.id, 'name': area.name} for area in areas]
            equipment_groups_list = [{'id': group.id, 'name': group.name} for group in equipment_groups]
            models_list = [{'id': model.id, 'name': model.name} for model in models]
            asset_numbers_list = [{'id': number.id, 'number': number.number} for number in asset_numbers]
            locations_list = [{'id': location.id, 'name': location.name} for location in locations]
            assemblies_list = [{'id': assembly.id, 'name': assembly.name, 'description': assembly.description}
                               for assembly in assemblies]
            assembly_views_list = [{'id': view.id, 'name': view.name, 'subassembly_id': view.subassembly_id}
                                   for view in assembly_views]
            subassemblies_list = [{'id': subassembly.id, 'name': subassembly.name,
                                   'description': subassembly.description, 'assembly_id': subassembly.assembly_id}
                                  for subassembly in subassemblies]


    except Exception as e:
        # Use logging instead of print for errors
        import logging
        logging.error(f"An error occurred while fetching list data: {e}")
        return jsonify({"error": "An error occurred while fetching data"}), 500

    # Combine all the lists into a single dictionary
    data = {
        'areas': areas_list,
        'equipment_groups': equipment_groups_list,
        'models': models_list,
        'asset_numbers': asset_numbers_list,
        'locations': locations_list,
        'assemblies': assemblies_list,
        'assembly_views': assembly_views_list,
        'subassemblies': subassemblies_list
    }

    # Return the data as JSON
    return jsonify(data)

# Ensure this Blueprint is registered with a Flask app in your application
