from flask import Blueprint, render_template, request, jsonify
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Subassembly, AssemblyView, ComponentAssembly
from blueprints.assembly_routes import assembly_model_bp

# Initialize the DatabaseConfig instance
db_config = DatabaseConfig()

@assembly_model_bp.route('/submit_assembly', methods=['POST'])
def submit_assembly():
    data = request.get_json()
    assembly_name = data.get('name')
    assembly_view_name = data.get('assembly_view')
    subassembly_name = data.get('subassembly_name')

    if not all([assembly_name, assembly_view_name, subassembly_name]):
        return jsonify({'error': 'Missing data'}), 400

    try:
        # Use the MainSession context manager for database operations
        with db_config.MainSession() as session:
            # Check if AssemblyView exists, else create it
            assembly_view = session.query(AssemblyView).filter_by(name=assembly_view_name).first()
            if not assembly_view:
                assembly_view = AssemblyView(name=assembly_view_name)
                session.add(assembly_view)
                session.flush()  # Flush to assign an ID without committing

            # Create Subassembly
            assembly = Subassembly(name=assembly_name)
            session.add(assembly)
            session.flush()  # Flush to assign an ID without committing

            # Create ComponentAssembly
            subassembly = ComponentAssembly(
                name=subassembly_name,
                assembly_id=assembly.id,
                assembly_view_id=assembly_view.id
            )
            session.add(subassembly)

            # Commit all changes at once
            session.commit()

        return jsonify({'message': 'Subassembly submitted successfully!'}), 200

    except Exception as e:
        # Optionally log the exception here
        return jsonify({'error': 'An error occurred while submitting the assembly.'}), 500
