# modules/position_data_assignment_tool_management.py

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from modules.tool_module.forms import ToolSearchForm  # Adjust the import path as needed
from modules.emtacdb.emtacdb_fts import (
    Tool, ToolCategory, ToolManufacturer,
    Position, ToolPositionAssociation)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger  # Import the logger

# Define the Blueprint (Ensure this matches your actual Blueprint definition)
position_data_assignment_bp = Blueprint('position_data_assignment_bp', __name__)

# Initialize DatabaseConfig
db_config = DatabaseConfig()

'''@position_data_assignment_bp.route('/manage_tools', methods=['GET', 'POST'])
def manage_tools():
    logger.info("Accessed /manage_tools route.")

    # Obtain a session from DatabaseConfig
    session = db_config.get_main_session()

    try:
        # Get 'position_id' from query parameters or form data
        position_id = request.args.get('position_id', type=int) or request.form.get('position_id', type=int)
        logger.debug(f"Retrieved position_id from request: {position_id}")

        if not position_id:
            logger.error("Position ID not provided in the request.")
            flash('Position ID is required.', 'danger')
            return redirect(url_for('main_bp.home'))  # Replace with your actual main route

        # Fetch the Position object
        position = session.query(Position).get(position_id)
        if not position:
            logger.error(f"Position with ID {position_id} not found.")
            flash('Position not found.', 'danger')
            return redirect(url_for('main_bp.home'))  # Replace with your actual main route

        logger.info(f"Managing tools for Position ID {position_id}: {position}")

        # Instantiate the ToolSearchForm
        tool_search_form = ToolSearchForm()
        logger.debug("Instantiated ToolSearchForm.")

        # Populate choices for SelectMultipleFields
        tool_search_form.tool_category.choices = [
            (c.id, c.name) for c in session.query(ToolCategory).order_by(ToolCategory.name).all()
        ]
        tool_search_form.tool_manufacturer.choices = [
            (m.id, m.name) for m in session.query(ToolManufacturer).order_by(ToolManufacturer.name).all()
        ]
        logger.debug("Populated ToolSearchForm choices for categories and manufacturers.")

        # Initialize an empty list for searched tools
        searched_tools = []
        logger.info("Initialized an empty list for searched tools.")

        if request.method == 'POST':
            if tool_search_form.validate_on_submit():
                logger.info("ToolSearchForm submitted and validated successfully.")

                # Extract form data
                search_name = tool_search_form.tool_name.data
                search_size = tool_search_form.tool_size.data
                search_type = tool_search_form.tool_type.data
                search_material = tool_search_form.tool_material.data
                search_categories = tool_search_form.tool_category.data
                search_manufacturers = tool_search_form.tool_manufacturer.data

                logger.debug(
                    f"Form Data - Name: {search_name}, Size: {search_size}, Type: {search_type}, "
                    f"Material: {search_material}, Categories: {search_categories}, Manufacturers: {search_manufacturers}"
                )

                # Build the query based on search criteria
                logger.info('Building the query based on search criteria.')
                query = session.query(Tool).filter(
                    Tool.tool_position_association.any(position_id=position_id)
                )

                # Apply additional filters
                if search_name:
                    query = query.filter(Tool.name.ilike(f'%{search_name}%'))
                    logger.debug(f"Applied filter: Tool.name LIKE '%{search_name}%'")

                if search_size:
                    query = query.filter(Tool.size.ilike(f'%{search_size}%'))
                    logger.debug(f"Applied filter: Tool.size LIKE '%{search_size}%'")

                if search_type:
                    query = query.filter(Tool.type.ilike(f'%{search_type}%'))
                    logger.debug(f"Applied filter: Tool.type LIKE '%{search_type}%'")

                if search_material:
                    query = query.filter(Tool.material.ilike(f'%{search_material}%'))
                    logger.debug(f"Applied filter: Tool.material LIKE '%{search_material}%'")

                if search_categories:
                    query = query.filter(Tool.tool_category_id.in_(search_categories))
                    logger.debug(f"Applied filter: Tool.tool_category_id IN {search_categories}")

                if search_manufacturers:
                    query = query.filter(Tool.tool_manufacturer_id.in_(search_manufacturers))
                    logger.debug(f"Applied filter: Tool.tool_manufacturer_id IN {search_manufacturers}")

                # Execute the query once after all filters are applied
                searched_tools = query.all()
                logger.info(f"Query executed. Number of tools found: {len(searched_tools)}")

                if not searched_tools:
                    logger.info("No tools found matching the criteria.")
                    flash('No tools found matching the criteria.', 'info')

                # Check if the request is AJAX
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    logger.debug("Detected AJAX request. Preparing JSON response.")

                    # Serialize the tools into JSON-friendly format
                    tools_data = []
                    for tool in searched_tools:
                        tools_data.append({
                            'id': tool.id,
                            'name': tool.name,
                            'size': tool.size or 'N/A',
                            'type': tool.type or 'N/A',
                            'material': tool.material or 'N/A',
                            'tool_category': tool.tool_category.name if tool.tool_category else 'N/A',
                            'tool_manufacturer': tool.tool_manufacturer.name if tool.tool_manufacturer else 'N/A',
                            'edit_url': url_for('position_data_assignment_bp.edit_tool', tool_id=tool.id),
                            'delete_url': url_for('position_data_assignment_bp.delete_tool', tool_id=tool.id)
                        })

                    logger.debug("Returning JSON response for AJAX request.")
                    return jsonify({'tools': tools_data})

            else:
                logger.warning("ToolSearchForm validation failed.")
                flash('Please correct the errors in the form.', 'danger')

        # For GET request or non-AJAX POST request, render the full page
        return render_template(
            'position_data_assignment/position_data_assignment.html',
            tool_search_form=tool_search_form,
            searched_tools=searched_tools,
            position=position
        )

    except Exception as e:
        session.rollback()
        logger.exception(f"An exception occurred in /manage_tools route: {e}")
        flash('An unexpected error occurred.', 'error')
        return redirect(url_for('main_bp.home'))  # Replace with your actual main route

    finally:
        session.close()
        logger.debug("Database session closed.")
'''

@position_data_assignment_bp.route('/edit_tool/<int:tool_id>', methods=['GET', 'POST'])
def edit_tool(tool_id):
    """
    Edit the details of a specific tool.
    """
    logger.info(f'Edit the details of tool with ID: {tool_id}')

    db_session = db_config.get_main_session()
    tool = db_session.query(Tool).filter_by(id=tool_id).first()
    if not tool:
        flash('Tool not found.', 'error')
        return redirect(url_for('position_data_assignment_bp.manage_tools'))  # Adjust endpoint as needed

    if request.method == 'POST':
        # Extract form data and update tool
        tool.name = request.form.get('tool_name')
        tool.size = request.form.get('tool_size')
        tool.type = request.form.get('tool_type')
        tool.material = request.form.get('tool_material')
        tool.description = request.form.get('tool_description')
        tool.tool_manufacturer_id = request.form.get('manufacturer_id')
        tool.tool_category_id = request.form.get('tool_category_id')  # Corrected attribute

        try:
            db_session.commit()
            flash('Tool updated successfully.', 'success')
            return redirect(url_for('position_data_assignment_bp.manage_tools', position_id=tool.tool_position_association[0].position_id))
        except Exception as e:
            db_session.rollback()
            flash('An error occurred while updating the tool.', 'error')
            logger.exception(f"Error updating tool with ID {tool_id}: {e}")
            return redirect(url_for('position_data_assignment_bp.manage_tools', position_id=tool.tool_position_association[0].position_id))

    # For GET request, render an edit form
    manufacturers = db_session.query(ToolManufacturer).all()
    categories = db_session.query(ToolCategory).all()

    return render_template('position_data_assignment/pda_partials/edit_tool.html', tool=tool,
                           manufacturers=manufacturers, categories=categories)

@position_data_assignment_bp.route('/pda_add_tool_to_position', methods=['POST'])
def pda_add_tool_to_position():
    """
    Creates a ToolPositionAssociation for an existing Tool and Position.
    Expects JSON body: { "tool_id": <int>, "position_id": <int> }
    """
    try:
        data = request.get_json()
        tool_id = data.get('tool_id')
        position_id = data.get('position_id')

        if not tool_id or not position_id:
            return jsonify({'message': 'tool_id and position_id are required'}), 400

        with db_config.get_main_session() as session:
            # Verify the tool actually exists
            tool = session.query(Tool).filter_by(id=tool_id).first()
            if not tool:
                return jsonify({'message': 'Tool not found'}), 404

            # Verify the position actually exists
            position = session.query(Position).filter_by(id=position_id).first()
            if not position:
                return jsonify({'message': 'Position not found'}), 404

            # Check if the association already exists
            existing_assoc = session.query(ToolPositionAssociation).filter_by(
                tool_id=tool_id, position_id=position_id
            ).first()
            if existing_assoc:
                return jsonify({'message': 'Tool is already associated with this position'}), 409

            # Create a new association
            new_assoc = ToolPositionAssociation(tool_id=tool_id, position_id=position_id)
            session.add(new_assoc)
            session.commit()

            logger.info(f"Added Tool ID {tool_id} to Position ID {position_id}")

            return jsonify({
                'message': 'Tool successfully added to position',
                'tool_id': tool.id,
                'tool_name': tool.name
            }), 200

    except Exception as e:
        logger.error(f"Error adding tool to position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to add tool to position'}), 500

@position_data_assignment_bp.route('/pda_remove_tool_from_position', methods=['POST'])
def pda_remove_tool_from_position():
    """
    Removes a ToolPositionAssociation record.
    Expects JSON body: { "tool_id": <int>, "position_id": <int> }
    """
    try:
        data = request.get_json()
        tool_id = data.get('tool_id')
        position_id = data.get('position_id')

        if not tool_id or not position_id:
            return jsonify({'message': 'tool_id and position_id are required'}), 400

        with db_config.get_main_session() as session:
            # Find the association
            assoc = session.query(ToolPositionAssociation).filter_by(
                tool_id=tool_id, position_id=position_id
            ).first()

            if not assoc:
                return jsonify({'message': 'Tool is not associated with this position'}), 404

            # Remove the association
            session.delete(assoc)
            session.commit()

            logger.info(f"Removed Tool ID {tool_id} from Position ID {position_id}")
            return jsonify({'message': 'Tool successfully removed from position'}), 200

    except Exception as e:
        logger.error(f"Error removing tool from position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to remove tool from position'}), 500

@position_data_assignment_bp.route('/pda_get_tools_by_position', methods=['GET'])
def pda_get_tools_by_position():
    """
    Returns a JSON list of tools associated with a specific position.
    Endpoint: /pda_get_tools_by_position?position_id=<some_id>
    """
    try:
        # Grab position_id from the query string
        position_id = request.args.get('position_id', type=int)
        if not position_id:
            return jsonify({'message': 'No position_id provided'}), 400

        with db_config.get_main_session() as session:
            # Verify the position exists
            position = session.query(Position).filter_by(id=position_id).first()
            if not position:
                return jsonify({'message': 'Position not found'}), 404

            # Fetch all tool-position associations for this position
            associations = session.query(ToolPositionAssociation)\
                                  .filter_by(position_id=position_id)\
                                  .all()
            if not associations:
                # If no associated tools, return an empty list
                return jsonify({'tools': []}), 200

            # Gather tool IDs
            tool_ids = [assoc.tool_id for assoc in associations]

            # Fetch those tools
            tools = session.query(Tool).filter(Tool.id.in_(tool_ids)).all()

            # Serialize tool data
            tools_list = []
            for t in tools:
                tools_list.append({
                    'id': t.id,
                    'name': t.name,
                    'size': t.size or '',
                    'type': t.type or '',
                    'material': t.material or '',
                    'description': t.description or '',
                    'manufacturer': t.tool_manufacturer.name if t.tool_manufacturer else '',
                    'category': t.tool_category.name if t.tool_category else ''
                })

            return jsonify({'tools': tools_list}), 200

    except Exception as e:
        logger.error(f"Error fetching tools by position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to fetch tools by position'}), 500

@position_data_assignment_bp.route('/pda_search_tools', methods=['POST'])
def pda_search_tools():
    try:
        session = db_config.get_main_session()

        # Since we're receiving form-encoded data (not JSON), just do this:
        name = request.form.get('tool_name', '').strip()
        size = request.form.get('tool_size', '').strip()
        tool_type = request.form.get('tool_type', '').strip()
        material = request.form.get('tool_material', '').strip()
        category_id = request.form.get('tool_category_id')
        manufacturer_id = request.form.get('tool_manufacturer_id')

        # Then build your query based on the form fields.
        query = session.query(Tool)
        if name:
            query = query.filter(Tool.name.ilike(f"%{name}%"))
        if size:
            query = query.filter(Tool.size.ilike(f"%{size}%"))
        if tool_type:
            query = query.filter(Tool.type.ilike(f"%{tool_type}%"))
        if material:
            query = query.filter(Tool.material.ilike(f"%{material}%"))
        if category_id:
            query = query.filter(Tool.tool_category_id == category_id)
        if manufacturer_id:
            query = query.filter(Tool.tool_manufacturer_id == manufacturer_id)

        results = query.all()

        # Serialize into JSON
        tools_data = []
        for t in results:
            tools_data.append({
                'id': t.id,
                'name': t.name,
                'size': t.size or '',
                'type': t.type or '',
                'material': t.material or '',
                'tool_category': t.tool_category.name if t.tool_category else '',
                'tool_manufacturer': t.tool_manufacturer.name if t.tool_manufacturer else ''
            })

        return jsonify({'tools': tools_data}), 200

    except Exception as e:
        logger.error(f"Error in /pda_search_tools: {e}", exc_info=True)
        return jsonify({'message': 'An error occurred while searching for tools'}), 500

    finally:
        session.close()


