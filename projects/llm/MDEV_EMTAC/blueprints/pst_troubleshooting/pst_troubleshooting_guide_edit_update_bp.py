# pst_troubleshooting_guide_edit_update_bp.py
import os
import traceback
from flask import Blueprint, request, redirect, url_for, jsonify, flash, render_template

from blueprints.assembly_routes import submit_assembly
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import (Drawing, Task, ImageTaskAssociation, Part, Solution, Position, Image,
                                         TaskPositionAssociation, PartTaskAssociation, DrawingTaskAssociation, CompleteDocumentTaskAssociation, CompleteDocument,
                                         Area, EquipmentGroup, AssetNumber, Subassembly, ComponentAssembly, AssemblyView,
                                         Model, Location, SiteLocation, TaskSolutionAssociation, Tool, TaskToolAssociation)
from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError
from modules.configuration.log_config import logger

# Initialize DatabaseConfig
db_config = DatabaseConfig()

# Blueprint initialization
pst_troubleshooting_guide_edit_update_bp = Blueprint('pst_troubleshooting_guide_edit_update_bp', __name__)

# Helper function to update associations
def update_associations(session, model, filter_field, target_id, item_ids, assoc_field, assoc_data_func, assoc_name):
    logger.debug(f"Starting update_associations for {assoc_name} with target_id={target_id} and item_ids={item_ids}")

    if not target_id or not item_ids:
        logger.warning(f"Missing target_id or item_ids for {assoc_name} update. Skipping.")
        return

    try:
        current_assocs = session.query(model).filter(getattr(model, filter_field) == target_id).all()
        current_ids = {getattr(assoc, assoc_field) for assoc in current_assocs}

        to_delete = [assoc for assoc in current_assocs if getattr(assoc, assoc_field) not in item_ids]
        for assoc in to_delete:
            session.delete(assoc)
            logger.debug(f"Deleted {assoc_name} association with {assoc_field}={getattr(assoc, assoc_field)}")

        new_assocs = [assoc_data_func(target_id, item_id) for item_id in item_ids if item_id not in current_ids]
        if new_assocs:
            session.bulk_save_objects(new_assocs)
            logger.info(f"Added {len(new_assocs)} new {assoc_name} associations for {filter_field}={target_id}")
        else:
            logger.info(f"No new {assoc_name} associations to add for {filter_field}={target_id}")

    except Exception as e:
        logger.error(f"Error updating {assoc_name} associations: {traceback.format_exc()}")

def handle_save_position(session, task_id, solution_id, area_id, equipment_group_id, model_id,
                        asset_number_id, location_id, site_location_id,
                        subassembly_id=None, component_assembly_id=None, assembly_view_id=None):
    """
       Handles saving a position to the database.

       Parameters:
           session: Database session.
           task_id (int): ID of the task.
           solution_id (int): ID of the solution.
           area_id (int): ID of the area.
           equipment_group_id (int): ID of the equipment group.
           model_id (int): ID of the model.
           asset_number_id (int): ID of the asset number.
           location_id (int): ID of the location.
           site_location_id (int): ID of the site location.
           subassembly_id (int, optional): ID of the subassembly.
           component_assembly_id (int, optional): ID of the component assembly.
           assembly_view_id (int, optional): ID of the assembly view.

       Returns:
           Flask Response: JSON response indicating success or failure.
       """
    try:
        # Fetch the Task instance
        task = session.query(Task).filter_by(id=task_id).first()
        if not task:
            return jsonify({'error': f'Task with ID {task_id} does not exist.'}), 404

        # Fetch the Solution instance
        solution = session.query(Solution).filter_by(id=solution_id).first()
        if not solution:
            return jsonify({'error': f'Solution with ID {solution_id} does not exist.'}), 404

        # Ensure the Task is associated with the Solution
        existing_task_solution = session.query(TaskSolutionAssociation).filter_by(
            task_id=task.id,
            solution_id=solution.id
        ).first()

        if not existing_task_solution:
            task_solution_association = TaskSolutionAssociation(
                task_id=task.id,
                solution_id=solution.id
            )
            session.add(task_solution_association)
            print(f"Associated Task ID {task.id} with Solution ID {solution.id}")

        # Fetch or create the Position instance
        position = session.query(Position).filter_by(
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            site_location_id=site_location_id,
            subassembly_id=subassembly_id,
            component_assembly_id=component_assembly_id,
            assembly_view_id=assembly_view_id
        ).first()

        if not position:
            # Create a new Position instance
            position = Position(
                area_id=area_id,
                equipment_group_id=equipment_group_id,
                model_id=model_id,
                asset_number_id=asset_number_id,
                location_id=location_id,
                site_location_id=site_location_id,
                subassembly_id=subassembly_id,
                component_assembly_id=component_assembly_id,
                assembly_view_id=assembly_view_id
            )
            session.add(position)
            session.flush()  # Flush to get the position ID
            print(f"Created new Position ID {position.id}")

        # At this point, position.id should be available
        position_id = position.id

        # Check if the association between Task and Position already exists
        existing_task_position = session.query(TaskPositionAssociation).filter_by(
            task_id=task.id,
            position_id=position.id
        ).first()

        if not existing_task_position:
            # Create the association between the task and the position
            task_position_association = TaskPositionAssociation(
                task_id=task.id,
                position_id=position.id
            )
            session.add(task_position_association)
            print(f"Associated Task ID {task.id} with Position ID {position.id}")

        # Commit the transaction
        session.commit()

        # Return the new position_id in the response
        return jsonify({
            'status': 'success',
            'message': 'Position and associations saved successfully.',
            'position_id': position_id  # Include the position_id
        }), 200

    except SQLAlchemyError as e:
        session.rollback()
        error_msg = str(e.__dict__.get('orig'))  # More detailed error message
        print(f"Database error: {error_msg}")
        return jsonify({'error': 'Database error occurred.', 'details': error_msg}), 500
    except Exception as e:
        session.rollback()
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred.', 'details': str(e)}), 500

# Route for editing/updating a Task
@pst_troubleshooting_guide_edit_update_bp.route('/troubleshooting_guide/edit_update_task', methods=['POST'])
def edit_update_task():
    logger.info("Accessed /troubleshooting_guide/edit_update_task route via POST method")

    form_data = request.form
    logger.debug(f"Form data received: {form_data}")

    # Retrieve task-specific data from the form
    task_id = form_data.get('task_id')
    task_name = form_data.get('task_name')
    task_description = form_data.get('task_description')
    logger.debug(f"Parsed Task Data - ID: {task_id}, Name: {task_name}, Description: {task_description}")

    # Collect associated IDs from the form related to the task
    selected_task_image_ids = form_data.getlist('edit_task_image[]')
    selected_document_ids = form_data.getlist('edit_document[]')
    selected_part_ids = form_data.getlist('edit_part[]')
    selected_drawing_ids = form_data.getlist('edit_drawing[]')
    selected_tool_ids = form_data.getlist('edit_tool[]')  # New line for tools
    logger.debug(f"Parsed Associated IDs - Image IDs: {selected_task_image_ids}, Document IDs: {selected_document_ids},"
                 f" Part IDs: {selected_part_ids}, Drawing IDs: {selected_drawing_ids}, Tool IDs: {selected_tool_ids}")

    # Start a database session
    session = db_config.get_main_session()

    try:
        # Update Task details
        task = session.query(Task).filter_by(id=task_id).first()
        if task:
            task.name = task_name
            task.description = task_description
            logger.info(f"Updated Task ID={task_id} with new name and description")
        else:
            flash(f"Task with ID {task_id} not found", 'danger')
            return render_template('troubleshooting_guide.html'), 404

        # Update various task-specific associations
        update_associations(
            session=session,
            model=ImageTaskAssociation,
            filter_field='task_id',
            target_id=task_id,
            item_ids=selected_task_image_ids,
            assoc_field='image_id',
            assoc_data_func=lambda tid, iid: ImageTaskAssociation(task_id=tid, image_id=iid),
            assoc_name='ImageTaskAssociation'
        )
        update_associations(
            session=session,
            model=CompleteDocumentTaskAssociation,
            filter_field='task_id',
            target_id=task_id,
            item_ids=selected_document_ids,
            assoc_field='complete_document_id',
            assoc_data_func=lambda tid, did: CompleteDocumentTaskAssociation(task_id=tid, complete_document_id=did),
            assoc_name='CompleteDocumentTaskAssociation'
        )
        update_associations(
            session=session,
            model=PartTaskAssociation,
            filter_field='task_id',
            target_id=task_id,
            item_ids=selected_part_ids,
            assoc_field='part_id',
            assoc_data_func=lambda tid, partid: PartTaskAssociation(task_id=tid, part_id=partid),
            assoc_name='PartTaskAssociation'
        )
        update_associations(
            session=session,
            model=DrawingTaskAssociation,
            filter_field='task_id',
            target_id=task_id,
            item_ids=selected_drawing_ids,
            assoc_field='drawing_id',
            assoc_data_func=lambda tid, drawingid: DrawingTaskAssociation(task_id=tid, drawing_id=drawingid),
            assoc_name='DrawingTaskAssociation'
        )
        update_associations(
            session=session,
            model=TaskToolAssociation,
            filter_field='task_id',
            target_id=task_id,
            item_ids=selected_tool_ids,
            assoc_field='tool_id',
            assoc_data_func=lambda tid, toolid: TaskToolAssociation(task_id=tid, tool_id=toolid),
            assoc_name='TaskToolAssociation'
        )

        # Commit transaction
        session.commit()
        logger.info("Successfully committed updates to the database for Task")
        flash("Task updated successfully", 'success')
    except Exception as e:
        logger.error(f"Error updating Task: {traceback.format_exc()}")
        session.rollback()
        flash("An error occurred while updating", 'danger')
    finally:
        session.close()
        logger.debug("Database session closed")

    return redirect(url_for('pst_troubleshooting_guide_edit_update_bp.troubleshooting_guide'))

# Search route for Drawings
@pst_troubleshooting_guide_edit_update_bp.route('/search_drawings')
def search_drawings():
    search_term = request.args.get('q', '')
    logger.info(f"Searching for drawings with term '{search_term}'")

    session = db_config.get_main_session()
    drawings = session.query(Drawing).filter(
        or_(
            Drawing.drw_number.ilike(f'%{search_term}%'),
            Drawing.drw_name.ilike(f'%{search_term}%'),
            Drawing.drw_revision.ilike(f'%{search_term}%'),
            Drawing.drw_equipment_name.ilike(f'%{search_term}%')
        )
    ).limit(10).all()

    results = [{'id': drawing.id, 'text': f"{drawing.drw_number} - {drawing.drw_name}"} for drawing in drawings]
    logger.debug(f"Search results: {results}")
    return jsonify(results)

@pst_troubleshooting_guide_edit_update_bp.route('/search_images')
def search_images():
    # Get the search query parameter from the request
    search_term = request.args.get('q', '').strip()
    logger.info(f"Searching images with term: '{search_term}'")

    # Start a database session
    session = db_config.get_main_session()

    try:
        # Query to find images by title or description
        images = session.query(Image).filter(
            or_(
                Image.title.ilike(f'%{search_term}%'),
                Image.description.ilike(f'%{search_term}%')
            )
        ).limit(10).all()

        # Format results for JSON response
        results = [{'id': image.id, 'title': image.title, 'description': image.description} for image in images]
        logger.info(f"Found {len(results)} images matching search term '{search_term}'")

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error occurred while searching images: {traceback.format_exc()}")
        return jsonify({'error': 'An error occurred while searching for images'}), 500

    finally:
        session.close()

@pst_troubleshooting_guide_edit_update_bp.route('/get_areas', methods=['GET'])
def get_areas():
    session = db_config.get_main_session()
    try:
        areas = session.query(Area).all()
        areas_data = [{'id': area.id, 'name': area.name} for area in areas]
        logger.info(f"Fetched {len(areas_data)} areas.")
        return jsonify({"areas": areas_data}), 200
    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching areas: {e}")
        return jsonify({"error": "Failed to fetch areas"}), 500
    finally:
        session.close()

@pst_troubleshooting_guide_edit_update_bp.route('/get_equipment_groups', methods=['GET'])
def get_equipment_groups():
    session = db_config.get_main_session()
    area_id = request.args.get('area_id')
    equipment_groups = session.query(EquipmentGroup).filter_by(area_id=area_id).all()
    data = [{'id': eg.id, 'name': eg.name} for eg in equipment_groups]
    return jsonify(data)

@pst_troubleshooting_guide_edit_update_bp.route('/get_models', methods=['GET'])
def get_models():
    session = db_config.get_main_session()
    equipment_group_id = request.args.get('equipment_group_id')
    models = session.query(Model).filter_by(equipment_group_id=equipment_group_id).all()
    data = [{'id': model.id, 'name': model.name} for model in models]
    return jsonify(data)

@pst_troubleshooting_guide_edit_update_bp.route('/get_asset_numbers', methods=['GET'])
def get_asset_numbers():
    session = db_config.get_main_session()
    model_id = request.args.get('model_id')
    asset_numbers = session.query(AssetNumber).filter_by(model_id=model_id).all()
    data = [{'id': asset.id, 'number': asset.number} for asset in asset_numbers]
    return jsonify(data)

@pst_troubleshooting_guide_edit_update_bp.route('/get_locations', methods=['GET'])
def get_locations():
    session = db_config.get_main_session()
    model_id = request.args.get('model_id')
    locations = session.query(Location).filter_by(model_id=model_id).all()
    data = [{'id': location.id, 'name': location.name} for location in locations]
    return jsonify(data)

@pst_troubleshooting_guide_edit_update_bp.route('/get_subassemblies', methods=['GET'])
def get_assemblies():
    session = db_config.get_main_session()
    location_id = request.args.get('location_id')
    logger.info("get_assemblies called with location_id: %s", location_id)

    # (No explicit type-check is done here; you might consider adding one if needed)
    assemblies = session.query(Subassembly).filter_by(location_id=location_id).all()
    logger.info("Found %d assemblies for location_id %s", len(assemblies), location_id)

    data = [{'id': assembly.id, 'name': assembly.name} for assembly in assemblies]
    logger.info("Returning data: %s", data)
    return jsonify(data)


@pst_troubleshooting_guide_edit_update_bp.route('/component_assemblies', methods=['GET'])
def get_subassemblies():
    session = db_config.get_main_session()
    # Change this line from 'subassembly_id' to 'assembly_id' so it picks up the correct parameter.
    assembly_id = request.args.get('assembly_id')
    logger.info("get_subassemblies called with assembly_id: %s", assembly_id)

    # Validate the presence of assembly_id
    if not assembly_id:
        logger.error("Missing assembly_id in request")
        return jsonify({'error': 'assembly_id is required.'}), 400

    # Validate that assembly_id is an integer
    try:
        assembly_id_int = int(assembly_id)
    except ValueError:
        logger.error("Invalid assembly_id (not an integer): %s", assembly_id)
        return jsonify({'error': 'assembly_id must be an integer.'}), 400

    # Query subassemblies based on assembly_id (filtering by the proper column in ComponentAssembly)
    subassemblies = session.query(ComponentAssembly).filter_by(subassembly_id=assembly_id_int).all()
    logger.info("Found %d subassemblies for assembly_id %d", len(subassemblies), assembly_id_int)

    data = [{'id': subassembly.id, 'name': subassembly.name} for subassembly in subassemblies]
    logger.info("Returning data: %s", data)
    return jsonify(data), 200

@pst_troubleshooting_guide_edit_update_bp.route('/get_assembly_views', methods=['GET'])
def get_assembly_views():
    session = db_config.get_main_session()
    # Try retrieving "component_assembly_id", if not found, fall back to "subassembly_id"
    component_assembly_id = request.args.get('component_assembly_id') or request.args.get('subassembly_id')
    logger.info("get_assembly_views called with component_assembly_id/subassembly_id: %s", component_assembly_id)

    if not component_assembly_id:
        logger.error("Missing component_assembly_id (or subassembly_id) in request")
        return jsonify({'error': 'component_assembly_id is required.'}), 400

    try:
        component_assembly_id_int = int(component_assembly_id)
    except ValueError:
        logger.error("Invalid component_assembly_id (not an integer): %s", component_assembly_id)
        return jsonify({'error': 'component_assembly_id must be an integer.'}), 400

    try:
        assembly_views = session.query(AssemblyView).filter_by(component_assembly_id=component_assembly_id_int).all()
        logger.info("Found %d assembly views for component_assembly_id %d", len(assembly_views), component_assembly_id_int)
    except Exception as e:
        logger.exception("Error querying AssemblyView for component_assembly_id %d", component_assembly_id_int)
        return jsonify({'error': 'Internal server error during query.'}), 500

    try:
        data = [{'id': av.id, 'name': av.name} for av in assembly_views]
        logger.info("Returning data: %s", data)
    except Exception as e:
        logger.exception("Error formatting assembly views data for component_assembly_id %d", component_assembly_id_int)
        return jsonify({'error': 'Internal server error during data formatting.'}), 500

    return jsonify(data), 200

@pst_troubleshooting_guide_edit_update_bp.route('/get_site_locations', methods=['GET'])
def get_site_locations():
    """
    Fetch all Site Locations with detailed logging for debugging.
    """
    session = None
    try:
        # Create a new session
        session = db_config.get_main_session()
        logger.info("Database session created successfully for /get_site_locations.")

        # Debug log: Starting to fetch site locations
        logger.info("Starting to fetch site locations from the database.")

        # Query all Site Locations
        site_locations = session.query(SiteLocation).all()

        # Debug log: Number of entries retrieved
        if site_locations:
            logger.info(f"Fetched {len(site_locations)} site locations from the database.")
        else:
            logger.warning("No site locations found in the database.")

        # Prepare data for JSON response
        data = [{'id': loc.id, 'title': loc.title, 'room_number': loc.room_number} for loc in site_locations]

        # Debug log: Data to be returned in JSON response
        logger.debug(f"Site Location data prepared for response: {data}")

        # Return JSON response
        return jsonify(data), 200

    except Exception as e:
        # Log detailed error information
        logger.error(f"Error fetching site locations: {e}", exc_info=True)

        # Return specific error message
        return jsonify({"error": "Failed to fetch site locations from database."}), 500

    finally:
        # Ensure the session is always closed
        if session:
            session.close()
            logger.info("Database session closed after fetching site locations.")

@pst_troubleshooting_guide_edit_update_bp.route('/save_position', methods=['POST'])
def save_position():
    """
    Endpoint to save a position associated with a task and solution.
    Expects JSON data with task_id, solution_id, and position_data.
    """
    logger.info("Entered '/save_position/' endpoint.")

    # Obtain the custom session
    session = None
    try:
        session = db_config.get_main_session()
        logger.debug("Database session created successfully.")

        # Parse JSON data from the request
        data = request.get_json()
        logger.debug(f"Received JSON data: {data}")

        if not data:
            logger.warning("No JSON data received in the request.")
            return jsonify({'error': 'Invalid or missing JSON data.'}), 400

        # Extract top-level fields
        task_id = data.get('task_id')
        solution_id = data.get('solution_id')
        position_data = data.get('position_data')

        logger.debug(f"Parsed task_id: {task_id}, solution_id: {solution_id}")
        logger.debug(f"Parsed position_data: {position_data}")

        # Validate required top-level fields
        if not task_id or not solution_id or not position_data:
            logger.warning("Missing task_id, solution_id, or position_data in the request.")
            return jsonify({'error': 'Missing task_id, solution_id, or position_data.'}), 400

        # Extract position fields with updated key names
        area_id = position_data.get('area_id')
        equipment_group_id = position_data.get('equipment_group_id')
        model_id = position_data.get('model_id')
        asset_number_id = position_data.get('asset_number_id')
        location_id = position_data.get('location_id')
        subassembly_id = position_data.get('subassembly_id')
        component_assembly_id = position_data.get('component_assembly_id')
        assembly_view_id = position_data.get('assembly_view_id')
        site_location_id = position_data.get('site_location_id')

        logger.debug(
            f"Extracted position fields - area_id: {area_id}, "
            f"equipment_group_id: {equipment_group_id}, model_id: {model_id}, "
            f"asset_number_id: {asset_number_id}, location_id: {location_id}, "
            f"site_location_id: {site_location_id}, "
            f"subassembly_id: {subassembly_id}, component_assembly_id: {component_assembly_id}, "
            f"assembly_view_id: {assembly_view_id}"
        )

        # Validate required position fields
        if not all([area_id, equipment_group_id, model_id]):
            logger.warning("Missing required position fields: area_id, equipment_group_id, or model_id.")
            return jsonify({'error': 'Missing required position fields.'}), 400

        logger.info(
            f"Saving position for task_id: {task_id}, solution_id: {solution_id}, "
            f"area_id: {area_id}, equipment_group_id: {equipment_group_id}, "
            f"model_id: {model_id}, asset_number_id: {asset_number_id}, "
            f"location_id: {location_id}, site_location_id: {site_location_id}, "
            f"subassembly_id: {subassembly_id}, component_assembly_id: {component_assembly_id}, "
            f"assembly_view_id: {assembly_view_id}"
        )

        # Call helper function to handle saving position
        response = handle_save_position(
            session=session,
            task_id=task_id,
            solution_id=solution_id,
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            site_location_id=site_location_id,
            subassembly_id=subassembly_id,
            component_assembly_id=component_assembly_id,
            assembly_view_id=assembly_view_id,
        )

        logger.info("Position saved successfully.")
        return response

    except Exception as e:
        if session:
            session.rollback()
            logger.debug("Database session rolled back due to an error.")

        logger.error(f"An unexpected error occurred in '/save_position/': {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred.', 'details': str(e)}), 500

    finally:
        if session:
            session.close()
            logger.debug("Database session closed.")

@pst_troubleshooting_guide_edit_update_bp.route('/search_documents', methods=['GET'])
def search_documents():
    """Search for documents based on a query term."""
    search_term = request.args.get('query', '').strip()  # Ensure 'query' matches frontend
    if not search_term:
        return jsonify([])  # Return empty list if no query provided

    # Obtain the custom session
    session = db_config.get_main_session()

    try:
        # Perform the query on the title or content fields
        results = session.query(CompleteDocument).filter(
            or_(
                CompleteDocument.title.ilike(f'%{search_term}%'),
                CompleteDocument.content.ilike(f'%{search_term}%')
            )
        ).all()

        # Format the results for JSON response with the title field
        documents = [
            {
                'id': doc.id,
                'text': doc.title,  # Select2 uses 'text' for display
                'file_path': doc.file_path,
                'rev': doc.rev
            }
            for doc in results
        ]
        return jsonify(documents)
    except Exception as e:
        print(f"Error searching documents: {e}")
        return jsonify({'error': 'An error occurred while searching for documents.'}), 500
    finally:
        session.close()  # Ensure the session is closed after the request

@pst_troubleshooting_guide_edit_update_bp.route('/save_task_documents', methods=['POST'])
def save_task_documents():
    """Save selected documents for a specific task."""
    data = request.get_json()
    task_id = data.get('task_id')
    document_ids = data.get('document_ids', [])

    # Validate inputs
    if not task_id or not isinstance(document_ids, list):
        return jsonify({'status': 'error', 'message': 'Invalid input data.'}), 400

    # Obtain the custom session
    session = db_config.get_main_session()

    try:
        # Retrieve existing associations for the task
        existing_associations = session.query(CompleteDocumentTaskAssociation).filter_by(task_id=task_id).all()
        existing_document_ids = {assoc.complete_document_id for assoc in existing_associations}

        # Determine which associations to add and which to remove
        new_document_ids = set(map(int, document_ids))
        to_add = new_document_ids - existing_document_ids
        to_remove = existing_document_ids - new_document_ids

        # Remove associations that are no longer selected
        if to_remove:
            session.query(CompleteDocumentTaskAssociation).filter(
                CompleteDocumentTaskAssociation.task_id == task_id,
                CompleteDocumentTaskAssociation.complete_document_id.in_(to_remove)
            ).delete(synchronize_session='fetch')

        # Add new associations
        for doc_id in to_add:
            new_assoc = CompleteDocumentTaskAssociation(task_id=task_id, complete_document_id=doc_id)
            session.add(new_assoc)

        # Commit the transaction
        session.commit()
        logger.info(f"Saved {len(new_document_ids)} document associations for task ID {task_id}")
        return jsonify({'status': 'success', 'message': 'Documents saved successfully.'})

    except Exception as e:
        session.rollback()  # Roll back in case of error
        logger.error(f"Error saving documents for task ID {task_id}: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while saving documents.'}), 500
    finally:
        session.close()  # Ensure the session is closed after the request

@pst_troubleshooting_guide_edit_update_bp.route('/save_task_drawings', methods=['POST'])
def save_task_drawings():
    data = request.json
    task_id = data.get('task_id')
    drawing_ids = data.get('drawing_ids', [])

    session = db_config.get_main_session()
    try:
        # Clear any existing associations for this task
        session.query(DrawingTaskAssociation).filter_by(task_id=task_id).delete()

        # Add new associations based on selected drawing IDs
        for drawing_id in drawing_ids:
            association = DrawingTaskAssociation(task_id=task_id, drawing_id=drawing_id)
            session.add(association)

        session.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving drawings for task {task_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        session.close()

@pst_troubleshooting_guide_edit_update_bp.route('/search_parts')
def search_parts():
    search_term = request.args.get('q', '')
    logger.info(f"Searching for parts with term '{search_term}'")

    session = db_config.get_main_session()
    parts = session.query(Part).filter(
        or_(
            Part.part_number.ilike(f'%{search_term}%'),
            Part.name.ilike(f'%{search_term}%')
        )
    ).limit(10).all()

    # Ensure 'text' is correctly formatted as part name
    results = [{'id': part.id, 'text': f"{part.part_number} - {part.name}"} for part in parts]
    logger.debug(f"Search results: {results}")
    return jsonify(results)

@pst_troubleshooting_guide_edit_update_bp.route('/save_task_parts', methods=['POST'])
def save_task_parts():
    data = request.json
    task_id = data.get('task_id')
    part_ids = data.get('part_ids', [])

    logger.info(f"Saving parts for task ID: {task_id} with part IDs: {part_ids}")

    session = db_config.get_main_session()
    task = session.query(Task).get(task_id)

    if not task:
        return jsonify({'status': 'error', 'message': 'Task not found'}), 404

    task.part_task = [PartTaskAssociation(part_id=part_id) for part_id in part_ids]

    session.commit()
    logger.info(f"Parts saved successfully for task ID: {task_id}")
    return jsonify({'status': 'success', 'message': 'Parts saved successfully'})

@pst_troubleshooting_guide_edit_update_bp.route('/save_task_images', methods=['POST'])
def save_task_images():
    data = request.get_json()
    task_id = data.get('task_id')
    image_ids = data.get('image_ids', [])

    logger.info(f"Saving images for task {task_id}: {image_ids}")

    if not task_id or not image_ids:
        return jsonify({'status': 'error', 'message': 'Task ID and image IDs are required.'}), 400

    session = db_config.get_main_session()
    try:
        # Clear existing associations for this task
        session.query(ImageTaskAssociation).filter_by(task_id=task_id).delete()

        # Create new associations
        for image_id in image_ids:
            association = ImageTaskAssociation(task_id=task_id, image_id=image_id)
            session.add(association)

        session.commit()
        logger.info(f"Images saved for task {task_id}")
        return jsonify({'status': 'success', 'message': 'Images saved successfully.'})

    except Exception as e:
        session.rollback()
        logger.error(f"Error saving images for task {task_id}: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': 'Failed to save images.'}), 500

    finally:
        session.close()

@pst_troubleshooting_guide_edit_update_bp.route('/remove_position', methods=['POST'])
def remove_position():
    """
    Endpoint to remove a Task's association with a Position.
    Expects JSON data with task_id and position_id.
    """
    logger.info("Entered '/remove_position' endpoint.")

    session = None
    try:
        session = db_config.get_main_session()
        logger.debug("Database session created successfully.")

        # Parse JSON data from the request
        data = request.get_json()
        logger.debug(f"Received JSON data: {data}")

        if not data:
            logger.warning("No JSON data received in the request.")
            return jsonify({'status': 'error', 'error': 'Invalid or missing JSON data.'}), 400

        # Extract task_id and position_id
        task_id = data.get('task_id')
        position_id = data.get('position_id')

        logger.debug(f"Parsed task_id: {task_id}, position_id: {position_id}")

        # Validate required fields
        if not task_id or not position_id:
            logger.warning("Missing task_id or position_id in the request.")
            return jsonify({'status': 'error', 'error': 'Missing task_id or position_id.'}), 400

        # Fetch the TaskPositionAssociation instance
        association = session.query(TaskPositionAssociation).filter_by(
            task_id=task_id,
            position_id=position_id
        ).first()

        if not association:
            logger.warning(f"No association found for Task ID {task_id} and Position ID {position_id}.")
            return jsonify({'status': 'error', 'error': 'Association not found.'}), 404

        # Delete the association
        session.delete(association)
        session.commit()
        logger.info(f"Association between Task ID {task_id} and Position ID {position_id} removed successfully.")

        return jsonify({'status': 'success', 'message': 'Position association removed successfully.'}), 200

    except Exception as e:
        if session:
            session.rollback()
            logger.debug("Database session rolled back due to an error.")

        logger.error(f"An unexpected error occurred in '/remove_position/': {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': 'An unexpected error occurred.', 'details': str(e)}), 500

    finally:
        if session:
            session.close()
            logger.debug("Database session closed.")

@pst_troubleshooting_guide_edit_update_bp.route('/update_task', methods=['POST'])
def update_task():
    """
    Endpoint to update task details.
    Expects JSON data with task_id, name, and description.
    """
    logger.info("Entered '/update_task' endpoint.")

    session = None
    try:
        session = db_config.get_main_session()
        logger.debug("Database session created successfully.")

        # Parse JSON data from the request
        data = request.get_json()
        logger.debug(f"Received JSON data: {data}")

        if not data:
            logger.warning("No JSON data received in the request.")
            return jsonify({'status': 'error', 'message': 'Invalid or missing JSON data.'}), 400

        # Extract fields
        task_id = data.get('task_id')
        name = data.get('name')
        description = data.get('description')

        logger.debug(f"Parsed task_id: {task_id}, name: {name}, description: {description}")

        # Validate inputs
        if not task_id or not name:
            logger.warning("Missing task_id or name in the request.")
            return jsonify({'status': 'error', 'message': 'Missing task_id or name.'}), 400

        # Fetch the Task instance
        task = session.query(Task).filter_by(id=task_id).first()
        if not task:
            logger.warning(f"Task with ID {task_id} not found.")
            return jsonify({'status': 'error', 'message': f'Task with ID {task_id} not found.'}), 404

        # Update the task
        task.name = name
        task.description = description

        # Commit the changes
        session.commit()
        logger.info(f"Task ID {task_id} updated successfully.")

        return jsonify({'status': 'success', 'message': 'Task updated successfully.'}), 200

    except Exception as e:
        if session:
            session.rollback()
            logger.debug("Database session rolled back due to an error.")

        logger.error(f"An unexpected error occurred in '/update_task': {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.', 'error': str(e)}), 500

    finally:
        if session:
            session.close()
            logger.debug("Database session closed.")

@pst_troubleshooting_guide_edit_update_bp.route('/remove_task_document', methods=['POST'])
def remove_task_document():
    data = request.get_json()
    task_id = data.get('task_id')
    document_id = data.get('document_id')

    logger.info(f"Attempting to remove document ID: {document_id} from task ID: {task_id}")

    if not task_id or not document_id:
        logger.error("Task ID and Document ID are required.")
        return jsonify({'status': 'error', 'message': 'Task ID and Document ID are required.'}), 400

    # Obtain the session
    session = db_config.get_main_session()

    try:
        # Fetch the task
        task = session.query(Task).get(task_id)
        if not task:
            logger.error(f"Task with ID {task_id} not found.")
            return jsonify({'status': 'error', 'message': 'Task not found.'}), 404

        # Fetch the association using the session
        association = session.query(CompleteDocumentTaskAssociation).filter_by(task_id=task_id, complete_document_id=document_id).first()
        if not association:
            logger.error(f"Document ID {document_id} is not associated with Task ID {task_id}.")
            return jsonify({'status': 'error', 'message': 'Document not associated with the specified task.'}), 404

        # Remove the association
        session.delete(association)
        session.commit()
        logger.info(f"Successfully removed Document ID {document_id} from Task ID {task_id}.")
        return jsonify({'status': 'success', 'message': 'Document removed successfully.'}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.exception("Database error occurred while removing document from task.")
        return jsonify({'status': 'error', 'message': 'Database error occurred.'}), 500

    except Exception as e:
        session.rollback()
        logger.exception("An unexpected error occurred while removing document from task.")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500

    finally:
        session.close()

@pst_troubleshooting_guide_edit_update_bp.route('/remove_task_drawing', methods=['POST'])
def remove_task_drawing():
    data = request.get_json()
    task_id = data.get('task_id')
    drawing_id = data.get('drawing_id')

    logger.info(f"Attempting to remove drawing ID: {drawing_id} from task ID: {task_id}")

    if not task_id or not drawing_id:
        logger.error("Task ID and Drawing ID are required.")
        return jsonify({'status': 'error', 'message': 'Task ID and Drawing ID are required.'}), 400

    # Obtain the session
    session = db_config.get_main_session()

    try:
        # Fetch the task
        task = session.query(Task).get(task_id)
        if not task:
            logger.error(f"Task with ID {task_id} not found.")
            return jsonify({'status': 'error', 'message': 'Task not found.'}), 404

        # Fetch the association using the session
        association = session.query(DrawingTaskAssociation).filter_by(task_id=task_id, drawing_id=drawing_id).first()
        if not association:
            logger.error(f"Drawing ID {drawing_id} is not associated with Task ID {task_id}.")
            return jsonify({'status': 'error', 'message': 'Drawing not associated with the specified task.'}), 404

        # Remove the association
        session.delete(association)
        session.commit()
        logger.info(f"Successfully removed Drawing ID {drawing_id} from Task ID {task_id}.")
        return jsonify({'status': 'success', 'message': 'Drawing removed successfully.'}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.exception("Database error occurred while removing drawing from task.")
        return jsonify({'status': 'error', 'message': 'Database error occurred.'}), 500

    except Exception as e:
        session.rollback()
        logger.exception("An unexpected error occurred while removing drawing from task.")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500

    finally:
        session.close()

@pst_troubleshooting_guide_edit_update_bp.route('/remove_task_part', methods=['POST'])
def remove_task_part():
    data = request.get_json()
    task_id = data.get('task_id')
    part_id = data.get('part_id')

    logger.info(f"Attempting to remove part ID: {part_id} from task ID: {task_id}")

    if not task_id or not part_id:
        logger.error("Task ID and Part ID are required.")
        return jsonify({'status': 'error', 'message': 'Task ID and Part ID are required.'}), 400

    # Obtain the session
    session = db_config.get_main_session()

    try:
        # Fetch the task
        task = session.query(Task).get(task_id)
        if not task:
            logger.error(f"Task with ID {task_id} not found.")
            return jsonify({'status': 'error', 'message': 'Task not found.'}), 404

        # Fetch the association using the session
        association = session.query(PartTaskAssociation).filter_by(task_id=task_id, part_id=part_id).first()
        if not association:
            logger.error(f"Part ID {part_id} is not associated with Task ID {task_id}.")
            return jsonify({'status': 'error', 'message': 'Part not associated with the specified task.'}), 404

        # Remove the association
        session.delete(association)
        session.commit()
        logger.info(f"Successfully removed Part ID {part_id} from Task ID {task_id}.")
        return jsonify({'status': 'success', 'message': 'Part removed successfully.'}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.exception("Database error occurred while removing part from task.")
        return jsonify({'status': 'error', 'message': 'Database error occurred.'}), 500

    except Exception as e:
        session.rollback()
        logger.exception("An unexpected error occurred while removing part from task.")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500

    finally:
        session.close()

@pst_troubleshooting_guide_edit_update_bp.route('/remove_task_image', methods=['POST'])
def remove_task_image():
    data = request.get_json()
    task_id = data.get('task_id')
    image_id = data.get('image_id')

    logger.info(f"Attempting to remove image ID: {image_id} from task ID: {task_id}")

    if not task_id or not image_id:
        logger.error("Task ID and Image ID are required.")
        return jsonify({'status': 'error', 'message': 'Task ID and Image ID are required.'}), 400

    # Obtain the session
    session = db_config.get_main_session()

    try:
        # Fetch the task
        task = session.query(Task).get(task_id)
        if not task:
            logger.error(f"Task with ID {task_id} not found.")
            return jsonify({'status': 'error', 'message': 'Task not found.'}), 404

        # Fetch the association using the session
        association = session.query(ImageTaskAssociation).filter_by(task_id=task_id, image_id=image_id).first()
        if not association:
            logger.error(f"Image ID {image_id} is not associated with Task ID {task_id}.")
            return jsonify({'status': 'error', 'message': 'Image not associated with the specified task.'}), 404

        # Remove the association
        session.delete(association)
        session.commit()
        logger.info(f"Successfully removed Image ID {image_id} from Task ID {task_id}.")
        return jsonify({'status': 'success', 'message': 'Image removed successfully.'}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.exception("Database error occurred while removing image from task.")
        return jsonify({'status': 'error', 'message': 'Database error occurred.'}), 500

    except Exception as e:
        session.rollback()
        logger.exception("An unexpected error occurred while removing image from task.")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500

    finally:
        session.close()

@pst_troubleshooting_guide_edit_update_bp.route('/search_tools', methods=['GET'])
def search_tools():
    # Get the search query parameter from the request
    search_term = request.args.get('q', '').strip()
    logger.info(f"Searching tools with term: '{search_term}'")

    # Start a database session
    session = db_config.get_main_session()

    try:
        if not search_term:
            logger.warning("Empty search term received for tools.")
            return jsonify({'error': 'No search term provided.'}), 400

        # Query to find tools by name or description
        tools = session.query(Tool).filter(
            or_(
                Tool.name.ilike(f'%{search_term}%'),
                Tool.description.ilike(f'%{search_term}%')
            )
        ).limit(10).all()

        # Format results for JSON response
        results = [{
            'id': tool.id,
            'name': tool.name,
            'size': tool.size,
            'type': tool.type,
            'material': tool.material,
            'description': tool.description
        } for tool in tools]

        logger.info(f"Found {len(results)} tools matching search term '{search_term}'")

        return jsonify(results), 200

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error occurred while searching tools: {error_trace}")
        return jsonify({'error': 'An error occurred while searching for tools.'}), 500

    finally:
        session.close()

@pst_troubleshooting_guide_edit_update_bp.route('/save_task_tools', methods=['POST'])
def save_task_tools():
    """Save selected tools for a specific task."""
    data = request.get_json()
    task_id = data.get('task_id')
    tool_ids = data.get('tool_ids', [])

    # Validate inputs
    if not task_id or not isinstance(tool_ids, list):
        logger.warning("Invalid input data received in save_task_tools.")
        return jsonify({'status': 'error', 'message': 'Invalid input data.'}), 400

    # Start a database session
    # Start a database session
    session = db_config.get_main_session()

    try:
        # Retrieve the task
        task = session.query(Task).filter_by(id=task_id).first()
        if not task:
            logger.warning(f"Task with ID {task_id} not found.")
            return jsonify({'status': 'error', 'message': 'Task not found.'}), 404

        # Retrieve existing tool associations for the task
        existing_associations = session.query(TaskToolAssociation).filter_by(task_id=task_id).all()
        existing_tool_ids = {assoc.tool_id for assoc in existing_associations}

        # Convert tool_ids to integers and remove duplicates
        new_tool_ids = set(map(int, tool_ids))

        # Determine which tool associations to add and which to remove
        to_add = new_tool_ids - existing_tool_ids
        to_remove = existing_tool_ids - new_tool_ids

        # Remove associations that are no longer selected
        if to_remove:
            session.query(TaskToolAssociation).filter(
                TaskToolAssociation.task_id == task_id,
                TaskToolAssociation.tool_id.in_(to_remove)
            ).delete(synchronize_session='fetch')
            logger.debug(f"Removed tool associations for task ID {task_id}: {to_remove}")

        # Add new tool associations
        for tool_id in to_add:
            # Verify that the tool exists
            tool = session.query(Tool).filter_by(id=tool_id).first()
            if tool:
                new_assoc = TaskToolAssociation(task_id=task_id, tool_id=tool_id)
                session.add(new_assoc)
                logger.debug(f"Added tool association: Task ID {task_id}, Tool ID {tool_id}")
            else:
                logger.warning(f"Tool with ID {tool_id} does not exist and cannot be associated with Task ID {task_id}.")

        # Commit the transaction
        session.commit()
        logger.info(f"Saved {len(new_tool_ids)} tool associations for Task ID {task_id}")
        return jsonify({'status': 'success', 'message': 'Tools saved successfully.'}), 200

    except SQLAlchemyError as e:
        session.rollback()  # Roll back in case of error
        error_trace = traceback.format_exc()
        logger.error(f"Database error occurred while saving tools for Task ID {task_id}: {error_trace}")
        return jsonify({'status': 'error', 'message': 'Database error occurred while saving tools.'}), 500

    except Exception as e:
        session.rollback()  # Roll back in case of error
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error occurred while saving tools for Task ID {task_id}: {error_trace}")
        return jsonify({'status': 'error', 'message': 'An error occurred while saving tools.'}), 500

    finally:
        session.close()  # Ensure the session is closed after the request

@pst_troubleshooting_guide_edit_update_bp.route('/remove_task_tools', methods=['POST'])
def remove_task_tool():
    """Remove a specific tool association from a task."""
    data = request.get_json()
    task_id = data.get('task_id')
    tool_id = data.get('tool_id')

    logger.info(f"Attempting to remove Tool ID: {tool_id} from Task ID: {task_id}")

    # Validate inputs
    if not task_id or not tool_id:
        logger.error("Task ID and Tool ID are required.")
        return jsonify({'status': 'error', 'message': 'Task ID and Tool ID are required.'}), 400

    # Start a database session
    session = db_config.get_main_session()

    try:
        # Fetch the task
        task = session.query(Task).filter_by(id=task_id).first()
        if not task:
            logger.error(f"Task with ID {task_id} not found.")
            return jsonify({'status': 'error', 'message': 'Task not found.'}), 404

        # Fetch the association using the session
        association = session.query(TaskToolAssociation).filter_by(task_id=task_id, tool_id=tool_id).first()
        if not association:
            logger.error(f"Tool ID {tool_id} is not associated with Task ID {task_id}.")
            return jsonify({'status': 'error', 'message': 'Tool not associated with the specified task.'}), 404

        # Remove the association
        session.delete(association)
        session.commit()
        logger.info(f"Successfully removed Tool ID {tool_id} from Task ID {task_id}.")
        return jsonify({'status': 'success', 'message': 'Tool removed successfully.'}), 200

    except Exception as e:
        session.rollback()  # Roll back in case of error
        logger.error(f"Error occurred while removing tool from task ID {task_id}: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': 'An error occurred while removing the tool.'}), 500

    finally:
        session.close()