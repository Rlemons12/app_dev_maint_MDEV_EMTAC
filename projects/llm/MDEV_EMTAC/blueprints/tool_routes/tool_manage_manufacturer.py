# tool_manage_manufacturer.py - Complete refactored version

from blueprints.tool_routes import tool_blueprint_bp
from flask import render_template, redirect, url_for, flash, request, jsonify
from modules.configuration.config_env import DatabaseConfig
from modules.tool_module.forms import ToolForm, ToolCategoryForm, ToolManufacturerForm, ToolSearchForm
from modules.emtacdb.emtacdb_fts import (Tool, ToolCategory, ToolManufacturer, ToolPositionAssociation, Position,
                                         Area, EquipmentGroup, Model, AssetNumber, ToolImageAssociation,
                                         Location, Subassembly, ComponentAssembly, AssemblyView, SiteLocation, Image)
from modules.configuration.log_config import logger
from modules.emtacdb.forms.position_form import PositionForm
import traceback
import time

# Initialize Database Configuration
db_config = DatabaseConfig()


def render_tool_search_page(session, active_tab="search", manufacturer_form=None):
    """
    Helper function to render the tool search page with all required context.

    Args:
        session: Database session
        active_tab: Which tab should be active (search, manufacturers, categories)
        manufacturer_form: Optional pre-filled manufacturer form (for validation errors)

    Returns:
        Rendered template
    """
    # Initialize all forms
    tool_form = ToolForm()
    category_form = ToolCategoryForm()
    search_tool_form = ToolSearchForm()
    position_form = PositionForm()

    # Use provided manufacturer form or create new one
    if manufacturer_form is None:
        manufacturer_form = ToolManufacturerForm()

    # Fetch required data
    manufacturers = session.query(ToolManufacturer).order_by(ToolManufacturer.name).all()
    categories = session.query(ToolCategory).order_by(ToolCategory.name).all()

    # Set default pagination values
    page = 1
    per_page = 20
    total_pages = 1
    tools = []

    return render_template(
        'tool_templates/tool_search_entry.html',
        tool_form=tool_form,
        position_form=position_form,
        manufacturer_form=manufacturer_form,
        category_form=category_form,
        search_tool_form=search_tool_form,
        tools=tools,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        manufacturers=manufacturers,
        categories=categories,
        active_tab=active_tab
    )


@tool_blueprint_bp.route('/tool_manufacturer/add', methods=['GET', 'POST'])
def add_manufacturer():
    """
    Route to add a new manufacturer.
    Handles both displaying the form and processing form submissions.
    """
    logger.info('Step 1: Start of tool_manufacturer.add_manufacturer')
    start_time = time.time()
    logger.info("Route '/tool_manufacturer/add' accessed.")

    try:
        main_session = db_config.get_main_session()
        logger.info('Step 3: Initialized ToolManufacturerForm')

        form = ToolManufacturerForm()
        logger.debug(f'Step 4: Form initialized with data: {form.data}')

        if request.method == 'POST':
            logger.info('Step 5: Detected POST request')
            logger.debug("Received POST request to add manufacturer.")

            if form.validate_on_submit():
                logger.info('Step 6: Form validation succeeded')

                manufacturer_name = form.name.data.strip()
                manufacturer_description = form.description.data.strip() if form.description.data else None
                manufacturer_country = form.country.data.strip() if form.country.data else None
                manufacturer_website = form.website.data.strip() if form.website.data else None

                logger.debug(
                    f"Step 7: Extracted form data - Name: '{manufacturer_name}', Description: '{manufacturer_description}', Country: '{manufacturer_country}', Website: '{manufacturer_website}'"
                )

                # Check for duplicate manufacturer
                logger.info('Step 8: Checking for existing manufacturer in the database')
                existing_manufacturer = main_session.query(ToolManufacturer).filter_by(name=manufacturer_name).first()

                if existing_manufacturer:
                    error_msg = f"Duplicate entry: Manufacturer '{manufacturer_name}' already exists (ID: {existing_manufacturer.id})."
                    logger.warning(f"Step 9: Duplicate manufacturer found - {error_msg}")

                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        logger.info('Step 10: Returning JSON response for duplicate manufacturer')
                        return jsonify({'success': False, 'message': error_msg}), 400
                    else:
                        flash(error_msg, 'warning')
                        logger.info('Step 11: Rendering page with warning for duplicate manufacturer')
                        return render_tool_search_page(main_session, active_tab="manufacturers", manufacturer_form=form)

                # Create and add new manufacturer
                logger.info('Step 12: Creating new ToolManufacturer instance')
                new_manufacturer = ToolManufacturer(
                    name=manufacturer_name,
                    description=manufacturer_description,
                    country=manufacturer_country,
                    website=manufacturer_website
                )
                logger.info('Step 13: Adding new manufacturer to the session')

                try:
                    main_session.add(new_manufacturer)
                    logger.info('Step 14: Committing the session to the database')
                    main_session.commit()

                    success_msg = f"Manufacturer '{manufacturer_name}' added successfully with ID: {new_manufacturer.id}."
                    logger.info(f"Step 15: Commit successful - {success_msg}")

                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        logger.info('Step 16: Returning JSON success response')
                        return jsonify({'success': True, 'message': success_msg}), 200
                    else:
                        flash(success_msg, 'success')
                        logger.info('Step 17: Rendering page with success message')
                        return render_tool_search_page(main_session, active_tab="manufacturers")

                except Exception as e:
                    main_session.rollback()
                    error_msg = f"Exception occurred while adding manufacturer '{manufacturer_name}': {str(e)}"
                    logger.error(f"Step 18: Exception during commit - {error_msg}")
                    logger.debug(traceback.format_exc())

                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        logger.info('Step 19: Returning JSON error response due to exception')
                        return jsonify({'success': False, 'message': error_msg}), 500
                    else:
                        flash(error_msg, 'danger')
                        logger.info('Step 20: Rendering page with error message')
                        return render_tool_search_page(main_session, active_tab="manufacturers", manufacturer_form=form)

            else:
                # Form validation failed
                error_details = {field: errors for field, errors in form.errors.items()}
                logger.warning(f"Step 21: Form validation failed - {error_details}")

                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    logger.info('Step 22: Returning JSON response for form validation errors')
                    return jsonify({'success': False, 'errors': error_details}), 400
                else:
                    for field, errors in form.errors.items():
                        for error in errors:
                            flash(f"Error in {getattr(form, field).label.text}: {error}", 'danger')
                            logger.debug(f"Step 23: Flashing form validation error - {error}")
                    logger.info('Step 24: Rendering page with validation errors')
                    return render_tool_search_page(main_session, active_tab="manufacturers", manufacturer_form=form)

        # For GET request, render the add manufacturer form
        logger.info('Step 25: Handling GET request - rendering add manufacturer form')
        logger.debug("Rendering add manufacturer form (GET request).")

        render_time = time.time() - start_time
        logger.debug(f"Step 26: Route '/tool_manufacturer/add' processed in {render_time:.4f} seconds.")

        return render_tool_search_page(main_session, active_tab="manufacturers")

    except Exception as e:
        # Catch any unforeseen exceptions and log them
        logger.critical(f"Unhandled exception in add_manufacturer route: {str(e)}")
        logger.debug(traceback.format_exc())
        return "An unexpected error occurred.", 500


@tool_blueprint_bp.route('/tool/tool_manufacturer/edit_manufacturer', methods=['POST'])
def edit_manufacturer():
    """
    Route to edit an existing manufacturer.
    Expects 'manufacturer_id' in the form data.
    Handles processing the form submission.
    """
    logger.info("Step 1: Start of edit_manufacturer")
    start_time = time.time()
    main_session = db_config.get_main_session()

    # Retrieve 'manufacturer_id' from form data
    manufacturer_id = request.form.get('manufacturer_id')
    if not manufacturer_id:
        error_msg = "Manufacturer ID is missing in the form data."
        logger.error(f"Step 2: {error_msg}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': error_msg}), 400
        else:
            flash(error_msg, 'danger')
            return render_tool_search_page(main_session, active_tab="manufacturers")

    try:
        manufacturer = main_session.query(ToolManufacturer).get(int(manufacturer_id))
        if not manufacturer:
            error_msg = f"Manufacturer with ID {manufacturer_id} not found."
            logger.error(f"Step 3: {error_msg}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'message': error_msg}), 404
            else:
                flash(error_msg, 'warning')
                return render_tool_search_page(main_session, active_tab="manufacturers")

        form = ToolManufacturerForm(request.form)
        if form.validate():
            updated_name = form.name.data.strip()
            updated_description = form.description.data.strip() if form.description.data else None
            updated_country = form.country.data.strip() if form.country.data else None
            updated_website = form.website.data.strip() if form.website.data else None

            # Check for duplicate manufacturer name (excluding current)
            existing_manufacturer = main_session.query(ToolManufacturer).filter(
                ToolManufacturer.name == updated_name,
                ToolManufacturer.id != int(manufacturer_id)
            ).first()
            if existing_manufacturer:
                error_msg = f"Duplicate entry: Another manufacturer with name '{updated_name}' already exists (ID: {existing_manufacturer.id})."
                logger.warning(f"Step 4: {error_msg}")
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'message': error_msg}), 400
                else:
                    flash(error_msg, 'warning')
                    return render_tool_search_page(main_session, active_tab="manufacturers", manufacturer_form=form)

            # Update manufacturer details
            manufacturer.name = updated_name
            manufacturer.description = updated_description
            manufacturer.country = updated_country
            manufacturer.website = updated_website

            try:
                main_session.commit()
                success_msg = f"Manufacturer ID {manufacturer_id} updated successfully to '{updated_name}'."
                logger.info(f"Step 5: {success_msg}")
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': True, 'message': success_msg}), 200
                else:
                    flash(success_msg, 'success')
                    return render_tool_search_page(main_session, active_tab="manufacturers")
            except Exception as e:
                main_session.rollback()
                error_msg = f"Exception occurred while updating manufacturer ID {manufacturer_id}: {str(e)}"
                logger.error(f"Step 6: {error_msg}")
                logger.debug(traceback.format_exc())
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'message': error_msg}), 500
                else:
                    flash(error_msg, 'danger')
                    return render_tool_search_page(main_session, active_tab="manufacturers", manufacturer_form=form)
        else:
            # Form validation failed
            error_details = {field: errors for field, errors in form.errors.items()}
            logger.warning(f"Step 7: Form validation failed - {error_details}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'errors': error_details}), 400
            else:
                for field, errors in form.errors.items():
                    for error in errors:
                        flash(f"Error in {getattr(form, field).label.text}: {error}", 'danger')
                        logger.debug(f"Step 8: Flashing form validation error - {error}")
                return render_tool_search_page(main_session, active_tab="manufacturers", manufacturer_form=form)
    except Exception as e:
        logger.critical(f"Unhandled exception in edit_manufacturer route: {str(e)}")
        logger.debug(traceback.format_exc())
        return "An unexpected error occurred.", 500
    finally:
        render_time = time.time() - start_time
        logger.debug(f"Step 9: Route '/tool_manufacturer/edit_manufacturer' processed in {render_time:.4f} seconds.")


@tool_blueprint_bp.route('/tool_manufacturer/delete', methods=['POST'])
def delete_manufacturer():
    """
    Route to delete an existing manufacturer.
    Expects 'manufacturer_id' in the form data.
    """
    start_time = time.time()
    logger.info("Step 1: Start of delete_manufacturer")
    logger.info("Route '/tool_manufacturer/delete' accessed.")
    main_session = db_config.get_main_session()
    manufacturer_id = request.form.get('manufacturer_id')

    if not manufacturer_id:
        error_msg = "Deletion attempt without 'manufacturer_id'."
        logger.error(f"Step 2: {error_msg}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': error_msg}), 400
        else:
            flash(error_msg, 'danger')
            return render_tool_search_page(main_session, active_tab="manufacturers")

    try:
        manufacturer = main_session.query(ToolManufacturer).get(int(manufacturer_id))
        if not manufacturer:
            error_msg = f"Manufacturer with ID {manufacturer_id} not found for deletion."
            logger.error(f"Step 3: {error_msg}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'message': error_msg}), 404
            else:
                flash(error_msg, 'warning')
                return render_tool_search_page(main_session, active_tab="manufacturers")

        logger.debug(f"Step 4: Attempting to delete Manufacturer ID {manufacturer_id}: '{manufacturer.name}'")

        # Check if manufacturer is associated with any tools
        associated_tools = manufacturer.tools  # Assuming 'tools' is the relationship attribute
        if associated_tools:
            tool_ids = [tool.id for tool in associated_tools]
            error_msg = (f"Cannot delete Manufacturer ID {manufacturer_id} because it is associated "
                         f"with Tools IDs: {tool_ids}.")
            logger.warning(f"Step 5: {error_msg}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'message': error_msg}), 400
            else:
                flash(error_msg, 'warning')
                return render_tool_search_page(main_session, active_tab="manufacturers")

        # Proceed to delete
        logger.info(f"Step 6: Deleting Manufacturer ID {manufacturer_id}: '{manufacturer.name}'")
        main_session.delete(manufacturer)
        main_session.commit()
        success_msg = f"Manufacturer ID {manufacturer_id} ('{manufacturer.name}') deleted successfully."
        logger.info(f"Step 7: {success_msg}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True, 'message': success_msg}), 200
        else:
            flash(success_msg, 'success')
            return render_tool_search_page(main_session, active_tab="manufacturers")

    except ValueError:
        error_msg = f"Invalid 'manufacturer_id' provided for deletion: '{manufacturer_id}'. Must be an integer."
        logger.error(f"Step 8: {error_msg}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': error_msg}), 400
        else:
            flash(error_msg, 'danger')
            return render_tool_search_page(main_session, active_tab="manufacturers")

    except Exception as e:
        main_session.rollback()
        error_msg = f"Exception occurred while deleting manufacturer ID {manufacturer_id}: {str(e)}"
        logger.error(f"Step 9: {error_msg}")
        logger.debug(traceback.format_exc())
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': error_msg}), 500
        else:
            flash(error_msg, 'danger')
            return render_tool_search_page(main_session, active_tab="manufacturers")

    finally:
        render_time = time.time() - start_time
        logger.debug(f"Step 10: Route '/tool_manufacturer/delete' processed in {render_time:.4f} seconds.")


@tool_blueprint_bp.route('/tool/tool_manufacturer/get/<int:manufacturer_id>', methods=['GET'])
def get_manufacturer(manufacturer_id):
    """
    Route to retrieve manufacturer details for editing.
    This route returns JSON data and doesn't need to use render_tool_search_page.
    """
    logger.info('Step 1: Start of get_manufacturer')
    main_session = db_config.get_main_session()
    try:
        manufacturer = main_session.query(ToolManufacturer).filter_by(id=manufacturer_id).first()
        if manufacturer:
            manufacturer_data = {
                'id': manufacturer.id,
                'name': manufacturer.name,
                'description': manufacturer.description,
                'country': manufacturer.country,
                'website': manufacturer.website
            }
            logger.debug(f"Step 2: Manufacturer found - {manufacturer_data}")
            return jsonify({'success': True, 'manufacturer': manufacturer_data}), 200
        else:
            error_msg = f"Manufacturer with ID {manufacturer_id} not found."
            logger.warning(f"Step 3: {error_msg}")
            return jsonify({'success': False, 'message': error_msg}), 404
    except Exception as e:
        error_msg = f"Exception occurred while fetching manufacturer {manufacturer_id}: {str(e)}"
        logger.error(f"Step 4: {error_msg}")
        logger.debug(traceback.format_exc())
        return jsonify({'success': False, 'message': error_msg}), 500