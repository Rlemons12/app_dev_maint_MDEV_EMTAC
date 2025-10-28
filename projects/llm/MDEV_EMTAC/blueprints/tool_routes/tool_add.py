# blueprints/tool_routes/tool_add.py
import os
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from werkzeug.utils import secure_filename
from sqlalchemy.orm import joinedload  # Import for eager loading
from modules.tool_module.forms import ToolForm, ToolCategoryForm, ToolManufacturerForm, ToolSearchForm
from modules.emtacdb.emtacdb_fts import (Tool, ToolCategory, ToolManufacturer, ToolPositionAssociation, Position,
                                         Area, EquipmentGroup, Model, AssetNumber, ToolImageAssociation,
                                         Location, Subassembly, ComponentAssembly, AssemblyView, SiteLocation, Image)
from modules.configuration.log_config import logger
from modules.emtacdb.forms.position_form import PositionForm

# Initialize Blueprint
tool_blueprint_bp = Blueprint('tool_routes', __name__)

# Allowed extensions for tool images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@tool_blueprint_bp.route('/submit_tool_data', methods=['GET', 'POST'])
def submit_tool_data():
    logger.info("Accessed /submit_tool_data route.")

    # Access db_config
    db_config = current_app.config.get('db_config')
    if not db_config:
        error_msg = "Database configuration not found."
        logger.error(error_msg)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Something went wrong'}), 500
        else:
            flash(error_msg, 'danger')
            return render_template(
                'tool_templates/tool_search_entry.html',
                tool_form=None,
                category_form=None,
                manufacturer_form=None,
                position_form=None,
                search_tool_form=None,
                manufacturers=[],
                categories=[],
                positions=[]
            )

    main_session = db_config.get_main_session()

    logger.info("Instantiating forms...")
    tool_form = ToolForm()
    category_form = ToolCategoryForm()
    manufacturer_form = ToolManufacturerForm()
    position_form = PositionForm()
    tool_search_form = ToolSearchForm()
    logger.info("Forms instantiated successfully.")

    try:
        # Populate form choices for various forms
        logger.info("Populating form choices...")

        tool_form.tool_category.choices = [
            (c.id, c.name)
            for c in main_session.query(ToolCategory).order_by(ToolCategory.name)
        ]

        tool_form.tool_manufacturer.choices = [
            (m.id, m.name)
            for m in main_session.query(ToolManufacturer).order_by(ToolManufacturer.name)
        ]

        category_form.parent_id.choices = [(0, 'None')] + [
            (c.id, c.name)
            for c in main_session.query(ToolCategory).order_by(ToolCategory.name)
        ]

        position_form.area.choices = [
            (a.id, a.name)
            for a in main_session.query(Area).order_by(Area.name)
        ]

        # Additional population for position form fields would go here

        tool_search_form.tool_category.choices = [
            (c.id, c.name)
            for c in main_session.query(ToolCategory).order_by(ToolCategory.name)
        ]

        tool_search_form.tool_manufacturer.choices = [
            (m.id, m.name)
            for m in main_session.query(ToolManufacturer).order_by(ToolManufacturer.name)
        ]

        logger.info("Finished populating all form choices successfully.")

    except Exception as e:
        error_msg = f"Error populating form choices: {e}"
        logger.error(error_msg, exc_info=True)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': error_msg}), 500
        else:
            flash(error_msg, 'danger')
            return render_template(
                'tool_templates/tool_search_entry.html',
                tool_form=tool_form,
                category_form=category_form,
                manufacturer_form=manufacturer_form,
                position_form=position_form,
                search_tool_form=tool_search_form,
                manufacturers=[],
                categories=[],
                positions=[]
            )

    is_ajax = (request.headers.get('X-Requested-With') == 'XMLHttpRequest')
    logger.debug(f"request.form data: {request.form.to_dict(flat=False)}")

    if request.method == 'POST':
        logger.info("Inside POST handling...")

        # Determine which form was submitted
        if 'submit_manufacturer' in request.form:
            form = manufacturer_form
            form_name = 'manufacturer'
        elif 'submit_category' in request.form:
            form = category_form
            form_name = 'category'
            logger.info(f"Category Form Submitted: Name - {form.name.data}, Description - {form.description.data}")
        elif 'submit_tool' in request.form:
            form = tool_form
            form_name = 'tool'
        elif 'submit_search' in request.form:
            form = tool_search_form
            form_name = 'search'
        else:
            form = None
            form_name = 'unknown'
            logger.info("No recognized form submission found.")

        if form and not form.validate_on_submit():
            logger.error(f"Form validation errors: {form.errors}")

        if form and form.validate_on_submit():
            logger.info(f"Form '{form_name}' validated successfully.")
            try:
                if form_name == 'manufacturer':
                    logger.info("Handling 'manufacturer' form logic...")
                    # Manufacturer form logic would go here
                    pass

                elif form_name == 'category':
                    logger.info("Handling 'category' form logic...")
                    # Category form logic would go here
                    pass

                elif form_name == 'tool':
                    logger.info("Handling 'tool' form logic...")

                    # Create the new tool
                    new_tool = Tool(
                        name=form.tool_name.data.strip(),
                        size=form.tool_size.data.strip() if form.tool_size.data else None,
                        type=form.tool_type.data.strip() if form.tool_type.data else None,
                        material=form.tool_material.data.strip() if form.tool_material.data else None,
                        description=form.tool_description.data.strip() if form.tool_description.data else None,
                        tool_category_id=form.tool_category.data,
                        tool_manufacturer_id=form.tool_manufacturer.data
                    )

                    logger.info(f"Created new tool instance: {new_tool.name}")
                    main_session.add(new_tool)
                    main_session.flush()  # Get ID without committing

                    # Process image uploads
                    successful_images = 0
                    if form.tool_images.data:
                        import datetime

                        # Add this line to define the timestamp variable
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        for file in form.tool_images.data:
                            try:
                                if file and allowed_file(file.filename):
                                    filename = secure_filename(file.filename)
                                    logger.info(f"Processing file: {filename}")

                                    # Save file
                                    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'tools')
                                    os.makedirs(upload_folder, exist_ok=True)
                                    file_path = os.path.join(upload_folder, filename)
                                    file.save(file_path)

                                    # Get description and create title with timestamp
                                    image_description = form.image_description.data.strip() if form.image_description.data else "Uploaded via the tool form"
                                    image_title = f"{new_tool.name}_{new_tool.id}"

                                    # Use the add_and_associate_with_tool class method
                                    logger.info(f"Adding image and associating with tool ID {new_tool.id}")

                                    new_image, tool_image_assoc = ToolImageAssociation.add_and_associate_with_tool(
                                        session=main_session,
                                        title=image_title,
                                        file_path=file_path,
                                        tool_id=new_tool.id,
                                        description=image_description,
                                        association_description=f"Tool image uploaded on {timestamp}"
                                    )

                                    if new_image and tool_image_assoc:
                                        logger.info(
                                            f"Successfully created image ID {new_image.id} and associated with tool ID {new_tool.id}")
                                        successful_images += 1
                                    else:
                                        logger.warning(f"Failed to create image or association for file {filename}")
                                else:
                                    logger.warning(
                                        f"Skipping invalid or empty file: {file.filename if file else 'None'}")
                            except Exception as img_error:
                                logger.error(f"Error processing image file: {img_error}", exc_info=True)
                                # Continue to next image rather than failing entire process

                    logger.info(f"Processed {successful_images} images successfully")

                    # Process position associations
                    position_associations = 0
                    try:
                        # Get selected values from the form
                        area_id = request.form.get('area', '__None')
                        equipment_group_id = request.form.get('equipment_group', '__None')
                        model_id = request.form.get('model', '__None')
                        asset_number_id = request.form.get('asset_number', '__None')
                        location_id = request.form.get('location', '__None')
                        subassembly_id = request.form.get('subassembly', '__None')
                        component_assembly_id = request.form.get('component_assembly', '__None')
                        assembly_view_id = request.form.get('assembly_view', '__None')
                        site_location_id = request.form.get('site_location', '__None')

                        # Convert string IDs to integers or None
                        def parse_id(id_str):
                            return int(id_str) if id_str != '__None' else None

                        # Parse all IDs
                        area_id = parse_id(area_id) if area_id != '__None' else None
                        equipment_group_id = parse_id(equipment_group_id) if equipment_group_id != '__None' else None
                        model_id = parse_id(model_id) if model_id != '__None' else None
                        asset_number_id = parse_id(asset_number_id) if asset_number_id != '__None' else None
                        location_id = parse_id(location_id) if location_id != '__None' else None
                        subassembly_id = parse_id(subassembly_id) if subassembly_id != '__None' else None
                        component_assembly_id = parse_id(
                            component_assembly_id) if component_assembly_id != '__None' else None
                        assembly_view_id = parse_id(assembly_view_id) if assembly_view_id != '__None' else None
                        site_location_id = parse_id(site_location_id) if site_location_id != '__None' else None

                        # If any position component is selected
                        if any([area_id, equipment_group_id, model_id, asset_number_id, location_id,
                                subassembly_id, component_assembly_id, assembly_view_id, site_location_id]):

                            # Get or create the position
                            position = Position.add_to_db(
                                session=main_session,
                                area_id=area_id,
                                equipment_group_id=equipment_group_id,
                                model_id=model_id,
                                asset_number_id=asset_number_id,
                                location_id=location_id,
                                subassembly_id=subassembly_id,
                                component_assembly_id=component_assembly_id,
                                assembly_view_id=assembly_view_id,
                                site_location_id=site_location_id
                            )

                            # Create one association between tool and this position
                            if position:
                                logger.info(f"Associating position ID {position.id} with Tool '{new_tool.name}'")
                                association = ToolPositionAssociation(
                                    tool_id=new_tool.id,
                                    position_id=position.id,
                                    description=f"Tool position association"
                                )
                                main_session.add(association)
                                position_associations += 1

                    except Exception as pos_assoc_error:
                        logger.error(f"Error processing position associations: {pos_assoc_error}", exc_info=True)

                    logger.info(f"Created {position_associations} position associations")

                    # Commit changes to database
                    main_session.commit()

                    # Success message
                    message = f'Tool "{new_tool.name}" added successfully with {successful_images} images and {position_associations} position associations!'
                    logger.info(message)

                    if is_ajax:
                        return jsonify({'success': True, 'message': message}), 200
                    else:
                        flash(message, 'success')
                        return redirect(url_for('tool_routes.submit_tool_data'))

                elif form_name == 'search':
                    logger.info("Handling 'search' form logic...")
                    # Search form logic would go here
                    pass

                logger.info(f"Completed {form_name} form logic without exceptions.")

            except Exception as e:
                main_session.rollback()
                error_msg = f"Error processing {form_name} form: {str(e)}"
                logger.error(error_msg, exc_info=True)

                if is_ajax:
                    return jsonify({'success': False, 'message': error_msg}), 500
                else:
                    flash(error_msg, 'danger')
                    try:
                        manufacturers = main_session.query(ToolManufacturer).order_by(ToolManufacturer.name).all()
                        categories = main_session.query(ToolCategory).order_by(ToolCategory.name).all()
                        positions = main_session.query(Position).order_by(Position.id).all()
                    except Exception as e2:
                        logger.error(f"Error fetching data during error handling: {e2}", exc_info=True)
                        manufacturers = []
                        categories = []
                        positions = []

                    return render_template(
                        'tool_templates/tool_search_entry.html',
                        tool_form=tool_form,
                        category_form=category_form,
                        manufacturer_form=manufacturer_form,
                        position_form=position_form,
                        search_tool_form=tool_search_form,
                        tools=[],
                        page=1,
                        per_page=20,
                        total_pages=0,
                        manufacturers=manufacturers,
                        categories=categories,
                        positions=positions
                    )
        else:
            # Form validation failed or no valid form submission
            logger.info("Form validation failed or no valid form submission")
            error_msg = "No valid form submission detected."

            if is_ajax:
                return jsonify({'success': False, 'message': error_msg}), 400
            else:
                flash(error_msg, 'danger')
                try:
                    manufacturers = main_session.query(ToolManufacturer).order_by(ToolManufacturer.name).all()
                    categories = main_session.query(ToolCategory).order_by(ToolCategory.name).all()
                    positions = main_session.query(Position).order_by(Position.id).all()
                except Exception as e:
                    logger.error(f"Error fetching data during form validation failure: {e}", exc_info=True)
                    manufacturers = []
                    categories = []
                    positions = []

                return render_template(
                    'tool_templates/tool_search_entry.html',
                    tool_form=tool_form,
                    category_form=category_form,
                    manufacturer_form=manufacturer_form,
                    position_form=position_form,
                    search_tool_form=tool_search_form,
                    tools=[],
                    page=1,
                    per_page=20,
                    total_pages=0,
                    manufacturers=manufacturers,
                    categories=categories,
                    positions=positions
                )

    # Default GET request handling - render the template
    return render_template(
        'tool_templates/tool_search_entry.html',
        tool_form=tool_form,
        category_form=category_form,
        manufacturer_form=manufacturer_form,
        position_form=position_form,
        search_tool_form=tool_search_form,
        manufacturers=[],
        categories=[],
        positions=[]
    )


@tool_blueprint_bp.route('/get_dependent_items')
def get_dependent_items():
    parent_type = request.args.get('parent_type')
    parent_id = request.args.get('parent_id')
    child_type = request.args.get('child_type')

    if not parent_type or not parent_id:
        return jsonify([])

    main_session = current_app.config.get('db_config').get_main_session()
    items = Position.get_dependent_items(main_session, parent_type, parent_id, child_type)

    # Empty result check
    if not items:
        return jsonify([])

    # For asset_number, include the number field
    if (parent_type == 'model' and child_type == 'asset_number') or parent_type == 'asset_number':
        return jsonify([{
            'id': item.id,
            'name': str(item.id),  # Default to ID as display text
            'number': item.number  # Include the actual number field
        } for item in items])

    # For site_location items, use a different format
    elif parent_type == 'site_location' or child_type == 'site_location':
        return jsonify([{
            'id': item.id,
            'name': f"{item.title} - Room {item.room_number}"
        } for item in items])

    # Default case for other items
    else:
        return jsonify([{
            'id': item.id,
            'name': getattr(item, 'name', str(item.id))
        } for item in items])

@tool_blueprint_bp.teardown_app_request
def remove_session(exception=None):
    try:
        db_config = current_app.config.get('db_config')
        if db_config:
            main_session_registry = db_config.get_main_session_registry()
            main_session_registry.remove()
            logger.info("SQLAlchemy session removed successfully.")
        else:
            logger.warning("Database configuration not found during teardown.")
    except Exception as e:
        logger.error(f"Error removing SQLAlchemy session: {e}")

