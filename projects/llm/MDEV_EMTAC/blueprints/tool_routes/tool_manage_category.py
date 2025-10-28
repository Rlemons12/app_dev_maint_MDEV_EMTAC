# tool_manage_category.py
from blueprints.tool_routes import tool_blueprint_bp
from flask import render_template, redirect, url_for, flash, request
from modules.configuration.config_env import DatabaseConfig
from modules.tool_module.forms import ToolForm,ToolCategoryForm,ToolSearchForm,ToolManufacturerForm
from modules.emtacdb.forms import PositionForm
from modules.emtacdb.emtacdb_fts import ToolCategory, ToolManufacturer
from modules.configuration.log_config import logger

db_config = DatabaseConfig()

def render_tool_search_page(session, active_tab="search", manufacturer_form=None, category_form=None):
    """
    Helper function to render the tool search page with all required context.

    Args:
        session: Database session
        active_tab: Which tab should be active (search, manufacturers, categories)
        manufacturer_form: Optional pre-filled manufacturer form (for validation errors)
        category_form: Optional pre-filled category form (for validation errors)

    Returns:
        Rendered template
    """
    # Initialize all forms
    tool_form = ToolForm()
    search_tool_form = ToolSearchForm()
    position_form = PositionForm()

    # Use provided forms or create new ones
    if manufacturer_form is None:
        manufacturer_form = ToolManufacturerForm()
    if category_form is None:
        category_form = ToolCategoryForm()

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

@tool_blueprint_bp.route('/tool_category/add', methods=['GET', 'POST'])
@tool_blueprint_bp.route('/tool_category/add_category', methods=['GET', 'POST'])
def add_tool_category():
    logger.info("Accessing /tool_category/add_category route for adding a new category.")

    # Instantiate the form and get a database session
    category_form = ToolCategoryForm()
    session = db_config.get_main_session()
    logger.info("Database session acquired successfully.")

    # Populate parent category choices
    try:
        logger.info("Querying database for existing tool categories to populate parent choices.")
        categories = session.query(ToolCategory).order_by(ToolCategory.name).all()
        num_categories = len(categories)
        logger.info(f"Retrieved {num_categories} categories from the database.")
        category_form.parent_id.choices = [(0, 'None')] + [(c.id, c.name) for c in categories]
        logger.info("Parent category choices populated successfully.")
    except Exception as e:
        logger.error(f"Error fetching category choices: {e}", exc_info=True)
        flash("An error occurred while fetching category choices.", "danger")

    # Process the form submission when validated
    if category_form.validate_on_submit():
        logger.info("Form validation successful. Processing form data.")
        category_name = category_form.name.data.strip()
        category_description = category_form.description.data.strip()
        parent_choice = category_form.parent_id.data
        logger.info(
            f"Form Data - Name: '{category_name}', Description: '{category_description}', Parent ID: {parent_choice}")

        # Create new ToolCategory instance
        new_category = ToolCategory(
            name=category_name,
            description=category_description,
            parent_id=parent_choice if parent_choice != 0 else None
        )
        logger.info(f"New ToolCategory instance created: {new_category}")

        # Add the new category to the session
        session.add(new_category)
        logger.info("New category added to the database session.")

        try:
            logger.info("Attempting to commit the new category to the database.")
            session.commit()
            logger.info(f"Category '{new_category.name}' saved successfully with ID: {new_category.id}!")
            flash("Category saved successfully!", "success")
            return redirect(url_for("tool_routes.add_tool_category"))
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving category: {e}. Rolled back the transaction.", exc_info=True)
            flash("An error occurred while saving the category. Please try again.", "danger")
    else:
        if request.method == 'POST':
            logger.info("Form submission failed validation. Errors: %s", category_form.errors)

    logger.info("Rendering add category template using render_tool_search_page.")
    return render_tool_search_page(
        session=session,
        active_tab="categories",
        category_form=category_form
    )

# NOt_FUNC: not editing category

@tool_blueprint_bp.route('/tool_category/edit_tool_category', methods=['GET', 'POST'])
@tool_blueprint_bp.route('/tool_category/edit_tool_category/<int:category_id>', methods=['GET', 'POST'])
def edit_tool_category(category_id):
    logger.info(f"Accessing /tool_category/edit_tool_category route for category ID: {category_id}")

    category_form = ToolCategoryForm()
    session = db_config.get_main_session()

    # Get the category to edit
    category = session.query(ToolCategory).get(category_id)
    if not category:
        logger.warning(f"Category with ID {category_id} not found!")
        flash("Category not found.", "danger")
        return redirect(url_for("tool_routes.add_tool_category"))

    # Populate parent category choices (exclude the current category to prevent circular references)
    try:
        logger.info("Populating parent category choices.")
        all_categories = session.query(ToolCategory).order_by(ToolCategory.name).all()
        category_form.parent_id.choices = [(0, 'None')] + [
            (c.id, c.name) for c in all_categories if c.id != category.id
        ]
        logger.info("Parent category choices populated successfully.")
    except Exception as e:
        logger.error(f"Error fetching category choices: {e}", exc_info=True)
        flash("An error occurred while fetching category choices.", "danger")
        return render_tool_search_page(session=session, active_tab="categories")

    # Pre-fill form with existing data on GET request
    if request.method == 'GET':
        logger.info(f"Pre-filling form with existing category data: {category.name}")
        category_form.name.data = category.name
        category_form.description.data = category.description
        category_form.parent_id.data = category.parent_id if category.parent_id else 0

    # Process form submission
    if category_form.validate_on_submit():
        logger.info(f"Form validated. Updating category ID {category_id}")
        logger.info(
            f"New data - Name: {category_form.name.data}, Description: {category_form.description.data}, Parent ID: {category_form.parent_id.data}")

        # Update category data
        category.name = category_form.name.data.strip()
        category.description = category_form.description.data.strip()
        category.parent_id = category_form.parent_id.data if category_form.parent_id.data != 0 else None

        try:
            session.commit()
            logger.info(f"Category '{category.name}' updated successfully!")
            flash("Category updated successfully!", "success")
            return redirect(url_for("tool_routes.add_tool_category"))
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating category: {e}", exc_info=True)
            flash("An error occurred while updating the category. Please try again.", "danger")
    else:
        if request.method == 'POST':
            logger.warning("Form validation failed. Errors: %s", category_form.errors)
            flash("Please correct the errors in the form.", "danger")

    logger.info("Rendering edit category template using render_tool_search_page.")
    return render_tool_search_page(
        session=session,
        active_tab="categories",
        category_form=category_form
    )

# NOt_FUNC: not editing category
@tool_blueprint_bp.route('/tool_category/delete_tool_category/<int:category_id>', methods=['POST'])
def delete_tool_category(category_id):
    logger.info(f"Accessing /tool_category/delete_tool_category route for category ID: {category_id}")

    session = db_config.get_main_session()

    try:
        # Get the category to delete
        category = session.query(ToolCategory).get(category_id)
        if not category:
            logger.warning(f"Category with ID {category_id} not found!")
            flash("Category not found.", "danger")
            return render_tool_search_page(session=session, active_tab="categories")

        # Check if category has child categories
        child_categories = session.query(ToolCategory).filter(ToolCategory.parent_id == category_id).count()
        if child_categories > 0:
            logger.warning(f"Cannot delete category '{category.name}' - it has {child_categories} child categories")
            flash(
                f"Cannot delete category '{category.name}' because it has child categories. Please delete or reassign child categories first.",
                "danger")
            return render_tool_search_page(session=session, active_tab="categories")

        # Check if category is being used by any tools
        # Note: You'll need to import your Tool model and adjust this query based on your schema
        # from modules.emtacdb.emtacdb_fts import Tool  # Adjust import as needed
        # tools_using_category = session.query(Tool).filter(Tool.category_id == category_id).count()
        # if tools_using_category > 0:
        #     logger.warning(f"Cannot delete category '{category.name}' - it's used by {tools_using_category} tools")
        #     flash(f"Cannot delete category '{category.name}' because it's being used by {tools_using_category} tools. Please reassign or delete those tools first.", "danger")
        #     return render_tool_search_page(session=session, active_tab="categories")

        # Delete the category
        category_name = category.name  # Store name for logging
        session.delete(category)
        session.commit()

        logger.info(f"Category '{category_name}' deleted successfully!")
        flash(f"Category '{category_name}' deleted successfully!", "success")

    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting category: {e}", exc_info=True)
        flash("An error occurred while deleting the category. Please try again.", "danger")
        return render_tool_search_page(session=session, active_tab="categories")

    finally:
        # Don't close session here since render_tool_search_page might need it
        pass

    # Return to the categories tab with success message
    return render_tool_search_page(session=session, active_tab="categories")


@tool_blueprint_bp.route('/tool_category/confirm_delete/<int:category_id>', methods=['GET'])
def confirm_delete_tool_category(category_id):
    logger.info(f"Accessing confirmation page for deleting category ID: {category_id}")

    session = db_config.get_main_session()

    try:
        category = session.query(ToolCategory).get(category_id)

        if not category:
            logger.warning(f"Category with ID {category_id} not found!")
            flash("Category not found.", "danger")
            return render_tool_search_page(session=session, active_tab="categories")

        # Check for dependencies
        child_categories = session.query(ToolCategory).filter(ToolCategory.parent_id == category_id).count()

        # You can either use the existing confirm template or show a confirmation message
        # Option 1: Use existing template (if you want to keep the separate confirmation page)
        return render_template(
            'tool_templates/confirm_delete_category.html',
            category=category,
            child_categories=child_categories
        )

        # Option 2: Show confirmation within the main page (alternative approach)
        # flash(f"Are you sure you want to delete '{category.name}'? This action cannot be undone.", "warning")
        # return render_tool_search_page(session=session, active_tab="categories")

    except Exception as e:
        logger.error(f"Error accessing category for confirmation: {e}", exc_info=True)
        flash("An error occurred while accessing the category.", "danger")
        return render_tool_search_page(session=session, active_tab="categories")

    finally:
        # Only close session if we're not passing it to render_tool_search_page
        if 'render_tool_search_page' not in locals():
            session.close()


# Alternative version of confirm_delete that uses render_tool_search_page entirely
@tool_blueprint_bp.route('/tool_category/confirm_delete_inline/<int:category_id>', methods=['GET'])
def confirm_delete_tool_category_inline(category_id):
    """
    Alternative approach that shows confirmation within the main tool page
    instead of a separate confirmation template.
    """
    logger.info(f"Accessing inline confirmation for deleting category ID: {category_id}")

    session = db_config.get_main_session()

    try:
        category = session.query(ToolCategory).get(category_id)

        if not category:
            logger.warning(f"Category with ID {category_id} not found!")
            flash("Category not found.", "danger")
        else:
            # Check for dependencies
            child_categories = session.query(ToolCategory).filter(ToolCategory.parent_id == category_id).count()

            if child_categories > 0:
                flash(
                    f"Warning: Category '{category.name}' has {child_categories} child categories. "
                    f"You must delete or reassign child categories before deleting this category.",
                    "warning"
                )
            else:
                flash(
                    f"Are you sure you want to delete category '{category.name}'? This action cannot be undone.",
                    "warning"
                )

        return render_tool_search_page(session=session, active_tab="categories")

    except Exception as e:
        logger.error(f"Error accessing category for confirmation: {e}", exc_info=True)
        flash("An error occurred while accessing the category.", "danger")
        return render_tool_search_page(session=session, active_tab="categories")