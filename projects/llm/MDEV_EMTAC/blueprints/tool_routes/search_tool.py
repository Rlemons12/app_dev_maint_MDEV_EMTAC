# blueprints/tool_routes/search_tools.py

from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from modules.tool_module.forms import (ToolSearchForm,ToolManufacturerForm,ToolCategoryForm,ToolForm)
from modules.emtacdb.emtacdb_fts import Tool, ToolCategory, ToolManufacturer
from modules.emtacdb.forms  import PositionForm
from modules.configuration.log_config import logger
from modules.configuration.config_env import DatabaseConfig
from blueprints.tool_routes import tool_blueprint_bp

# Instantiate DatabaseConfig
db_config = DatabaseConfig()

@tool_blueprint_bp.route('/search_tools', methods=['GET', 'POST'])
def search_tools():
    logger.info("Accessed /search_tools.")

    try:
        # Obtain a session from DatabaseConfig
        session = db_config.get_main_session()
        logger.debug("Database session obtained.")

        # Initialize all forms
        tool_form = ToolSearchForm(request.form)
        position_form = PositionForm(request.form)
        manufacturer_form = ToolManufacturerForm(request.form)
        category_form = ToolCategoryForm(request.form)
        logger.debug("All forms initialized.")

        # Fetch existing manufacturers and categories for display in other tabs
        manufacturers = session.query(ToolManufacturer).order_by(ToolManufacturer.name).all()
        categories = session.query(ToolCategory).order_by(ToolCategory.name).all()
        logger.debug(f"Fetched {len(manufacturers)} manufacturers and {len(categories)} categories.")

        tools = []
        page = request.args.get('page', 1, type=int)
        per_page = 20  # Adjust as needed
        total_pages = 0  # Initialize with default

        if request.method == 'POST':
            logger.info("Received POST request for tool search.")
            if tool_form.validate():
                logger.info("Form validation successful.")

                # Start building the query
                query = session.query(Tool)
                logger.debug("Started building the query.")

                # Apply filters based on form data
                if tool_form.tool_name.data:
                    logger.debug(f"Applying filter: Tool Name contains '{tool_form.tool_name.data}'.")
                    query = query.filter(Tool.name.ilike(f"%{tool_form.tool_name.data}%"))
                if tool_form.tool_size.data:
                    logger.debug(f"Applying filter: Tool Size contains '{tool_form.tool_size.data}'.")
                    query = query.filter(Tool.size.ilike(f"%{tool_form.tool_size.data}%"))
                if tool_form.tool_type.data:
                    logger.debug(f"Applying filter: Tool Type contains '{tool_form.tool_type.data}'.")
                    query = query.filter(Tool.type.ilike(f"%{tool_form.tool_type.data}%"))
                if tool_form.tool_material.data:
                    logger.debug(f"Applying filter: Tool Material contains '{tool_form.tool_material.data}'.")
                    query = query.filter(Tool.material.ilike(f"%{tool_form.tool_material.data}%"))
                if tool_form.tool_category.data:
                    logger.debug(f"Applying filter: Tool Category IDs {tool_form.tool_category.data}.")
                    query = query.filter(Tool.category_id.in_(tool_form.tool_category.data))
                if tool_form.tool_manufacturer.data:
                    logger.debug(f"Applying filter: Tool Manufacturer IDs {tool_form.tool_manufacturer.data}.")
                    query = query.filter(Tool.manufacturer_id.in_(tool_form.tool_manufacturer.data))

                # Calculate total tools and total pages
                total_tools = query.count()
                total_pages = (total_tools + per_page - 1) // per_page
                logger.info(f"Total tools found: {total_tools}. Total pages: {total_pages}.")

                # Apply pagination
                tools = query.order_by(Tool.name).offset((page - 1) * per_page).limit(per_page).all()
                logger.debug(f"Retrieved {len(tools)} tools for page {page}.")

            else:
                logger.warning("Form validation failed.")
                flash("Please correct the errors in the form.", "danger")
        else:
            logger.info("Received GET request for tool search. No search performed.")
            # Optionally, you can populate default search results here or leave it empty

        return render_template(
            'tool_templates/tool_search_entry.html',
            tool_form=tool_form,
            position_form=position_form,
            manufacturer_form=manufacturer_form,
            category_form=category_form,
            tools=tools,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            manufacturers=manufacturers,
            categories=categories
        )

    except Exception as e:
        logger.exception(f"An error occurred in /search_tools route: {e}")
        flash("An unexpected error occurred. Please try again later.", "danger")
        return redirect(url_for('tool_routes.submit_tool_data'))
