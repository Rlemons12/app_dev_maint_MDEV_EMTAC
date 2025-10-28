#tool_get_data.py
from flask import request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from blueprints.tool_routes import tool_blueprint_bp
from modules.emtacdb.emtacdb_fts import Position, ToolPackage, ToolManufacturer,ToolCategory
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger  # Import the logger
from modules.emtacdb.forms  import PositionForm

db_config = DatabaseConfig()

@tool_blueprint_bp.route('/get_tool_positions', methods=['GET'])
def get_tool_positions():
    try:
        with db_config.MainSession() as session:
            positions = session.query(Position).all()  # Assuming `Position` is a defined model
            # Convert data to JSON format
            position_data = [{'id': position.id, 'name': position.name} for position in positions]
            return jsonify(position_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tool_blueprint_bp.route('/get_tool_packages', methods=['GET'])
def get_tool_packages():
    try:
        with db_config.MainSession() as session:
            packages = session.query(ToolPackage).all()
            # Convert data to JSON format
            package_data = [{'id': package.id, 'name': package.name} for package in packages]
            return jsonify(package_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tool_blueprint_bp.route('/get_tool_manufacturers', methods=['GET'])
def get_tool_manufacturers():
    """
    Route to fetch all tool manufacturers.
    Returns:
        JSON response containing a list of tool manufacturers with their IDs and names.
    """
    try:
        with db_config.MainSession() as session:
            # Query all ToolManufacturer records
            manufacturers = session.query(ToolManufacturer).all()
            manufacturers_data = [
                {'id': manufacturer.id, 'name': manufacturer.name}
                for manufacturer in manufacturers
            ]
            # Return the serialized data as JSON with a 200 OK status
            return jsonify(manufacturers_data), 200

    except SQLAlchemyError as e:
        # Log the database error
        logger.error(f"Database error occurred: {str(e)}")

        # Return a JSON error response with a 500 Internal Server Error status
        return jsonify({'error': 'Database error occurred while fetching tool manufacturers.'}), 500

    except Exception as e:
        # Log any unexpected errors
        logger.error(f"Unexpected error occurred: {str(e)}")

        # Return a JSON error response with a 500 Internal Server Error status
        return jsonify({'error': 'An unexpected error occurred while fetching tool manufacturers.'}), 500

@tool_blueprint_bp.route('/get_tool_categories', methods=['GET'])
def get_tool_categories():
    """
    Route to fetch all tool categories.
    Optional Query Parameters:
        - page: The page number for pagination (default: 1)
        - per_page: Number of items per page (default: 10)
    Returns:
        JSON response containing a list of tool categories with their details.
    """
    try:
        # Handle pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        # Create a new database session
        with db_config.MainSession() as session:
            # Query all ToolCategory records
            query = session.query(ToolCategory)
            total = query.count()
            categories = query.offset((page - 1) * per_page).limit(per_page).all()

            # Serialize the data into a list of dictionaries
            categories_data = [
                {
                    'id': category.id,
                    'name': category.name,
                    'description': category.description,
                    'parent_id': category.parent_id,
                    'subcategories': [
                        {'id': sub.id, 'name': sub.name}
                        for sub in category.subcategories
                    ]
                }
                for category in categories
            ]

            # Log successful retrieval
            logger.info(f"Fetched {len(categories_data)} tool categories successfully (Page {page}).")

            # Return the serialized data as JSON with pagination info
            return jsonify({
                'total': total,
                'page': page,
                'per_page': per_page,
                'categories': categories_data
            }), 200

    except ValueError:
        # Log invalid pagination parameters
        logger.error("Invalid pagination parameters provided.")
        return jsonify({'error': 'Invalid pagination parameters.'}), 400

    except SQLAlchemyError as e:
        # Log the database error
        logger.error(f"Database error occurred while fetching tool categories: {str(e)}")

        # Return a JSON error response with a 500 Internal Server Error status
        return jsonify({'error': 'Database error occurred while fetching tool categories.'}), 500

    except Exception as e:
        # Log any unexpected errors
        logger.error(f"Unexpected error occurred while fetching tool categories: {str(e)}")

        # Return a JSON error response with a 500 Internal Server Error status
        return jsonify({'error': 'An unexpected error occurred while fetching tool categories.'}), 500

@tool_blueprint_bp.route('/get_tool_position_associations', methods=['GET', 'POST'])
def get_tool_position_associations():
    form = PositionForm()
    if form.validate_on_submit():
        logger.info('PositionForm submitted with data: %s', form.data)

        session = db_config.get_main_session()
        try:
            # Check if the position already exists
            position = session.query(Position).filter_by(
                area=form.area.data,
                equipment_group=form.equipment_group.data,
                model=form.model.data,
                asset_number=form.asset_number.data,
                location=form.location.data,
                subassembly=form.subassembly.data,
                component_assembly=form.component_assembly.data,
                assembly_view=form.assembly_view.data,
                site_location=form.site_location.data
            ).first()

            if not position:
                # Create a new Position
                position = Position(
                    area=form.area.data,
                    equipment_group=form.equipment_group.data,
                    model=form.model.data,
                    asset_number=form.asset_number.data,
                    location=form.location.data,
                    subassembly=form.subassembly.data,
                    component_assembly=form.component_assembly.data,
                    assembly_view=form.assembly_view.data,
                    site_location=form.site_location.data
                )
                session.add(position)
                session.commit()
                logger.info('New Position created with ID: %s', position.id)

            # Get Tool ID from query parameters
            tool_id = request.args.get('tool_id')
            if not tool_id:
                flash('Tool ID is required!', 'danger')
                logger.warning('Tool ID is missing in the request.')
                return redirect(request.url)

            # Create ToolPositionAssociation
            tool_position_association = ToolPositionAssociation(
                tool_id=tool_id,
                position_id=position.id,
                description=form.position_description.data
            )
            session.add(tool_position_association)
            session.commit()
            logger.info('ToolPositionAssociation created with Tool ID: %s and Position ID: %s', tool_id, position.id)

            flash('Tool Position Association created successfully!', 'success')
            return redirect('/get_tool_position_associations')

        except Exception as e:
            session.rollback()
            logger.error('Error while processing form: %s', str(e), exc_info=True)
            flash('An error occurred while processing the request.', 'danger')
        finally:
            session.close()

    return render_template('tool_position_association.html', position_form=form)


