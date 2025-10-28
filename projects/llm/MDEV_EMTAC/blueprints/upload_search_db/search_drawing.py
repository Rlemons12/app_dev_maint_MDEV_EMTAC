# blueprints/upload_search_db/search_drawing.py

from modules.configuration.log_config import get_request_id
from flask import Blueprint, request, jsonify
from modules.emtacdb.emtacdb_fts import Drawing
from modules.configuration.config import DATABASE_URL
from modules.configuration.log_config import logger, debug_id, error_id, info_id, log_timed_operation
from modules.configuration.config_env import DatabaseConfig

# Create a blueprint for drawing routes
drawing_routes = Blueprint('drawing_routes', __name__)


@drawing_routes.route('/drawings/search', methods=['GET'])
def search_drawings():
    """
    Search endpoint for drawings

    Query Parameters:
        search_text (str): Text to search across multiple fields
        fields (List[str]): Fields to search in (comma-separated)
        exact_match (bool): Whether to perform exact matching
        drawing_id (int): ID to filter by
        drw_equipment_name (str): Equipment name to filter by
        drw_number (str): Drawing number to filter by
        drw_name (str): Drawing name to filter by
        drw_revision (str): Revision to filter by
        drw_spare_part_number (str): Spare part number to filter by
        drw_type (str): Drawing type to filter by (e.g., 'Electrical', 'Mechanical')
        file_path (str): File path to filter by
        limit (int): Maximum number of results (default 100)
        include_part_images (bool): Whether to include associated part images (default false)

    Returns:
        JSON response with list of matching drawings and their associated part images
    """
    # Get request ID for tracking
    request_id = get_request_id()
    debug_id(f"Starting search_drawings endpoint request", request_id)

    # Extract and validate query parameters
    search_text = request.args.get('search_text')
    debug_id(f"Search text: {search_text}", request_id)

    # Handle fields as a comma-separated list
    fields_param = request.args.get('fields')
    fields = fields_param.split(',') if fields_param else None
    debug_id(f"Search fields: {fields}", request_id)

    # Convert exact_match to boolean
    exact_match_param = request.args.get('exact_match', 'false').lower()
    exact_match = exact_match_param in ('true', 'yes', '1')
    debug_id(f"Exact match: {exact_match}", request_id)

    # Convert include_part_images to boolean
    include_images_param = request.args.get('include_part_images', 'false').lower()
    include_part_images = include_images_param in ('true', 'yes', '1')
    debug_id(f"Include part images: {include_part_images}", request_id)

    # Convert numeric parameters
    drawing_id = None
    if 'drawing_id' in request.args and request.args.get('drawing_id').strip():
        try:
            drawing_id = int(request.args.get('drawing_id'))
            debug_id(f"Drawing ID: {drawing_id}", request_id)
        except ValueError:
            error_id(f"Invalid drawing_id parameter: {request.args.get('drawing_id')}", request_id)
            return jsonify({
                'error': 'Invalid drawing_id parameter',
                'message': 'drawing_id must be an integer'
            }), 400

    # Get string parameters and only use them if they're not empty
    drw_equipment_name = request.args.get('drw_equipment_name')
    drw_equipment_name = drw_equipment_name if drw_equipment_name and drw_equipment_name.strip() else None

    drw_number = request.args.get('drw_number')
    drw_number = drw_number if drw_number and drw_number.strip() else None

    drw_name = request.args.get('drw_name')
    drw_name = drw_name if drw_name and drw_name.strip() else None

    drw_revision = request.args.get('drw_revision')
    drw_revision = drw_revision if drw_revision and drw_revision.strip() else None

    # Handle drawing type parameter with validation
    drw_type = request.args.get('drw_type')
    drw_type = drw_type if drw_type and drw_type.strip() else None

    # Validate drawing type if provided
    if drw_type:
        try:
            available_types = Drawing.get_available_types()
            if drw_type not in available_types:
                error_id(f"Invalid drawing type: {drw_type}. Available types: {available_types}", request_id)
                return jsonify({
                    'error': 'Invalid drawing type',
                    'message': f'Valid types are: {", ".join(available_types)}',
                    'available_types': available_types
                }), 400
            debug_id(f"Drawing type: {drw_type}", request_id)
        except Exception as e:
            error_id(f"Error validating drawing type: {str(e)}", request_id)
            # Continue without validation if there's an error getting available types

    # Special handling for spare part number parameter
    spare_part_param = request.args.get('drw_spare_part_number')
    spare_part_param = spare_part_param if spare_part_param and spare_part_param.strip() else None

    # Also check if spare part is being searched via search_text and fields
    if search_text and fields and 'drw_spare_part_number' in fields:
        # Use search_text as spare part if not already provided
        if not spare_part_param and search_text.strip():
            spare_part_param = search_text.strip()
            search_text = None  # Clear search_text to avoid duplicate search

    # Normalize spare part number for better searching
    drw_spare_part_number = None
    if spare_part_param:
        # Remove common separators and normalize
        drw_spare_part_number = spare_part_param.replace('-', '').replace(' ', '').replace('_', '')
        debug_id(f"Normalized spare part number for search: {drw_spare_part_number}", request_id)
        # Always use partial matching for spare part numbers
        exact_match = False

    file_path = request.args.get('file_path')
    file_path = file_path if file_path and file_path.strip() else None

    # Log string parameters if present
    params = {
        'drw_equipment_name': drw_equipment_name,
        'drw_number': drw_number,
        'drw_name': drw_name,
        'drw_revision': drw_revision,
        'drw_spare_part_number': drw_spare_part_number,
        'drw_type': drw_type,
        'file_path': file_path
    }
    # Filter out None values for cleaner logging
    logged_params = {k: v for k, v in params.items() if v is not None}
    if logged_params:
        debug_id(f"String parameters: {logged_params}", request_id)

    # Get and validate limit
    limit = 100
    if 'limit' in request.args:
        try:
            limit = int(request.args.get('limit'))
            debug_id(f"Limit: {limit}", request_id)
            if limit <= 0:
                error_id(f"Invalid limit parameter: {limit} (must be positive)", request_id)
                return jsonify({
                    'error': 'Invalid limit parameter',
                    'message': 'limit must be a positive integer'
                }), 400
        except ValueError:
            error_id(f"Invalid limit parameter: {request.args.get('limit')} (not an integer)", request_id)
            return jsonify({
                'error': 'Invalid limit parameter',
                'message': 'limit must be an integer'
            }), 400

    # Get database session
    db_config = DatabaseConfig()
    session = None

    try:
        # Create a database session
        session = db_config.get_main_session()
        debug_id(f"Created database session for search_drawings endpoint", request_id)

        # Use the timed operation context manager with request ID
        with log_timed_operation("search_drawings.Drawing.search", request_id):
            # Import models here to avoid circular imports
            from modules.emtacdb.emtacdb_fts import Drawing, DrawingPartAssociation, PartsPositionImageAssociation, \
                Image
            from sqlalchemy import func, or_

            # SPECIAL HANDLING FOR SPARE PART SEARCHES
            if drw_spare_part_number:
                debug_id(f"Using custom spare part search for: {drw_spare_part_number}", request_id)
                # Create a query with custom SQL for more flexible spare part matching
                query = session.query(Drawing)

                # Apply drawing type filter if provided
                if drw_type:
                    query = query.filter(Drawing.drw_type == drw_type)
                    debug_id(f"Applied drawing type filter: {drw_type}", request_id)

                # Try various search patterns for spare part numbers
                spare_part_patterns = [
                    f"%{drw_spare_part_number}%",  # Contains the normalized number
                    f"{drw_spare_part_number}%",  # Starts with the normalized number
                    f"%{drw_spare_part_number}"  # Ends with the normalized number
                ]

                # If the number is longer than 5 chars, also try partial matches
                if len(drw_spare_part_number) > 5:
                    # Use the last 5 chars (often these are the most specific part of a part number)
                    spare_part_patterns.append(f"%{drw_spare_part_number[-5:]}%")

                # Build OR conditions for flexible part number matching
                conditions = []
                for pattern in spare_part_patterns:
                    # Basic LIKE search (case insensitive)
                    conditions.append(func.lower(Drawing.drw_spare_part_number).like(func.lower(pattern)))

                    # Also search for versions with common separators removed
                    conditions.append(
                        func.lower(
                            func.replace(
                                func.replace(
                                    func.replace(Drawing.drw_spare_part_number, '-', ''),
                                    ' ', ''),
                                '_', '')
                        ).like(func.lower(pattern))
                    )

                # Apply the OR conditions
                query = query.filter(or_(*conditions))

                # Apply limit
                query = query.limit(limit)

                # Execute query
                results = query.all()
                debug_id(f"Custom spare part search found {len(results)} results", request_id)
            else:
                # For non-spare part searches, use the standard search method
                results = Drawing.search(
                    search_text=search_text,
                    fields=fields,
                    exact_match=exact_match,
                    drawing_id=drawing_id,
                    drw_equipment_name=drw_equipment_name,
                    drw_number=drw_number,
                    drw_name=drw_name,
                    drw_revision=drw_revision,
                    drw_spare_part_number=drw_spare_part_number,
                    drw_type=drw_type,  # Pass the drawing type parameter
                    file_path=file_path,
                    limit=limit,
                    request_id=request_id,
                    session=session
                )

            debug_id(f"Drawing search completed, found {len(results)} results", request_id)

            # Convert results to JSON-serializable format
            drawings_data = []
            for drawing in results:
                drawing_data = {
                    'id': drawing.id,
                    'drw_equipment_name': drawing.drw_equipment_name,
                    'drw_number': drawing.drw_number,
                    'drw_name': drawing.drw_name,
                    'drw_revision': drawing.drw_revision,
                    'drw_spare_part_number': drawing.drw_spare_part_number,
                    'drw_type': drawing.drw_type,  # Include drawing type in response
                    'file_path': drawing.file_path
                }

                # Add part images if requested
                if include_part_images:
                    debug_id(f"Fetching part images for drawing ID {drawing.id}", request_id)

                    # Get associated parts for this drawing
                    with log_timed_operation(f"search_drawings.get_parts_by_drawing.{drawing.id}", request_id):
                        parts = DrawingPartAssociation.get_parts_by_drawing(
                            drawing_id=drawing.id,
                            request_id=request_id,
                            session=session
                        )

                    debug_id(f"Found {len(parts)} parts for drawing ID {drawing.id}", request_id)

                    # Find images for these parts
                    part_images = []
                    for part in parts:
                        debug_id(f"Getting images for part ID {part.id}", request_id)

                        # Find images via PartsPositionImageAssociation
                        with log_timed_operation(f"search_drawings.part_image_search.{part.id}", request_id):
                            associations = PartsPositionImageAssociation.search(
                                session=session,
                                part_id=part.id
                            )

                        debug_id(f"Found {len(associations)} image associations for part ID {part.id}", request_id)

                        for assoc in associations:
                            if assoc.image_id:
                                # Get image details
                                with log_timed_operation(f"search_drawings.serve_image.{assoc.image_id}", request_id):
                                    image_data = Image.serve_image(
                                        image_id=assoc.image_id,
                                        request_id=request_id,
                                        session=session
                                    )

                                if image_data:
                                    debug_id(f"Adding image ID {image_data['id']} to results", request_id)
                                    part_images.append({
                                        'part_id': part.id,
                                        'part_number': part.part_number if hasattr(part, 'part_number') else None,
                                        'part_name': part.name if hasattr(part, 'name') else None,
                                        'image_id': image_data['id'],
                                        'image_title': image_data['title'],
                                        'image_path': image_data['file_path'],
                                        'image_url': f"/images/{image_data['id']}"
                                    })

                    drawing_data['part_images'] = part_images
                    debug_id(f"Added {len(part_images)} part images to drawing ID {drawing.id}", request_id)

                drawings_data.append(drawing_data)

            info_id(f"search_drawings endpoint completed successfully with {len(drawings_data)} results", request_id)
            return jsonify({
                'count': len(drawings_data),
                'results': drawings_data
            })

    except Exception as e:
        # Log the error with request_id
        error_id(f"Error in search_drawings endpoint: {str(e)}", request_id, exc_info=True)

        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred while processing your request'
        }), 500
    finally:
        # Close the session if it was created
        if session:
            session.close()
            debug_id(f"Closed database session for search_drawings endpoint", request_id)


@drawing_routes.route('/drawings/types', methods=['GET'])
def get_drawing_types():
    """
    Get all available drawing types

    Returns:
        JSON response with list of available drawing types
    """
    request_id = get_request_id()
    debug_id(f"Starting get_drawing_types endpoint request", request_id)

    try:
        available_types = Drawing.get_available_types()
        info_id(f"get_drawing_types endpoint completed successfully", request_id)
        return jsonify({
            'available_types': available_types,
            'count': len(available_types)
        })
    except Exception as e:
        error_id(f"Error in get_drawing_types endpoint: {str(e)}", request_id, exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred while retrieving drawing types'
        }), 500


@drawing_routes.route('/drawings/search/by-type/<drawing_type>', methods=['GET'])
def search_drawings_by_type(drawing_type):
    """
    Search drawings by specific type

    Path Parameters:
        drawing_type (str): The type of drawing to search for

    Query Parameters:
        limit (int): Maximum number of results (default 100)

    Returns:
        JSON response with list of drawings of the specified type
    """
    request_id = get_request_id()
    debug_id(f"Starting search_drawings_by_type endpoint for type: {drawing_type}", request_id)

    # Validate drawing type
    try:
        available_types = Drawing.get_available_types()
        if drawing_type not in available_types:
            error_id(f"Invalid drawing type: {drawing_type}", request_id)
            return jsonify({
                'error': 'Invalid drawing type',
                'message': f'Valid types are: {", ".join(available_types)}',
                'available_types': available_types
            }), 400
    except Exception as e:
        error_id(f"Error validating drawing type: {str(e)}", request_id)
        return jsonify({
            'error': 'Internal server error',
            'message': 'Error validating drawing type'
        }), 500

    # Get and validate limit
    limit = 100
    if 'limit' in request.args:
        try:
            limit = int(request.args.get('limit'))
            if limit <= 0:
                error_id(f"Invalid limit parameter: {limit}", request_id)
                return jsonify({
                    'error': 'Invalid limit parameter',
                    'message': 'limit must be a positive integer'
                }), 400
        except ValueError:
            error_id(f"Invalid limit parameter: {request.args.get('limit')}", request_id)
            return jsonify({
                'error': 'Invalid limit parameter',
                'message': 'limit must be an integer'
            }), 400

    session = None
    try:
        # Get database session
        db_config = DatabaseConfig()
        session = db_config.get_main_session()

        # Search for drawings by type
        with log_timed_operation(f"search_drawings_by_type.{drawing_type}", request_id):
            results = Drawing.search_by_type(
                drawing_type=drawing_type,
                request_id=request_id,
                session=session
            )

            # Apply limit manually since search_by_type doesn't have limit parameter
            if len(results) > limit:
                results = results[:limit]

        # Convert results to JSON format
        drawings_data = []
        for drawing in results:
            drawings_data.append({
                'id': drawing.id,
                'drw_equipment_name': drawing.drw_equipment_name,
                'drw_number': drawing.drw_number,
                'drw_name': drawing.drw_name,
                'drw_revision': drawing.drw_revision,
                'drw_spare_part_number': drawing.drw_spare_part_number,
                'drw_type': drawing.drw_type,
                'file_path': drawing.file_path
            })

        info_id(f"search_drawings_by_type endpoint completed with {len(drawings_data)} results", request_id)
        return jsonify({
            'drawing_type': drawing_type,
            'count': len(drawings_data),
            'results': drawings_data
        })

    except Exception as e:
        error_id(f"Error in search_drawings_by_type endpoint: {str(e)}", request_id, exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred while searching drawings'
        }), 500
    finally:
        if session:
            session.close()
            debug_id(f"Closed database session for search_drawings_by_type endpoint", request_id)