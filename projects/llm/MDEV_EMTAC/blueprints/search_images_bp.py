import time
from flask import Blueprint, jsonify, request, flash
from modules.emtacdb.emtacdb_fts import Image
from modules.configuration.log_config import logger
from modules.configuration.config_env import DatabaseConfig

db_config = DatabaseConfig()

search_images_bp = Blueprint('search_images_bp', __name__)

@search_images_bp.route('/serve_image/<int:image_id>')
def serve_image_route(image_id):
    """
    Route to serve an image file by ID using the Image.serve_file class method.
    """
    logger.debug(f"Request to serve image with ID: {image_id}")

    try:
        # Use the Image.serve_file class method
        success, response, status_code = Image.serve_file(image_id)

        if success:
            return response
        else:
            logger.error(f"Failed to serve image {image_id}: {response}")
            # Remove flash for API - just return error response
            return response, status_code

    except Exception as e:
        logger.error(f"Error serving image {image_id}: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@search_images_bp.route('/', methods=['GET'])
@search_images_bp.route('', methods=['GET'])  # Handle both with/without trailing slash
def search_images():
    """
    Enhanced route to search for images using the enhanced search method.
    Now finds images associated through both direct positions AND completed documents.
    """
    with db_config.get_main_session() as session:
        try:
            # Extract and validate search parameters from request
            search_params = _extract_search_parameters(request)

            # Log the search request with more detail
            logger.info(f"Image search initiated with {len(search_params)} parameters")
            logger.debug(f"Search parameters: {search_params}")

            # Validate parameters before searching
            validation_errors = _validate_search_parameters(search_params)
            if validation_errors:
                logger.warning(f"Parameter validation errors: {validation_errors}")
                return jsonify({
                    "error": "Invalid search parameters",
                    "validation_errors": validation_errors,
                    "images": [],
                    "count": 0
                }), 400

            # Use the ENHANCED Image.search_images method
            start_time = time.time()

            # CHANGE THIS LINE - use the enhanced method
            images = Image.search_images(session, **search_params)

            search_duration = time.time() - start_time

            # Prepare response
            response_data = {
                "images": images,
                "count": len(images),
                "search_params": search_params,
                "search_duration_ms": round(search_duration * 1000, 2),
                "limit": search_params.get('limit', 50),
                "search_method": "enhanced_with_documents"  # Add this to indicate new method
            }

            if images:
                logger.info(f"Found {len(images)} images matching criteria in {search_duration:.3f}s")

                # Add pagination info if at limit
                if len(images) == search_params.get('limit', 50):
                    response_data[
                        "pagination_note"] = f"Results limited to {search_params.get('limit', 50)}. There may be more results available."

                # Add search summary
                response_data["search_summary"] = _generate_search_summary(search_params, len(images))

            else:
                logger.info("No images found matching the criteria")
                response_data["message"] = "No images found matching the search criteria"
                response_data["suggestions"] = _generate_search_suggestions(search_params)

            return jsonify(response_data)

        except ValueError as e:
            logger.error(f"Validation error in search_images route: {e}")
            return jsonify({
                "error": "Invalid parameter values",
                "details": str(e),
                "images": [],
                "count": 0
            }), 400

        except Exception as e:
            logger.error(f"Unexpected error in search_images route: {e}", exc_info=True)
            return jsonify({
                "error": "An internal error occurred while searching images",
                "error_id": getattr(e, 'request_id', None),
                "images": [],
                "count": 0
            }), 500


def _validate_search_parameters(params):
    """
    Validate search parameters and return any validation errors.

    Args:
        params: Dictionary of search parameters

    Returns:
        List of validation error messages, empty if no errors
    """
    errors = []

    # Check ID parameters are positive integers
    id_params = [
        'position_id', 'tool_id', 'task_id', 'problem_id', 'completed_document_id',
        'area_id', 'equipment_group_id', 'model_id', 'asset_number_id', 'location_id',
        'subassembly_id', 'component_assembly_id', 'assembly_view_id', 'site_location_id'
    ]

    for param in id_params:
        if param in params:
            if not isinstance(params[param], int) or params[param] <= 0:
                errors.append(f"{param} must be a positive integer")

    # Validate limit
    if 'limit' in params:
        if not isinstance(params['limit'], int) or params['limit'] < 1:
            errors.append("limit must be a positive integer")
        elif params['limit'] > 1000:
            errors.append("limit cannot exceed 1000")

    # Validate text parameters
    text_params = ['title', 'description']
    for param in text_params:
        if param in params:
            if not isinstance(params[param], str) or len(params[param].strip()) == 0:
                errors.append(f"{param} must be a non-empty string")
            elif len(params[param]) > 500:
                errors.append(f"{param} cannot exceed 500 characters")

    return errors


def _generate_search_summary(search_params, result_count):
    """
    Generate a human-readable summary of the search performed.

    Args:
        search_params: Dictionary of search parameters used
        result_count: Number of results found

    Returns:
        String summary of the search
    """
    summary_parts = []

    if 'title' in search_params:
        summary_parts.append(f"title containing '{search_params['title']}'")
    if 'description' in search_params:
        summary_parts.append(f"description containing '{search_params['description']}'")

    # Hierarchy searches
    hierarchy_map = {
        'area_id': 'area',
        'equipment_group_id': 'equipment group',
        'model_id': 'model',
        'asset_number_id': 'asset number',
        'location_id': 'location'
    }

    for param, name in hierarchy_map.items():
        if param in search_params:
            summary_parts.append(f"in {name} {search_params[param]}")

    # Direct associations
    association_map = {
        'position_id': 'position',
        'tool_id': 'tool',
        'task_id': 'task',
        'problem_id': 'problem',
        'completed_document_id': 'completed document'
    }

    for param, name in association_map.items():
        if param in search_params:
            summary_parts.append(f"associated with {name} {search_params[param]}")

    if summary_parts:
        return f"Found {result_count} images {' and '.join(summary_parts)}"
    else:
        return f"Found {result_count} recent images (no specific criteria)"


def _generate_search_suggestions(search_params):
    """
    Generate helpful suggestions when no results are found.

    Args:
        search_params: Dictionary of search parameters used

    Returns:
        List of suggestion strings
    """
    suggestions = []

    if any(param in search_params for param in ['title', 'description']):
        suggestions.append("Try using broader or fewer keywords")
        suggestions.append("Check spelling and try alternative terms")

    if any(param in search_params for param in ['area_id', 'equipment_group_id', 'model_id']):
        suggestions.append("Try searching at a higher level in the hierarchy (e.g., area instead of model)")

    if len(search_params) > 3:
        suggestions.append("Try using fewer filters to broaden the search")

    if not suggestions:
        suggestions = [
            "Try using text search with keywords from image titles or descriptions",
            "Browse by area or equipment group to find relevant images",
            "Check if the referenced entities (positions, tools, etc.) exist and have associated images"
        ]

    return suggestions


def _extract_search_parameters(request):
    """
    Helper function to extract and convert search parameters from Flask request.

    Args:
        request: Flask request object

    Returns:
        Dictionary of search parameters for Image.search_images
    """
    params = {}

    # Text-based searches
    description = request.args.get('description', '').strip()
    if description:
        params['description'] = description

    title = request.args.get('title', '').strip()
    if title:
        params['title'] = title

    # Direct ID searches
    position_id = request.args.get('position_id')
    if position_id:
        try:
            params['position_id'] = int(position_id)
        except ValueError:
            logger.warning(f"Invalid position_id: {position_id}")

    tool_id = request.args.get('tool_id')
    if tool_id:
        try:
            params['tool_id'] = int(tool_id)
        except ValueError:
            logger.warning(f"Invalid tool_id: {tool_id}")

    task_id = request.args.get('task_id')
    if task_id:
        try:
            params['task_id'] = int(task_id)
        except ValueError:
            logger.warning(f"Invalid task_id: {task_id}")

    problem_id = request.args.get('problem_id')
    if problem_id:
        try:
            params['problem_id'] = int(problem_id)
        except ValueError:
            logger.warning(f"Invalid problem_id: {problem_id}")

    completed_document_id = request.args.get('completed_document_id')
    if completed_document_id:
        try:
            params['completed_document_id'] = int(completed_document_id)
        except ValueError:
            logger.warning(f"Invalid completed_document_id: {completed_document_id}")

    # Hierarchy-based searches (supporting both old and new parameter names)
    area_id = request.args.get('area_id') or request.args.get('searchimage_area') or request.args.get('area')
    if area_id:
        try:
            params['area_id'] = int(area_id)
        except ValueError:
            logger.warning(f"Invalid area_id: {area_id}")

    equipment_group_id = (request.args.get('equipment_group_id') or
                          request.args.get('searchimage_equipment_group') or
                          request.args.get('equipment_group'))
    if equipment_group_id:
        try:
            params['equipment_group_id'] = int(equipment_group_id)
        except ValueError:
            logger.warning(f"Invalid equipment_group_id: {equipment_group_id}")

    model_id = request.args.get('model_id') or request.args.get('searchimage_model') or request.args.get('model')
    if model_id:
        try:
            params['model_id'] = int(model_id)
        except ValueError:
            logger.warning(f"Invalid model_id: {model_id}")

    asset_number_id = (request.args.get('asset_number_id') or
                       request.args.get('searchimage_asset_number') or
                       request.args.get('asset_number'))
    if asset_number_id:
        try:
            params['asset_number_id'] = int(asset_number_id)
        except ValueError:
            logger.warning(f"Invalid asset_number_id: {asset_number_id}")

    location_id = (request.args.get('location_id') or
                   request.args.get('searchimage_location') or
                   request.args.get('location'))
    if location_id:
        try:
            params['location_id'] = int(location_id)
        except ValueError:
            logger.warning(f"Invalid location_id: {location_id}")

    subassembly_id = request.args.get('subassembly_id')
    if subassembly_id:
        try:
            params['subassembly_id'] = int(subassembly_id)
        except ValueError:
            logger.warning(f"Invalid subassembly_id: {subassembly_id}")

    component_assembly_id = request.args.get('component_assembly_id')
    if component_assembly_id:
        try:
            params['component_assembly_id'] = int(component_assembly_id)
        except ValueError:
            logger.warning(f"Invalid component_assembly_id: {component_assembly_id}")

    assembly_view_id = request.args.get('assembly_view_id')
    if assembly_view_id:
        try:
            params['assembly_view_id'] = int(assembly_view_id)
        except ValueError:
            logger.warning(f"Invalid assembly_view_id: {assembly_view_id}")

    site_location_id = request.args.get('site_location_id')
    if site_location_id:
        try:
            params['site_location_id'] = int(site_location_id)
        except ValueError:
            logger.warning(f"Invalid site_location_id: {site_location_id}")

    # Limit parameter
    limit = request.args.get('limit', '50')
    try:
        params['limit'] = int(limit)
        # Ensure reasonable limits
        if params['limit'] > 1000:
            params['limit'] = 1000
        elif params['limit'] < 1:
            params['limit'] = 50
    except ValueError:
        logger.warning(f"Invalid limit: {limit}, using default 50")
        params['limit'] = 50

    return params


# Additional utility routes that might be useful

@search_images_bp.route('/image/<int:image_id>/details', methods=['GET'])
def get_image_details(image_id):
    """
    Route to get detailed information about a specific image including all associations.
    """
    with db_config.get_main_session() as session:
        try:
            # Direct query - no need for search_images call
            image = session.query(Image).filter_by(id=image_id).first()

            if image:
                # Get associations using the helper method
                associations = Image._get_image_associations(session, image_id)

                image_details = {
                    "id": image.id,
                    "title": image.title,
                    "description": image.description,
                    "file_path": image.file_path,
                    "img_metadata": getattr(image, 'img_metadata', None),
                    "view_url": f"/add_document/image/{image.id}",
                    "associations": associations
                }

                return jsonify(image_details)
            else:
                return jsonify({"error": "Image not found"}), 404

        except Exception as e:
            logger.error(f"Error getting image details for ID {image_id}: {e}")
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@search_images_bp.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "service": "search_images_bp"
    })