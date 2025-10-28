from flask import request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload
from blueprints.tool_routes import tool_blueprint_bp
from modules.emtacdb.emtacdb_fts import Tool, ToolImageAssociation, Image
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger

db_config = DatabaseConfig()

@tool_blueprint_bp.route('/tools_search', methods=['GET'])
def tools_search():
    """
    Endpoint to search tools based on query parameters, including associated images.
    Implements pagination and optimized querying.
    """
    try:
        # Extract query parameters
        name = request.args.get('name', type=str, default=None)
        material = request.args.get('material', type=str, default=None)
        category_id = request.args.get('category_id', type=int, default=None)
        manufacturer_id = request.args.get('manufacturer_id', type=int, default=None)
        page = request.args.get('page', type=int, default=1)
        per_page = request.args.get('per_page', type=int, default=10)

        logger.debug(f"Search parameters - Name: {name}, Category ID: {category_id}, "
                     f"Manufacturer ID: {manufacturer_id}, Page: {page}, Per Page: {per_page}")

        with db_config.get_main_session() as session:
            logger.info("Building base query for tool search with eager loading.")
            # Build the base query with eager loading to prevent N+1 queries
            query = session.query(Tool).options(
                joinedload(Tool.tool_category),
                joinedload(Tool.tool_manufacturer),
                joinedload(Tool.tool_image_association).joinedload(ToolImageAssociation.image)
            )
            logger.debug("Base query constructed.")

            # Apply filters based on query parameters
            if name:
                query = query.filter(Tool.name.ilike(f'%{name}%'))
                logger.debug(f"Applied filter: Tool name LIKE '%{name}%'")
            if material:
                logger.debug(f"Applied filter: Tool material LIKE '%{material}%'")
                query = query.filter(Tool.material.ilike(f'%{material}%'))
            if category_id:
                query = query.filter(Tool.tool_category_id == category_id)
                logger.debug(f"Applied filter: Tool category ID == {category_id}")
            if manufacturer_id:
                query = query.filter(Tool.tool_manufacturer_id == manufacturer_id)
                logger.debug(f"Applied filter: Tool manufacturer ID == {manufacturer_id}")

            total = query.count()
            logger.info(f"Total tools found after applying filters: {total}")
            tools = query.offset((page - 1) * per_page).limit(per_page).all()
            logger.info(f"Retrieved {len(tools)} tools for page {page} with per_page {per_page}")

            # Prepare response data
            tool_data = []
            for tool in tools:
                logger.debug(f"Processing tool ID: {tool.id}, Name: {tool.name}")
                images = []
                for assoc in tool.tool_image_association:
                    if assoc.image:
                        logger.debug(f"Tool ID {tool.id}: Found associated image with ID {assoc.image.id}")
                        images.append({
                            'id': assoc.image.id,
                            'title': assoc.image.title,
                            'description': assoc.image.description,
                            'file_path': assoc.image.file_path,
                        })
                        logger.info(f'image title {assoc.image.title}')
                        logger.info(f'image description {assoc.image.description}')
                        logger.info(f'image file path {assoc.image.file_path}')
                    else:
                        logger.warning(f"Tool ID {tool.id}: Association found without image.")
                tool_data.append({
                    'id': tool.id,
                    'name': tool.name,
                    'size': tool.size,
                    'type': tool.type,
                    'material': tool.material,
                    'description': tool.description,
                    'category': tool.tool_category.name if tool.tool_category else None,
                    'manufacturer': tool.tool_manufacturer.name if tool.tool_manufacturer else None,
                    'images': images,
                })
                logger.debug(f"Appended tool data for tool ID: {tool.id}")

            response = {
                'total': total,
                'page': page,
                'per_page': per_page,
                'tools': tool_data
            }
            logger.info("Tool search response prepared successfully.")
            return jsonify(response), 200

    except SQLAlchemyError as e:
        logger.error(f"Database error occurred during tool search: {e}", exc_info=True)
        return jsonify({"error": "Database error occurred.", "details": str(e)}), 500

    except Exception as e:
        logger.error(f"Unexpected error occurred during tool search: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500
