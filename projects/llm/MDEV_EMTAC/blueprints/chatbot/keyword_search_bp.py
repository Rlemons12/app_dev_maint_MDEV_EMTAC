from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from modules.emtacdb.emtacdb_fts import KeywordSearch
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger
import os
import tempfile
import json
from functools import wraps

# Create the blueprint
keyword_search_bp = Blueprint('keyword_search_bp', __name__)


# ======== Helper Functions ========

def require_auth(f):
    """Decorator to require authentication for sensitive endpoints"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # You can replace this with your actual authentication logic
        # For now, we'll just check for an API key in headers
        api_key = request.headers.get('X-API-Key')

        # Skip auth check in development mode
        if current_app.config.get('DEVELOPMENT', False):
            return f(*args, **kwargs)

        if not api_key or api_key != current_app.config.get('API_KEY'):
            return jsonify({
                'status': 'error',
                'message': 'Authentication required'
            }), 401
        return f(*args, **kwargs)

    return decorated_function


def format_response(data, status_code=200):
    """Helper to format consistent API responses"""
    if isinstance(data, dict) and 'status' not in data:
        data['status'] = 'success' if status_code < 400 else 'error'
    return jsonify(data), status_code


# ======== Search Routes ========

@keyword_search_bp.route('/search', methods=['POST'])
def process_keyword_search():
    """
    Process a keyword search from the chatbot.

    This endpoint receives a user's input text and performs a search based on
    registered keywords and search patterns.

    Request Body:
        {
            "input": "string - The user's input text to search for"
        }

    Returns:
        JSON object containing search results
    """
    data = request.json
    if not data or 'input' not in data:
        return format_response({
            'message': 'Missing required "input" field in request'
        }, 400)

    user_input = data.get('input', '')

    # Create searcher with context manager to auto-close session
    with KeywordSearch() as searcher:
        result = searcher.execute_search(user_input)
        return format_response(result)


@keyword_search_bp.route('/test', methods=['GET'])
def test_search():
    """
    Test search functionality with a GET endpoint for easier testing.

    Query Parameters:
        query: The search text to test

    Returns:
        JSON object containing search results
    """
    query = request.args.get('query', '')
    if not query:
        return format_response({
            'message': 'Missing "query" parameter'
        }, 400)

    with KeywordSearch() as searcher:
        result = searcher.execute_search(query)
        return format_response(result)


# ======== Keyword Management Routes ========

@keyword_search_bp.route('/keyword', methods=['POST'])
@require_auth
def register_keyword():
    """
    Register a new keyword or update an existing one.

    Request Body:
        {
            "keyword": "string - The keyword to register (required)",
            "action_type": "string - Type of action (e.g., 'image_search') (required)",
            "search_pattern": "string - Pattern for parameter extraction (optional)",
            "entity_type": "string - Type of entity to search (optional)",
            "description": "string - Description of the keyword (optional)"
        }

    Returns:
        JSON object with registration status
    """
    data = request.json
    if not data or 'keyword' not in data or 'action_type' not in data:
        return format_response({
            'message': 'Missing required fields: "keyword" and "action_type" are required'
        }, 400)

    keyword = data.get('keyword')
    action_type = data.get('action_type')
    search_pattern = data.get('search_pattern')
    entity_type = data.get('entity_type')
    description = data.get('description')

    with KeywordSearch() as searcher:
        result = searcher.register_keyword(
            keyword=keyword,
            action_type=action_type,
            search_pattern=search_pattern,
            entity_type=entity_type,
            description=description
        )
        return format_response(result)


@keyword_search_bp.route('/keyword/<keyword>', methods=['GET'])
def get_keyword(keyword):
    """
    Get details for a specific keyword.

    Path Parameters:
        keyword: The keyword to retrieve

    Returns:
        JSON object with keyword details
    """
    with KeywordSearch() as searcher:
        # Get all keywords
        all_keywords = searcher.get_all_keywords()

        # Find the matching keyword
        matching = next((k for k in all_keywords if k['keyword'] == keyword), None)

        if matching:
            return format_response(matching)
        else:
            return format_response({
                'message': f'Keyword "{keyword}" not found'
            }, 404)


@keyword_search_bp.route('/keyword/<keyword>', methods=['DELETE'])
@require_auth
def delete_keyword(keyword):
    """
    Delete a keyword.

    Path Parameters:
        keyword: The keyword to delete

    Returns:
        JSON object with deletion status
    """
    with KeywordSearch() as searcher:
        result = searcher.delete_keyword(keyword)
        if result.get('status') == 'not_found':
            return format_response(result, 404)
        return format_response(result)


@keyword_search_bp.route('/keywords', methods=['GET'])
def get_all_keywords():
    """
    Get all registered keywords.

    Query Parameters:
        search: (optional) Filter keywords by pattern

    Returns:
        JSON object with list of keywords
    """
    search_filter = request.args.get('search', '').lower()

    with KeywordSearch() as searcher:
        all_keywords = searcher.get_all_keywords()

        # Apply filter if provided
        if search_filter:
            filtered_keywords = [k for k in all_keywords
                                 if search_filter in k['keyword'].lower()
                                 or (k.get('description') and search_filter in k['description'].lower())]
        else:
            filtered_keywords = all_keywords

        return format_response({
            'keywords': filtered_keywords,
            'count': len(filtered_keywords),
            'total': len(all_keywords)
        })


# ======== Bulk Import Routes ========

@keyword_search_bp.route('/keywords/upload', methods=['POST'])
@require_auth
def upload_keywords():
    """
    Upload keywords from Excel file.

    Form Data:
        file: Excel file (.xlsx or .xls) containing keywords

    Excel File Format:
        Required columns:
        - keyword: The keyword to register
        - action_type: Type of action (e.g., 'image_search')

        Optional columns:
        - search_pattern: Pattern for parameter extraction
        - entity_type: Type of entity to search
        - description: Description of the keyword
        - action: Legacy column (will be parsed if action_type is missing)

    Returns:
        JSON object with import results
    """
    if 'file' not in request.files:
        return format_response({
            'message': 'No file provided'
        }, 400)

    file = request.files['file']
    if file.filename == '':
        return format_response({
            'message': 'No file selected'
        }, 400)

    if not file.filename.endswith(('.xlsx', '.xls')):
        return format_response({
            'message': 'File must be an Excel file (.xlsx or .xls)'
        }, 400)

    # Save file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp:
        file.save(temp.name)
        temp_path = temp.name

    try:
        with KeywordSearch() as searcher:
            result = searcher.load_keywords_from_excel(temp_path)

        # Delete the temporary file
        os.unlink(temp_path)

        return format_response(result)

    except Exception as e:
        # Delete the temporary file if it exists
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        logger.error(f"Error processing Excel file: {e}")
        return format_response({
            'message': f'Error processing Excel file: {str(e)}'
        }, 500)


@keyword_search_bp.route('/keywords/export', methods=['GET'])
def export_keywords():
    """
    Export all keywords to Excel file.

    Returns:
        Excel file download
    """
    try:
        import pandas as pd
        import io

        with KeywordSearch() as searcher:
            keywords = searcher.get_all_keywords()

        # Convert to DataFrame
        df = pd.DataFrame(keywords)

        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Keywords', index=False)

            # Auto-adjust columns' width
            worksheet = writer.sheets['Keywords']
            for i, col in enumerate(df.columns):
                # Find the maximum length of the column
                max_len = max(
                    df[col].astype(str).apply(len).max(),  # max length of values
                    len(col)  # length of column name
                ) + 2  # adding a little extra space
                worksheet.set_column(i, i, max_len)

        output.seek(0)

        # Send file
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='keywords_export.xlsx'
        )

    except Exception as e:
        logger.error(f"Error exporting keywords: {e}")
        return format_response({
            'message': f'Error exporting keywords: {str(e)}'
        }, 500)


# ======== Debug/Utility Routes ========

@keyword_search_bp.route('/match', methods=['POST'])
def test_pattern_matching():
    """
    Test pattern matching functionality.

    Request Body:
        {
            "pattern": "string - The pattern with {param} placeholders",
            "text": "string - The text to match against"
        }

    Returns:
        JSON object with extracted parameters
    """
    data = request.json
    if not data or 'pattern' not in data or 'text' not in data:
        return format_response({
            'message': 'Missing required fields: "pattern" and "text" are required'
        }, 400)

    pattern = data.get('pattern')
    text = data.get('text')

    with KeywordSearch() as searcher:
        params = searcher.match_pattern(pattern, text)

        if params:
            return format_response({
                'matched': True,
                'parameters': params
            })
        else:
            return format_response({
                'matched': False,
                'message': 'No match found for the given pattern and text'
            })


@keyword_search_bp.route('/debug/<entity_type>/<entity_id>', methods=['GET'])
def debug_entity(entity_type, entity_id):
    """
    Debug endpoint to retrieve raw entity data.

    Path Parameters:
        entity_type: Type of entity (image, part, drawing, tool, position, problem, task)
        entity_id: ID of the entity

    Returns:
        JSON object with entity data
    """
    try:
        entity_id = int(entity_id)
    except ValueError:
        return format_response({
            'message': 'Invalid entity ID. Must be an integer.'
        }, 400)

    # Map entity types to model classes
    from modules.emtacdb.emtacdb_fts import (
        Image, Part, Drawing, Tool, Position, Problem, Task
    )

    entity_map = {
        'image': Image,
        'part': Part,
        'drawing': Drawing,
        'tool': Tool,
        'position': Position,
        'problem': Problem,
        'task': Task
    }

    if entity_type not in entity_map:
        return format_response({
            'message': f'Unknown entity type: {entity_type}. Valid types are: {", ".join(entity_map.keys())}'
        }, 400)

    model_class = entity_map[entity_type]

    db_config = DatabaseConfig()
    with db_config.get_main_session() as session:
        entity = session.query(model_class).get(entity_id)

        if not entity:
            return format_response({
                'message': f'{entity_type.capitalize()} with ID {entity_id} not found'
            }, 404)

        # Convert entity to dict (handling relationships appropriately)
        result = {}
        for column in entity.__table__.columns:
            result[column.name] = getattr(entity, column.name)

        # Add count of relationships
        # This will vary by entity type, so we'll do case-by-case additions
        if entity_type == 'image':
            result['position_count'] = len(entity.image_position_association) if hasattr(entity,
                                                                                         'image_position_association') else 0
            result['problem_count'] = len(entity.image_problem) if hasattr(entity, 'image_problem') else 0
            result['task_count'] = len(entity.image_task) if hasattr(entity, 'image_task') else 0
        elif entity_type == 'problem':
            result['solution_count'] = len(entity.solutions) if hasattr(entity, 'solutions') else 0
            result['position_count'] = len(entity.problem_position) if hasattr(entity, 'problem_position') else 0

        return format_response({
            'entity_type': entity_type,
            'entity_id': entity_id,
            'data': result
        })


@keyword_search_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.

    Returns:
        JSON object with health status
    """
    try:
        # Check database connection
        db_config = DatabaseConfig()
        with db_config.get_main_session() as session:
            from modules.emtacdb.emtacdb_fts import KeywordAction
            # Simple query to verify DB connection
            count = session.query(KeywordAction).count()

        return format_response({
            'healthy': True,
            'database': 'connected',
            'keyword_count': count,
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return format_response({
            'healthy': False,
            'error': str(e)
        }, 500)