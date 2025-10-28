# blueprints/image_compare_bp.py

from flask import Blueprint, jsonify, request, send_from_directory, send_file, flash
import os
import numpy as np
from PIL import Image as PILImage
from werkzeug.utils import secure_filename
from modules.emtacdb.emtacdb_fts import Image, ImageEmbedding
from plugins.image_modules.image_models import get_image_model_handler
from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER, DATABASE_DIR
from modules.configuration.log_config import logger
from modules.configuration.config_env import DatabaseConfig
from plugins.ai_modules import ModelsConfig

db_config = DatabaseConfig()

image_compare_bp = Blueprint('image_compare_bp', __name__)

# Instantiate the appropriate handler using the function from image_modules.py
image_handler = ModelsConfig.load_image_model()


@image_compare_bp.route('/upload_and_compare', methods=['POST'])
def upload_and_compare():
    logger.info('Received request to upload and compare an image.')

    # Log information about the image handler
    logger.info(f'Using image handler: {type(image_handler).__name__}')

    if 'query_image' not in request.files:
        logger.error('Request missing file part.')
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['query_image']
    if file.filename == '':
        logger.error('File uploaded without a filename.')
        return jsonify({'error': 'No selected file'}), 400

    # Log the file extension for debugging
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    logger.info(f'Uploaded file: {file.filename} with extension: {file_ext}')

    # Check if image handler has the allowed_file method
    if not hasattr(image_handler, 'allowed_file'):
        logger.error('Image handler does not have allowed_file method')
        # Implement fallback file type checking
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        is_allowed = file_ext in allowed_extensions
        logger.info(f'Using fallback validation - extension: {file_ext}, allowed: {is_allowed}')
        if not is_allowed:
            return jsonify({'error': f'File type not allowed: {file_ext}'}), 400
    else:
        # Use the handler's allowed_file method
        try:
            is_allowed = image_handler.allowed_file(file.filename)
            logger.info(f'Handler validation - file: {file.filename}, allowed: {is_allowed}')
            if not is_allowed:
                return jsonify({'error': f'File type not allowed by handler: {file_ext}'}), 400
        except Exception as e:
            logger.exception(f'Error checking file type: {e}')
            return jsonify({'error': 'Error validating file type'}), 500

    # Continue with file processing since file type is allowed
    filename = secure_filename(file.filename)
    logger.info(f'Processing uploaded file: {filename}')

    # Define upload folder and ensure it exists
    upload_folder = os.path.join(DATABASE_PATH_IMAGES_FOLDER, 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    logger.debug(f'Upload folder verified: {upload_folder}')

    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    logger.info(f'File saved at {file_path}')

    # Process image
    try:
        logger.info('Opening and processing uploaded image...')
        image = PILImage.open(file_path).convert("RGB")

        # Check if handler has get_image_embedding method
        if not hasattr(image_handler, 'get_image_embedding'):
            logger.error('Image handler does not have get_image_embedding method')
            return jsonify({'error': 'Image handler configuration error'}), 500

        embedding = image_handler.get_image_embedding(image)
        logger.info('Image successfully processed and embedding obtained.')
    except Exception as e:
        logger.exception(f'Error processing uploaded image: {e}')
        return jsonify({'error': 'Failed to process the uploaded image.'}), 500

    if embedding is None:
        logger.error('Failed to extract image embedding.')
        return jsonify({'error': 'Failed to process the uploaded image.'}), 500

    logger.info('Starting enhanced similarity search using the advanced search_images method.')

    try:
        # Use db_config.get_main_session() to get the session
        session = db_config.get_main_session()

        # Convert embedding to list format if needed
        if hasattr(embedding, 'tolist'):
            embedding_list = embedding.tolist()
        elif isinstance(embedding, np.ndarray):
            embedding_list = embedding.flatten().tolist()
        else:
            embedding_list = list(embedding)

        # ENHANCED: Use the sophisticated search_images method with pgvector
        search_results = Image.search_images(
            session=session,
            similarity_query_embedding=embedding_list,
            similarity_threshold=0.3,  # Lower threshold to get more results
            embedding_model_name=type(image_handler).__name__,
            use_hybrid_ranking=True,  # Use intelligent ranking
            limit=10,  # Return top 10 matches
        )

        logger.info(f'Enhanced search found {len(search_results)} similar images.')

        # Convert search results to the expected format for frontend
        results = []
        for result in search_results:
            similarity_score = result.get('search_metadata', {}).get('similarity_score')

            result_item = {
                'id': result['id'],
                'title': result['title'],
                'description': result['description'],
                'file_path': os.path.basename(result['file_path']),
                'similarity': float(similarity_score) if similarity_score is not None else 0.0
            }
            results.append(result_item)

            # Log the result with similarity score
            similarity_str = f"{similarity_score:.4f}" if similarity_score is not None else "N/A"
            logger.debug(f'Enhanced result: ID={result["id"]}, similarity={similarity_str}, '
                         f'paths={result.get("search_metadata", {}).get("search_paths", [])}')

        logger.info('Successfully processed and compared the image using enhanced search.')
        return jsonify({'image_similarity_search': results})

    except Exception as e:
        logger.exception(f'An error occurred during the enhanced comparison process: {e}')
        return jsonify({'error': 'An error occurred during the comparison process.'}), 500
    finally:
        session.close()
        logger.info('Database session closed.')


@image_compare_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    logger.info(f'Serving file {filename} from uploads.')
    return send_from_directory(DATABASE_PATH_IMAGES_FOLDER, filename)


@image_compare_bp.route('/serve_image/<int:image_id>')
def serve_image_route(image_id):
    logger.info(f"Received request to serve image with ID: {image_id}")

    with db_config.get_main_session() as session:
        try:
            return serve_image(session, image_id)
        except Exception as e:
            logger.exception(f"Error serving image {image_id}:")
            flash(f"Error serving image {image_id}", "error")
            return "Image not found", 404


def serve_image(session, image_id):
    logger.info(f"Attempting to retrieve image with ID: {image_id}")
    try:
        image = session.query(Image).filter_by(id=image_id).first()
        if image:
            file_path = os.path.join(DATABASE_DIR, image.file_path)
            if os.path.exists(file_path):
                logger.info(f"Serving file: {file_path}")
                return send_file(file_path, mimetype='image/jpeg', as_attachment=False)
            else:
                logger.error(f"File not found on disk: {file_path}")
                return "Image file not found", 404
        else:
            logger.error(f"No image found in database with ID: {image_id}")
            return "Image not found", 404
    except Exception as e:
        logger.exception("Unhandled error while serving the image:")
        return "Internal Server Error", 500