import os
import logging
from flask import Blueprint, jsonify, request, render_template
from PIL import Image as PILImage, ImageFile
from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER, DATABASE_URL
from plugins.image_modules import CLIPModelHandler, NoImageModel

from sqlalchemy import (create_engine)
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker
from plugins.ai_modules import ModelsConfig
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))  # Use scoped_session here
session = Session()

folder_image_embedding_bp = Blueprint('folder_image_embedding_bp', __name__)

# Instantiate the appropriate handler using the function from image_modules.py
image_handler = ModelsConfig.load_image_model()

def process_and_store_images(folder):
    session = Session()
    for filename in os.listdir(folder):
        if image_handler.allowed_file(filename):
            source_file_path = os.path.join(folder, filename)
            dest_file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
            
            try:
                image = PILImage.open(source_file_path).convert("RGB")
                if not image_handler.is_valid_image(image):
                    logging.info(f"Skipping {filename}: Image does not meet the required dimensions or aspect ratio.")
                    continue
                
                embedding = image_handler.get_image_embedding(image)
                if embedding is not None:
                    # Save the image to the destination folder
                    os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)
                    image.save(dest_file_path)
                    image_handler.store_image_metadata(session, filename, "Auto-generated description", dest_file_path, embedding, CURRENT_IMAGE_MODEL)
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")
    session.close()

@folder_image_embedding_bp.route('/process_folder', methods=['POST'])
def process_folder():
    folder_path = request.form.get('folder_path')
    if not folder_path or not os.path.isdir(folder_path):
        return jsonify({'error': 'Invalid folder path.'}), 400

    process_and_store_images(folder_path)
    return jsonify({'success': 'Images processed and stored.'}), 200

@folder_image_embedding_bp.route('/compare_image', methods=['GET'])
def upload_and_compare_form():
    return render_template('upload_and_compare.html')
