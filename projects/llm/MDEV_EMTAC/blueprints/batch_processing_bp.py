# blueprints/batch_processing_bp.py

from flask import Blueprint, request, jsonify, redirect, url_for
from modules.emtacdb.emtacdb_fts import (FileLog)
from modules.configuration.config_env import DatabaseConfig
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the database configuration
db_config = DatabaseConfig()
Session = db_config.get_main_session_registry()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the directory containing this script
LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
LOG_FILE = 'script_sql_speed_test.csv'  # Change the log file extension to .csv
log_file_path = os.path.join(LOG_FOLDER, LOG_FILE)

# Create the log file if it doesn't exist
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w') as file:
        file.write('file_path,total_time\n')  # Header for the CSV file

batch_processing_bp = Blueprint("batch_processing_bp", __name__)


@batch_processing_bp.route("/batch_processing", methods=["POST"])
def batch_processing():
    logger.info("Received form data:")
    logger.info(request.form)

    # Check if the request contains any folder path
    folder_path = request.form.get("batchFolderPath")
    logger.info(f'Folder Path: {folder_path}')
    if not folder_path:
        return jsonify({"message": "No folder path provided"}), 400

    batch_title = request.form.get("title")
    batch_area = request.form.get("batchArea")
    batch_equipment_group = request.form.get("batchEquipmentGroup")
    batch_model = request.form.get("batchModel")
    batch_asset_number = request.form.get("batchAssetNumber")
    batch_location = request.form.get("batchLocation")

    process_folder(folder_path, batch_title, batch_area, batch_equipment_group, batch_model, batch_asset_number,
                   batch_location)

    logger.info(f"Batch Title: {batch_title}")
    logger.info(f"Batch Area: {batch_area}")
    logger.info(f"Batch Equipment Group: {batch_equipment_group}")
    logger.info(f"Batch Model: {batch_model}")
    logger.info(f"Batch Asset Number: {batch_asset_number}")
    logger.info(f"Batch Location: {batch_location}")

    return jsonify({"message": "Batch processing completed successfully"})


@batch_processing_bp.route("/add_batch_folder", methods=["POST"])
def add_batch_folder():
    if "files" not in request.files:
        return jsonify({"message": "No files uploaded"}), 400

    files = request.files.getlist("files")

    for file in files:
        if file.filename == "":
            continue

        try:
            title = request.form.get("title", "")
            if not title:
                filename = secure_filename(file.filename)
                file_name_without_extension = os.path.splitext(filename)[0]
                title = file_name_without_extension.replace('_', ' ')

            area = request.form.get("area", "")
            equipment_group = request.form.get("equipment_group", "")
            model = request.form.get("model", "")
            asset_number = request.form.get("asset_number", "")
            location = request.form.get("location", "")

            filename = secure_filename(file.filename)
            file_path = os.path.join("DB_IMAGES", filename)
            file.save(file_path)

            # Process file based on its type
            if file_path.endswith(('.ppt', '.pptx')):
                with open(file_path, 'rb') as ppt_file:
                    files = {'powerpoint': ppt_file}
                    data = {
                        'title': title,
                        'area': area,
                        'equipment_group': equipment_group,
                        'model': model,
                        'asset_number': asset_number,
                        'location': location
                    }
                    response = requests.post('http://127.0.0.1:5000/powerpoints/upload_powerpoint', files=files,
                                             data=data)
                    logger.info(f"Data for powerpoint upload: {data}")
                    if response.status_code == 200:
                        logger.info("PowerPoint file processed successfully")
                    else:
                        logger.error(f"Failed to process PowerPoint file: {response.text}")

            elif file_path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                with open(file_path, 'rb') as image_file:
                    files = {'image': image_file}
                    data = {
                        'title': title,
                        'area': area,
                        'equipment_group': equipment_group,
                        'model': model,
                        'asset_number': asset_number,
                        'location': location
                    }
                    response = requests.post('http://127.0.0.1:5000/image/upload_image', files=files, data=data)
                    logger.info(f"Data for image upload: {data}")
                    if response.status_code == 200:
                        logger.info("Image file processed successfully")
                    else:
                        logger.error(f"Failed to process image file: {response.text}")

            else:
                with open(file_path, 'rb') as document_file:
                    files = {'files': document_file}
                    data = {
                        'title': title,
                        'area': area,
                        'equipment_group': equipment_group,
                        'model': model,
                        'asset_number': asset_number,
                        'location': location
                    }
                    # The correct endpoint for add_document route
                    response = requests.post('http://127.0.0.1:5000/documents/add_document', files=files, data=data)
                    logger.info(f"Data for document upload: {data}")
                    if response.status_code == 200:
                        logger.info("Document processed successfully")
                    else:
                        logger.error(f"Failed to process document: {response.text}")

        except Exception as e:
            return jsonify({"message": str(e)}), 500

    return redirect(url_for('upload_success'))


def process_single_file(file_path, session_id, session_datetime, title, area, equipment_group, model, asset_number,
                        location):
    logger.info(f"file_path: {file_path}")
    start_time = datetime.now()

    try:
        if file_path.endswith(('.ppt', '.pptx')):
            with open(file_path, 'rb') as ppt_file:
                files = {'powerpoint': ppt_file}
                data = {
                    'title': title,
                    'area': area,
                    'equipment_group': equipment_group,
                    'model': model,
                    'asset_number': asset_number,
                    'location': location,
                    'session': session_id,
                    'session_datetime': session_datetime
                }
                response = requests.post('http://127.0.0.1:5000/powerpoints/upload_powerpoint', files=files, data=data)
                logger.info(f"Data for powerpoint upload: {data}")
                if response.status_code == 200:
                    logger.info("PowerPoint file processed successfully")
                else:
                    logger.error(f"Failed to process PowerPoint file: {response.text}")

        elif file_path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            with open(file_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {
                    'title': title,
                    'area': area,
                    'equipment_group': equipment_group,
                    'model': model,
                    'asset_number': asset_number,
                    'location': location,
                    'session': session_id,
                    'session_datetime': session_datetime
                }
                response = requests.post('http://127.0.0.1:5000/image/upload_image', files=files, data=data)
                logger.info(f"Data for image upload: {data}")
                if response.status_code == 200:
                    logger.info("Image file processed successfully")
                else:
                    logger.error(f"Failed to process image file: {response.text}")

        else:
            with open(file_path, 'rb') as file:
                files = {'files': file}
                data = {
                    'title': title,
                    'area': area,
                    'equipment_group': equipment_group,
                    'model': model,
                    'asset_number': asset_number,
                    'location': location,
                    'session': session_id,
                    'session_datetime': session_datetime
                }
                response = requests.post('http://127.0.0.1:5000/documents/add_document', files=files, data=data)
                logger.info(f"Data for document upload: {data}")
                if response.status_code == 200:
                    logger.info("Document processed successfully")
                else:
                    logger.error(f"Failed to process document: {response.text}")

        end_time = datetime.now()
        total_time = end_time - start_time
        total_time_str = str(total_time)
        file_log_entry = FileLog(session=session_id, session_datetime=session_datetime, file_processed=file_path,
                                 total_time=total_time_str)

        # Use the session from the database configuration
        session = Session()
        try:
            session.add(file_log_entry)
            session.commit()
            logger.info(f"Added file_log entry to database: {file_log_entry.file_processed}")
        except Exception as db_error:
            session.rollback()
            logger.error(f"Database error saving file log: {str(db_error)}")
        finally:
            session.close()

        # Write the processing time to the log file
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'{file_path},{total_time_str}\n')

    except Exception as e:
        logger.error(f"Error processing file: {file_path}. Error: {str(e)}")


def process_folder(folder_path, batch_title, batch_area, batch_equipment_group, batch_model, batch_asset_number,
                   batch_location):
    folder_path = os.path.abspath(folder_path)
    current_time = datetime.now()
    session_id = current_time.strftime("%Y%m%d%H%M%S")
    logger.info(f"Batch processing started at {current_time}")

    try:
        num_workers = os.cpu_count()  # Dynamically set the number of workers
        logger.info(f"Number of CPU cores available: {num_workers}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for root, dirs, files in os.walk(folder_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    logger.info(f"Scheduling file for processing: {file_path}")
                    future = executor.submit(process_single_file, file_path, session_id, current_time, batch_title,
                                             batch_area, batch_equipment_group, batch_model, batch_asset_number,
                                             batch_location)
                    futures.append(future)

            for future in futures:
                future.result()  # wait for all the futures to complete

    except Exception as e:
        logger.error(f"Error processing folder: {folder_path}. Error: {str(e)}")

    logger.info(f"Batch processing completed at {datetime.now()}")