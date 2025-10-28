from flask import Blueprint, request
from werkzeug.utils import secure_filename
from modules.emtacdb.utlity.main_database.database import create_position, add_powerpoint_to_db, add_document_to_db
from modules.configuration.config import PPT2PDF_PPT_FILES_PROCESS, PPT2PDF_PDF_FILES_PROCESS, DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
import os
import re
import comtypes.client
import pythoncom  # Correctly imported from pywin32
import logging

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))  # Use scoped_session here
session = Session()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all log levels
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='a'),  # Append mode
        logging.StreamHandler()                    # Log to the console
    ]
)

logger = logging.getLogger(__name__)

def extract_title_from_filename(filename):
    # Remove file extensions from filename
    title = os.path.splitext(filename)[0]
    # Remove any additional extensions (e.g., pptx, jpg, etc.)
    title = re.sub(r'\.\w+', '', title)
    return title

def convert_pptx_to_pdf(pptx_file, pdf_file):
    # Initialize COM
    pythoncom.CoInitialize()
    # Create a PowerPoint application object
    powerpoint = comtypes.client.CreateObject("PowerPoint.Application")

    try:
        # Open the PowerPoint presentation
        presentation = powerpoint.Presentations.Open(pptx_file)

        # Save the presentation as a PDF file
        presentation.SaveAs(pdf_file, 32)  # 32 represents PDF format
    finally:
        # Close the presentation and quit PowerPoint
        if presentation:
            presentation.Close()
        if powerpoint:
            powerpoint.Quit()
            pythoncom.CoUninitialize()

upload_powerpoint_bp = Blueprint('upload_powerpoint_bp', __name__)

@upload_powerpoint_bp.route('/upload_powerpoint', methods=['POST'])
def upload_powerpoint():
    try:
        title = request.form.get('title')
        area = request.form.get('area')
        equipment_group = request.form.get('equipment_group')
        model = request.form.get('model')
        asset_number = request.form.get('asset_number')
        location = request.form.get('location')
        site_location = request.form.get('site_location')  # Get site_location from the request
        description = request.form.get('description')
        ppt_file = request.files.get('powerpoint')

        # Print the form data to the console for debugging
        print(f"title: {title}")
        print(f"area: {area}")
        print(f"equipment_group: {equipment_group}")
        print(f"model: {model}")
        print(f"asset_number: {asset_number}")
        print(f"location: {location}")
        print(f"site_location: {site_location}")
        print(f"description: {description}")

        # Log the form data
        logger.debug(f"title: {title}")
        logger.debug(f"area: {area}")
        logger.debug(f"equipment_group: {equipment_group}")
        logger.debug(f"model: {model}")
        logger.debug(f"asset_number: {asset_number}")
        logger.debug(f"location: {location}")
        logger.debug(f"site_location: {site_location}")
        logger.debug(f"description: {description}")

        if not title:
            filename = secure_filename(ppt_file.filename)
            title = os.path.splitext(filename)[0]

        if ppt_file is None:
            return "No PowerPoint file provided", 400

        if not os.path.exists(PPT2PDF_PPT_FILES_PROCESS):
            os.makedirs(PPT2PDF_PPT_FILES_PROCESS)

        ppt_filename = secure_filename(ppt_file.filename)
        ppt_path = os.path.join(PPT2PDF_PPT_FILES_PROCESS, ppt_filename)
        ppt_file.save(ppt_path)

        pdf_filename = ppt_filename.replace(".pptx", ".pdf")
        pdf_file_path = os.path.join(PPT2PDF_PDF_FILES_PROCESS, pdf_filename)
        convert_pptx_to_pdf(ppt_path, pdf_file_path)

        # Convert form values to integers or None
        area_id = int(area) if area else None
        equipment_group_id = int(equipment_group) if equipment_group else None
        model_id = int(model) if model else None
        asset_number_id = int(asset_number) if asset_number else None
        location_id = int(location) if location else None

        # Print the values to the console and log them
        print(f"area_id: {area_id}, equipment_group_id: {equipment_group_id}, model_id: {model_id}, asset_number_id: {asset_number_id}, location_id: {location_id}, site_location: {site_location}")
        logger.debug(f"area_id: {area_id}, equipment_group_id: {equipment_group_id}, model_id: {model_id}, asset_number_id: {asset_number_id}, location_id: {location_id}, site_location: {site_location}")

        logger.debug(f"Creating position with parameters - area: {area}, equipment_group: {equipment_group}, model: {model}, asset_number: {asset_number}, location: {location}, site_location: {site_location}")
        position_id = create_position(area, equipment_group, model, asset_number, location, site_location, )
        logger.debug(f"Position ID created: {position_id}")

        if not position_id:
            logger.error("Failed to create position")
            return "Failed to create position", 500

        complete_document_id, success = add_document_to_db(title, pdf_file_path, position_id)
        logger.info(f"Complete document ID: {complete_document_id}, Success: {success}")

        if success:
            with Session() as session:
                new_powerpoint_id = add_powerpoint_to_db(session, title, ppt_path, pdf_file_path, complete_document_id, description)
                if not new_powerpoint_id:
                    return "Failed to add PowerPoint to the database", 500
        else:
            return "Failed to add document to the database", 500

        if success:
            return "Document uploaded and processed successfully"
        else:
            return "Failed to process the document", 500

    except Exception as e:
        logger.error(f"Error during PowerPoint upload and conversion: {str(e)}")
        return f"Error during PowerPoint upload and conversion: {str(e)}", 500
