from flask import Blueprint, request, render_template, send_file
from modules.emtacdb.emtacdb_fts import PowerPoint
from modules.emtacdb.utlity.main_database.database import get_powerpoints_by_title
from sqlalchemy.orm import Session
from modules.configuration.config import DATABASE_URL, PPT2PDF_PPT_FILES_PROCESS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import os

# Create a SQLAlchemy engine using the DATABASE_URL from your config
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = sessionmaker(bind=engine)

search_powerpoint_bp = Blueprint('search_powerpoint_bp', __name__)


@search_powerpoint_bp.route('/pdf/<title>', methods=['GET'])
def view_pdf_by_title(title):
    # Retrieve the PowerPoint presentation based on its title
    powerpoints = get_powerpoints_by_title(title=title)

    if powerpoints:
        # Assuming you want to display the first matching PowerPoint presentation
        powerpoint = powerpoints[0]
        
        # Construct the full path to the PDF file
        #full_pdf_path = os.path.join( DATABASE_DOC, powerpoint.pdf_file_path)
        full_ppt_path = os.path.join( PPT2PDF_PPT_FILES_PROCESS, powerpoint.ppt_file_path)

        if os.path.exists(full_ppt_path):
            # Serve the PowerPoint file as a response
            return send_file(
                full_ppt_path,
                mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation',
                as_attachment=True,
                download_name=f"{title}.pptx"  # Adjusted file extension to .pptx
            )
            # Handle the case where the PDF file does not exist
            return "PowerPoint file not found", 404
    else:
        # Handle the case where the PowerPoint presentation is not found
        return "PowerPoint presentation not found", 404


@search_powerpoint_bp.route('/', methods=['GET'])
def search_powerpoint():
    session = Session()  # Create a session instance
    query = session.query(PowerPoint)
    
    # Retrieve parameters from the request
    title = request.args.get('title', '')
    area = request.args.get('searchpowerpoint_area', '')
    equipment_group = request.args.get('searchpowerpoint_equipment_group', '')
    model = request.args.get('searchpowerpoint_model', '')
    asset_number = request.args.get('searchpowerpoint_asset_number', '')
    location = request.args.get('searchpowerpoint_location','')  # Corrected typo in parameter name
    description = request.args.get('description', '')

    print("Debug: Received search request with the following parameters:")
    print(f"Title: {title}")
    print(f"Area: {area}")
    print(f"Equipment Group: {equipment_group}")
    print(f"Model: {model}")
    print(f"Asset Number: {asset_number}")
    print(f"Location: {location}")  # Corrected typo in print statement
    print(f"Description: {description}")

    print(f'# Apply filters based on provided parameters')
    
    if any([title, area, equipment_group, model, asset_number, location]):
        # Apply filters based on AND logic
        query_filters = []
        if title:
            query_filters.append(PowerPoint.title.ilike(f"%{title}%"))
        if area:
            query_filters.append(PowerPoint.area == area)
        if equipment_group:
            query_filters.append(PowerPoint.equipment_group == equipment_group)
        if model:
            query_filters.append(PowerPoint.model == model)
        if asset_number:
            query_filters.append(PowerPoint.asset_number == asset_number)
        if location:
            query_filters.append(PowerPoint.location == location)

        # Apply all filters using AND logic
        for filter_condition in query_filters:
            query = query.filter(filter_condition)
        
        powerpoints = query.all()  # Corrected variable name from `powerpoint` to `powerpoints`
        session.close()
        
        if powerpoints:
            print(f"Debug: Found {len(powerpoints)} PowerPoint presentations")
            # Pass the powerpoints data to the template
            return render_template('powerpoint_search_results.html', powerpoints=powerpoints)
    
    # If no search criteria provided or no matching results found, render the search form or a default page
    print("Debug: No search criteria provided or no matching results found. Rendering search form.")
    return render_template('upload_search_database/upload_search_database.html')