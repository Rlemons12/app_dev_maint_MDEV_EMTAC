from flask import Blueprint, request, render_template, send_file
from sqlalchemy.orm import sessionmaker
import os
from modules.configuration.config import DATABASE_URL, PPT2PDF_PDF_FILES_PROCESS
from modules.emtacdb.emtacdb_fts import PowerPoint  # Assuming you have a Powerpoint model defined in 'models.py'
from sqlalchemy import create_engine, text

# Create a SQLAlchemy engine using the DATABASE_URL from your config
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = sessionmaker(bind=engine)

search_powerpoint_fts_bp = Blueprint('search_powerpoint_fts_bp', __name__)

@search_powerpoint_fts_bp.route('/', methods=['GET'])
def search_powerpoint_fts():
    query = request.args.get('query', '')

    # Create a SQLAlchemy session
    session = Session()

    try:
        # Construct the full-text search query in powerpoints_fts table for the title
        title_search_query = text(
            "SELECT title FROM powerpoint WHERE title MATCH :query"
        )

        # Execute the query with the user-provided search query
        title_search_results = session.execute(title_search_query, {"query": query})

        # Fetch the search results
        titles = [row.title for row in title_search_results]

        # Fetch Powerpoint objects from the database based on matched titles
        powerpoints = session.query(Powerpoint).filter(Powerpoint.title.in_(titles)).all()

        # Close the session
        session.close()

        if powerpoints:
            # Render the template with clickable links for each PowerPoint
            return render_template('powerpoint_results.html', powerpoints=powerpoints)
        else:
            return "No PowerPoint documents found", 404
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during the search.", 500

# Define the view_powerpoint route within the blueprint
@search_powerpoint_fts_bp.route('/view_powerpoint/<int:powerpoint_id>')
def view_powerpoint(powerpoint_id):
    # Create an SQLAlchemy session
    session = Session()

    try:
        # Fetch the PowerPoint from the database based on the ID
        powerpoint = session.query(PowerPoint).get(powerpoint_id)

        if powerpoint:
            file_path = os.path.join(PPT2PDF_PDF_FILES_PROCESS, powerpoint.pdf_file_path)
            if os.path.exists(file_path):
                return send_file(file_path)
            else:
                return "PowerPoint file not found", 404
        else:
            return "PowerPoint document not found", 404
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching the PowerPoint document.", 500
    finally:
        # Close the session
        session.close()
