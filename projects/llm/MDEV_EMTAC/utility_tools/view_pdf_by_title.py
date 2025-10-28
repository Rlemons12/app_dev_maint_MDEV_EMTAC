from flask import Flask, send_file
import os

app = Flask(__name__)

# Import your functions from emtacdb_fts.py
from modules.emtacdb.utlity.main_database.database import get_powerpoints_by_title


@app.route('/view_pdf_by_title/<string:title>')
def view_pdf_by_title(title):
    try:
        # Retrieve the PowerPoint presentations by title from the database
        powerpoints = get_powerpoints_by_title(title)

        if powerpoints:
            # Assuming you want to display the first matching PowerPoint presentation
            powerpoint = powerpoints[0]

            # Extract the relative PDF file path from the PowerPoint object
            relative_pdf_file_path = powerpoint.pdf_file_path

            # Construct the absolute file path
            absolute_pdf_file_path = os.path.join(os.path.dirname(__file__), relative_pdf_file_path)

            # Check if the PDF file exists
            if os.path.exists(absolute_pdf_file_path):
                # Serve the PDF file for viewing
                return send_file(absolute_pdf_file_path, as_attachment=False)
            else:
                return "PDF file not found", 404
        else:
            return "PowerPoint not found", 404

    except Exception as e:
        return f"Error while viewing PDF: {str(e)}", 500
        
if __name__ == '__main__':
    app.run(debug=True)
