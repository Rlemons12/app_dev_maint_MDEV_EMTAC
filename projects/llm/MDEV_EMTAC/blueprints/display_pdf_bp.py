from flask import Blueprint, send_file
import os
from modules.emtacdb.utlity.main_database.database import get_powerpoints_by_title
from modules.configuration.config import DATABASE_DIR

PPT2PDF_PDF_FILES_PROCESS = os.path.join(DATABASE_DIR, 'PDF_FILES')

display_pdf_bp = Blueprint('display_pdf_bp', __name__)

@display_pdf_bp.route('/view_pdf_by_title/test')
def view_pdf_by_title_test():
    test_title = "test ppt2pdf3"
    print(f"Test title: {test_title}")

    try:
        # Retrieve the PowerPoint presentations by test title from the database
        powerpoints = get_powerpoints_by_title(test_title)

        if powerpoints:
            print(f"Found PowerPoint presentations for title '{test_title}'. Number of presentations found: {len(powerpoints)}")

            # Assuming you want to display the first matching PowerPoint presentation
            powerpoint = powerpoints[0]

            # Extract the PDF relative file path from the PowerPoint object
            pdf_relative_path = powerpoint.pdf_file_path
            print(f"PDF relative path: {pdf_relative_path}")

            # Construct the full path to the PDF file in the PDF_FILES directory
            pdf_full_path = os.path.join(BASE_DIR, 'Database', 'PDF_FILES', pdf_relative_path)
            print(f"Full PDF path: {pdf_full_path}")

            # Check if the PDF file exists at the specified path
            if os.path.exists(pdf_full_path):
                print(f"PDF file exists. Serving the file.")
                # Serve the PDF file for viewing
                return send_file(pdf_full_path, as_attachment=False)
            else:
                print(f"PDF file does not exist at path: {pdf_full_path}")

        else:
            print(f"No PowerPoint presentations found for title '{test_title}'")

        return "PowerPoint not found", 404
    except Exception as e:
        print(f"Error while viewing PDF: {str(e)}")
        return f"Error while viewing PDF: {str(e)}", 500
