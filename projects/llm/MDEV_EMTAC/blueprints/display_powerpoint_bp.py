from flask import Blueprint, send_file
import os

display_pdf_bp = Blueprint('display_pdf_bp', __name__)

@display_pdf_bp.route('/view_pdf_by_title/<string:title>')
def serve_pdf(powerpoint_id):
    # Create a new session
    Session = sessionmaker(bind=engine)
    with Session() as session:
        powerpoint = session.query(PowerPoint).filter_by(id=powerpoint_id).first()
        if powerpoint and powerpoint.pdf_file_path:
            # Ensure the PDF file exists
            if os.path.exists(powerpoint.pdf_file_path):
                return send_file(
                    powerpoint.pdf_file_path,
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=f"{powerpoint.title}.pdf"
                )
            else:
                return "PDF file not found", 404
        else:
            return "PowerPoint not found or PDF file path is empty", 404