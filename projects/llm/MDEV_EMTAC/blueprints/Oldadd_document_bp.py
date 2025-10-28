from flask import Blueprint, request, jsonify, redirect, url_for, current_app
from werkzeug.utils import secure_filename
import os
from modules.emtacdb.utlity.main_database.database import add_document_to_db

add_document_bp = Blueprint('add_document_bp', __name__)

@add_document_bp.route('/add_document', methods=['POST'])
def add_document():
    # Check if the post request has the 'document' part
    if 'document' not in request.files:
        return jsonify({'error': 'No document file provided'}), 400

    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No selected document'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        title = request.form.get('title')
        success = add_document_to_db(title, file_path)

        if success:
            return redirect(url_for('upload_success'))  # Corrected redirect
        else:
            return jsonify({'error': 'Failed to add document to the database'}), 500

    return jsonify({'error': 'An error occurred during file upload'}), 500
