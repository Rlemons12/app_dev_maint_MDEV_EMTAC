from flask import request, redirect, url_for, render_template, Blueprint
from werkzeug.utils import secure_filename
from modules.emtacdb.emtacdb_fts import (load_image_model_config_from_db)
from modules.emtacdb.utlity.main_database.database import create_position, add_image_to_db
from plugins.image_modules import CLIPModelHandler, NoImageModel
import os
import base64
import requests
from modules.configuration.config import API_KEY, DATABASE_PATH_IMAGES_FOLDER

upload_image_bp = Blueprint('upload_image_bp', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_description(image_path):
    base64_image = encode_image(image_path)
    print("Base64 Image:", base64_image)  # Print the base64 encoded image data
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    print("Request Payload:", payload)  # Print the request payload
    print("Headers:", headers)  # Print the request headers
    
    if response.status_code == 200:
        response_data = response.json()
        print("Response Data:", response_data)  # Print the response data
        if 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content']
        else:
            return "No description available."
    else:
        print("Error in API request. Status Code:", response.status_code)  # Print the error status code
        return "Error in API request."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_image_bp.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        title = request.form.get('title')
        if not title:  # If no title is provided, use the filename as the title
            filename = secure_filename(request.files['image'].filename)
            title = os.path.splitext(filename)[0]
        area = request.form.get('area')
        equipment_group = request.form.get('equipment_group')
        model = request.form.get('model')
        asset_number = request.form.get('asset_number')
        location = request.form.get('location')
        description = request.form.get('description')
        
        area_id = int(area) if area else None
        equipment_group_id = int(equipment_group) if equipment_group else None
        model_id = int(model) if model else None
        asset_number_id = int(asset_number) if asset_number else None
        location_id = int(location) if location else None
        
        position_id = create_position(area_id, equipment_group_id, model_id, asset_number_id, location_id)
        
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Define relative and absolute paths
            relative_path = os.path.join('images', filename)
            file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
            file.save(file_path)

            # Generate description if not provided
            if not description:
                description = generate_description(file_path)

            # Load current image model from the database
            current_image_model = load_image_model_config_from_db()
            model_handler = CLIPModelHandler() if current_image_model == "CLIPModelHandler" else NoImageModel()

            # Add the image metadata to the database with embeddings
            add_image_to_db(title, file_path, model_handler, position_id, None, None, description)

            return redirect(url_for('upload_image_bp.upload_image'))
    return render_template('upload.html')


