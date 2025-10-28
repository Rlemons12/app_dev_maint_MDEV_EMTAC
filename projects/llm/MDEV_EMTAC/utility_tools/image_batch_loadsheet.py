import pandas as pd
from modules.emtacdb.utlity.main_database.database import add_image_to_db
import os
import base64
import requests
from modules.configuration.config import API_KEY
from PIL import Image as PILImage

# Define constants and configurations
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MINIMUM_SIZE = (250, 250)  # Define the minimum width and height for the image

# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_encoded

# Function to generate a description for an image using OpenAI API
def generate_description(image_path):
    base64_image = encode_image(image_path)
    payload = {
        "model": "gpt-4-vision-preview",
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
    
    if response.status_code == 200:
        response_data = response.json()
        if 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content']
        else:
            return "No description available."
    else:
        return "Error in API request."

# Function to process images and add them to the database
def process_images_from_load_sheet(load_sheet_path):
    if os.path.exists(load_sheet_path):
        load_sheet_data = pd.read_excel(load_sheet_path)
        for index, row in load_sheet_data.iterrows():
            image_path = row['Image Path']
            part_number = row['Part #']
            description = row['Manufacturer Description']  # Assuming 'Manufacturer Description' is a column in the load sheet
            if os.path.exists(image_path):
                try:
                    with PILImage.open(image_path) as img:
                        width, height = img.size
                        if width >= MINIMUM_SIZE[0] and height >= MINIMUM_SIZE[1]:
                            if not description:
                                description = generate_description(image_path)
                            with open(image_path, 'rb') as file:
                                file_data = file.read()
                                add_image_to_db(part_number, None, None, None, None, None, file_data, None, description)
                        else:
                            print("Image does not meet minimum size requirement. Skipping processing...")
                except Exception as e:
                    print("Error processing image:", str(e))
            else:
                print("Image file does not exist:", image_path)
    else:
        print("Load sheet file does not exist.")

# Entry point of the script
if __name__ == "__main__":
    load_sheet_path = input("Please enter the full path to your load sheet file: ")
    process_images_from_load_sheet(load_sheet_path)
