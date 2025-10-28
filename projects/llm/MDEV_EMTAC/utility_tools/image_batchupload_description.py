import os
import base64
import requests
from modules.emtacdb.emtacdb_fts import Session, Image
from modules.configuration.config import API_KEY

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# OpenAI API Key
api_key = API_KEY

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
        "Authorization": f"Bearer {api_key}"
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_image_to_db(title, area, equipment_group, model, asset_number, image_blob, description=""):
    with Session() as session:
        existing_image = session.query(Image).filter(Image.title == title).first()
        if existing_image is None:
            new_image = Image(
                title=title,
                area=area,
                equipment_group=equipment_group,
                model=model,
                asset_number=asset_number,
                image_blob=image_blob,
                description=description)
            session.add(new_image)
            session.commit()
            print(f"Added image: {title}")
        else:
            print(f"Image already exists: {title}")

def batch_upload_images(folder_path):
    for filename in os.listdir(folder_path):
        if allowed_file(filename):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                # Extract title from filename
                title = os.path.splitext(filename)[0]
                # Check if the image already exists in the database
                existing_image = Session().query(Image).filter(Image.title == title).first()
                if existing_image is None:
                    # Generate description for the image
                    description = generate_description(file_path)
                    # You can set other fields like area, equipment_group, etc. here if needed
                    # For demonstration, I'll leave them empty
                    add_image_to_db(title=title, area="", equipment_group="", model="", asset_number="", image_blob=file_data, description=description)
                else:
                    print(f"Image already exists: {title}")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path from which you want to upload images: ")
    batch_upload_images(folder_path)
