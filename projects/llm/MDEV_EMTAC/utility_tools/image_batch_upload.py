import os
from modules.emtacdb.emtacdb_fts import Session, Image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

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
                # You can set other fields like area, equipment_group, etc. here if needed
                # For demonstration, I'll leave them empty
                add_image_to_db(title=title, area="", equipment_group="", model="", asset_number="", image_blob=file_data)
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path from which you want to upload images: ")
    batch_upload_images(folder_path)
