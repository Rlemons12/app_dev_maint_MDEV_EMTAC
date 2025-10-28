from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from modules.emtacdb.emtacdb_fts import Image
from modules.configuration.config import DATABASE_URL
from sqlalchemy.ext.declarative import declarative_base
from PIL import Image as PILImage
from io import BytesIO

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))  # Use scoped_session here
session = Session()
MINIMUM_SIZE = (125, 125)  # Define the minimum width and height for the image
MAX_ASPECT_RATIO = 4  # Define the maximum allowable aspect ratio

def meets_image_requirements(image):
    # Function to check if the image meets the minimum size and aspect ratio requirements
    with PILImage.open(BytesIO(image.image_blob)) as img:
        width, height = img.size
        aspect_ratio = width / height
        # Check minimum size and aspect ratio constraints
        return (width >= MINIMUM_SIZE[0] and height >= MINIMUM_SIZE[1]) and (aspect_ratio <= MAX_ASPECT_RATIO)

# Query all images from the database
images = session.query(Image).all()

# Iterate through each image
for image in images:
    # Check if the image title starts with "A1"; if yes, skip it
    if image.title.startswith(("A1", "1")):
        continue
    
    # Check if the image meets the extended requirements
    if not meets_image_requirements(image):
        # Delete the image from the database
        session.delete(image)

# Commit the changes to the database
session.commit()

print("Images not meeting the requirements have been deleted from the database.")