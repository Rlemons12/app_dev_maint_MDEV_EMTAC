# convert heif images to jpeg
import os
import pillow_heif
from PIL import Image

# Prompt the user for the folder path
folder_path = input("Please enter the folder path containing HEIC files: ")

# Check if the provided folder exists
if not os.path.isdir(folder_path):
    print("The provided folder does not exist. Please check the path and try again.")
    exit(1)

# Create a subfolder for converted images
subfolder_name = "converted"
output_folder = os.path.join(folder_path, subfolder_name)
os.makedirs(output_folder, exist_ok=True)

# Process each file in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.heic'):
        input_file = os.path.join(folder_path, filename)
        # Create an output filename with a .jpg extension inside the subfolder
        output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')
        try:
            # Read the HEIC file
            heif_file = pillow_heif.read_heif(input_file)
            # Convert the HEIC file to a Pillow image
            image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
            # Save the image in JPEG format
            image.save(output_file, 'JPEG')
            print(f"Converted {input_file} to {output_file}")
        except Exception as e:
            print(f"Failed to convert {input_file}: {e}")

print("Conversion complete!")
