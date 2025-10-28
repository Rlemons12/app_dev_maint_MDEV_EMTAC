import pyexiv2

def add_metadata_to_image():
    image_path = input("Enter the path of the PNG image file: ")
    metadata = input("Enter the metadata to add: ")

    try:
        metadata_dict = {'Exif.Image.ImageDescription': metadata}
        image = pyexiv2.Image(image_path)
        image.modify_exif(metadata_dict)
        image.close()
        print(f"Metadata added to {image_path} successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
add_metadata_to_image()
