import pyexiv2

def read_metadata_from_image(image_path):
    try:
        image = pyexiv2.Image(image_path)
        metadata = image.read_exif()
        image.close()
        if metadata:
            print("Metadata found in the image:")
            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print("No metadata found in the image.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
image_path = input("Enter the path of the PNG image file: ")
read_metadata_from_image(image_path)
