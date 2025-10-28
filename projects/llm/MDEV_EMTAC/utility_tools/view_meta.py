from PIL import Image

def view_metadata(image_path):
    try:
        image = Image.open(image_path)
        metadata = image.info
        print("Metadata for", image_path)
        if metadata:
            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print("No metadata found.")
    except Exception as e:
        print(f"Error viewing metadata for {image_path}: {e}")

# Example usage
image_path = input("Enter the path of the PNG image file: ")
view_metadata(image_path)
