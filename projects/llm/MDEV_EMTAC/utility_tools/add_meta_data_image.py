import csv
import os
import pyexiv2


"""
This script reads metadata from a CSV file and inserts it into image files located in the same directory.
The CSV file must contain a column with 'file name' (case-insensitive), which specifies the target images.
Metadata from each CSV row is combined into a single string and added to each corresponding image's EXIF 'ImageDescription' tag using pyexiv2.
"""


# Function to load metadata from CSV file
def load_metadata_from_csv(csv_file_path):
    metadata = []
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata.append(row)
    return metadata

# Function to find the key containing "file name"
def find_image_path_key(meta):
    for key in meta.keys():
        if 'file name' in key.lower():
            return key
    return None

# Function to add metadata to images
def add_metadata_to_images(metadata, csv_file_folder):
    for meta in metadata:
        image_path_key = find_image_path_key(meta)
        if image_path_key is not None:
            image_path = os.path.join(csv_file_folder, meta[image_path_key])
            metadata_string = ", ".join(f"{key}: {value}" for key, value in meta.items())
            metadata_dict = {'Exif.Image.ImageDescription': metadata_string}
            try:
                image = pyexiv2.Image(image_path)
                image.modify_exif(metadata_dict)
                image.close()
                print(f"Metadata added to {image_path} successfully.")
            except Exception as e:
                print(f"Error adding metadata to {image_path}: {e}")
        else:
            print("Error: 'file name' key not found in metadata dictionary.")

# Main function
def main():
    csv_file_path = input("Enter the path of the CSV file: ")
    csv_file_folder = os.path.dirname(csv_file_path)
    metadata = load_metadata_from_csv(csv_file_path)
    add_metadata_to_images(metadata, csv_file_folder)

if __name__ == "__main_add_metta_data_image__":
    main()
