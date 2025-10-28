import os
import shutil
import pillow_heif
from PIL import Image
from openpyxl import Workbook


"""
This script processes images in a user-specified folder by performing the following steps:
1. Converts any .HEIC images to .JPG format using `pillow_heif`, and copies non-HEIC images to a 'converted' subfolder.
2. Extracts predefined EXIF metadata from each processed image using the `Pillow` library.
3. Generates an Excel workbook with two sheets:
   - 'Image Load Sheet': Lists image names, a blank column for new image names, and extracted EXIF tag values.
   - 'EXIF Reference': Describes each supported EXIF tag, including its name, tag ID, IFD section, data type, and description.

The final Excel file, `exif_load_sheet_with_reference.xlsx`, is saved in the 'converted' folder alongside the processed images.
"""


# ----- Configuration -----
# Predefined EXIF tags (for both reference and load sheet)
exif_tags = [
    {"Tag Name": "ImageDescription", "Tag ID": "0x010E", "IFD": "0th", "Data Type": "ASCII", "Description": "Image title or caption"},
    {"Tag Name": "Make", "Tag ID": "0x010F", "IFD": "0th", "Data Type": "ASCII", "Description": "Manufacturer of the recording equipment"},
    {"Tag Name": "Model", "Tag ID": "0x0110", "IFD": "0th", "Data Type": "ASCII", "Description": "Model name of the recording equipment"},
    {"Tag Name": "Orientation", "Tag ID": "0x0112", "IFD": "0th", "Data Type": "SHORT", "Description": "Orientation of the image"},
    {"Tag Name": "XResolution", "Tag ID": "0x011A", "IFD": "0th", "Data Type": "RATIONAL", "Description": "Pixels per unit in the X direction"},
    {"Tag Name": "YResolution", "Tag ID": "0x011B", "IFD": "0th", "Data Type": "RATIONAL", "Description": "Pixels per unit in the Y direction"},
    {"Tag Name": "ResolutionUnit", "Tag ID": "0x0128", "IFD": "0th", "Data Type": "SHORT", "Description": "Unit for XResolution and YResolution"},
    {"Tag Name": "Software", "Tag ID": "0x0131", "IFD": "0th", "Data Type": "ASCII", "Description": "Software used for processing the image"},
    {"Tag Name": "DateTime", "Tag ID": "0x0132", "IFD": "0th", "Data Type": "ASCII", "Description": "File change date and time"},
    {"Tag Name": "Artist", "Tag ID": "0x013B", "IFD": "0th", "Data Type": "ASCII", "Description": "Artist or creator of the image"},
    {"Tag Name": "Copyright", "Tag ID": "0x8298", "IFD": "0th", "Data Type": "ASCII", "Description": "Copyright information"},
    {"Tag Name": "ExposureTime", "Tag ID": "0x829A", "IFD": "Exif", "Data Type": "RATIONAL", "Description": "Exposure time"},
    {"Tag Name": "FNumber", "Tag ID": "0x829D", "IFD": "Exif", "Data Type": "RATIONAL", "Description": "F number (aperture)"},
    {"Tag Name": "ExposureProgram", "Tag ID": "0x8822", "IFD": "Exif", "Data Type": "SHORT", "Description": "Exposure program used"},
    {"Tag Name": "ISOSpeedRatings", "Tag ID": "0x8827", "IFD": "Exif", "Data Type": "SHORT", "Description": "ISO speed rating"},
    {"Tag Name": "ExifVersion", "Tag ID": "0x9000", "IFD": "Exif", "Data Type": "UNDEFINED", "Description": "Exif version"},
    {"Tag Name": "DateTimeOriginal", "Tag ID": "0x9003", "IFD": "Exif", "Data Type": "ASCII", "Description": "Original date and time of image creation"},
    {"Tag Name": "DateTimeDigitized", "Tag ID": "0x9004", "IFD": "Exif", "Data Type": "ASCII", "Description": "Digitization date and time"},
    {"Tag Name": "ShutterSpeedValue", "Tag ID": "0x9201", "IFD": "Exif", "Data Type": "SRATIONAL", "Description": "Shutter speed value"},
    {"Tag Name": "ApertureValue", "Tag ID": "0x9202", "IFD": "Exif", "Data Type": "RATIONAL", "Description": "Lens aperture value"},
    {"Tag Name": "FocalLength", "Tag ID": "0x920A", "IFD": "Exif", "Data Type": "RATIONAL", "Description": "Focal length of the lens"},
    {"Tag Name": "UserComment", "Tag ID": "0x9286", "IFD": "Exif", "Data Type": "UNDEFINED", "Description": "User comments"}
]

def extract_exif_data(img):
    """Extract EXIF data from a Pillow image and return a mapping of tag names to values."""
    exif_data = img._getexif()  # Returns a dict with tag IDs as keys.
    exif_values = {}
    if exif_data:
        for tag in exif_tags:
            tag_id_int = int(tag["Tag ID"], 16)
            value = exif_data.get(tag_id_int, "")
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8', errors='replace')
                except Exception:
                    value = str(value)
            exif_values[tag["Tag Name"]] = value
    else:
        for tag in exif_tags:
            exif_values[tag["Tag Name"]] = ""
    return exif_values

# ----- Prompt for Folder -----
folder_path = input("Please enter the folder path containing image files: ").strip()

if not os.path.isdir(folder_path):
    print("The provided folder does not exist. Please check the path and try again.")
    exit(1)

# ----- Setup Conversion Subfolder -----
subfolder_name = "converted"
output_folder = os.path.join(folder_path, subfolder_name)
os.makedirs(output_folder, exist_ok=True)

# ----- Process Files: Convert or Copy to Output Folder -----
# For each file in the folder (non-recursive), if it's HEIC, convert to JPEG;
# if not, copy the file to the output folder.
processed_files = []  # List to store full paths of files in the output folder

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    # Skip directories and the output folder itself.
    if os.path.isdir(file_path) or filename == subfolder_name:
        continue

    lower_name = filename.lower()
    if lower_name.endswith('.heic'):
        # Convert HEIC to JPEG
        try:
            heif_file = pillow_heif.read_heif(file_path)
            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
            # Create a new filename with .jpg extension.
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            dest_path = os.path.join(output_folder, new_filename)
            img.save(dest_path, 'JPEG')
            print(f"Converted {file_path} to {dest_path}")
            processed_files.append(dest_path)
        except Exception as e:
            print(f"Failed to convert {file_path}: {e}")
            continue
    else:
        # Copy non-HEIC file to the output folder.
        try:
            dest_path = os.path.join(output_folder, filename)
            shutil.copy2(file_path, dest_path)
            print(f"Copied {file_path} to {dest_path}")
            processed_files.append(dest_path)
        except Exception as e:
            print(f"Failed to copy {file_path}: {e}")
            continue

# ----- Create Excel Workbook with Two Sheets -----
wb = Workbook()

# Sheet 1: Image Load Sheet
load_sheet = wb.active
load_sheet.title = "Image Load Sheet"
# Header: "Image Name", "New Image Name", then each EXIF tag name.
load_headers = ["Image Name", "New Image Name"] + [tag["Tag Name"] for tag in exif_tags]
load_sheet.append(load_headers)

# Sheet 2: EXIF Reference
ref_sheet = wb.create_sheet(title="EXIF Reference")
ref_headers = ["Tag Name", "Tag ID", "IFD", "Data Type", "Description"]
ref_sheet.append(ref_headers)
for tag in exif_tags:
    ref_sheet.append([tag["Tag Name"], tag["Tag ID"], tag["IFD"], tag["Data Type"], tag["Description"]])

# ----- Extract EXIF Data and Populate the Load Sheet -----
for proc_file in processed_files:
    try:
        with Image.open(proc_file) as img:
            exif_values = extract_exif_data(img)
    except Exception as e:
        print(f"Failed to open {proc_file}: {e}")
        continue

    # Create a row for the Excel load sheet:
    # Column 1: the image name (basename from the output folder)
    # Column 2: New Image Name (left blank for later editing)
    # Columns 3+: corresponding EXIF tag values.
    row = [os.path.basename(proc_file), ""] + [exif_values.get(tag["Tag Name"], "") for tag in exif_tags]
    load_sheet.append(row)

# Save the Excel workbook in the output folder.
excel_file = os.path.join(output_folder, "exif_load_sheet_with_reference.xlsx")
wb.save(excel_file)
print(f"Excel workbook saved as '{excel_file}'.")
