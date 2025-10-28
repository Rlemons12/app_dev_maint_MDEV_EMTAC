import openpyxl
from openpyxl import Workbook


"""
This script generates an Excel workbook containing two sheets:
1. 'Image Load Sheet': Provides a structured template for entering image file names and corresponding EXIF metadata tags.
2. 'EXIF Reference': Includes a detailed reference of commonly used EXIF metadata tags, specifying their Tag Names, Tag IDs, IFD sections, Data Types, and Descriptions.

This workbook serves as a structured tool for managing and organizing EXIF metadata before embedding it into image files.
"""


# Define the EXIF tags as provided
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

# Create a new workbook
wb = Workbook()

# --------------------------
# Sheet 1: Image Load Sheet
# --------------------------
# Rename the default sheet to "Image Load Sheet"
load_sheet = wb.active
load_sheet.title = "Image Load Sheet"

# Create header: first column for the image name, then one column per EXIF tag (using Tag Name)
load_headers = ["Image Name"] + [tag["Tag Name"] for tag in exif_tags]
load_sheet.append(load_headers)

# Optionally, add a sample row
sample_row = ["image_0001"] + ["" for _ in exif_tags]
load_sheet.append(sample_row)

# --------------------------
# Sheet 2: EXIF Reference
# --------------------------
# Create a new sheet for EXIF reference
ref_sheet = wb.create_sheet(title="EXIF Reference")

# Define header row for reference
ref_headers = ["Tag Name", "Tag ID", "IFD", "Data Type", "Description"]
ref_sheet.append(ref_headers)

# Append each tag's details
for tag in exif_tags:
    ref_sheet.append([
        tag["Tag Name"],
        tag["Tag ID"],
        tag["IFD"],
        tag["Data Type"],
        tag["Description"]
    ])

# Save the workbook to a file
excel_file = "exif_load_sheet_with_reference.xlsx"
wb.save(excel_file)
print(f"Excel workbook saved as '{excel_file}'.")
