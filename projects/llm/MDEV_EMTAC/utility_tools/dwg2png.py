import win32com.client
import pythoncom
import os

def save_dwg_as_image(solidworks_app, dwg_file_path, image_file_path):
    pythoncom.CoInitialize()  # Initialize the COM library if in a thread
    
    # Ensure SolidWorks is visible
    solidworks_app.Visible = True

    # Open a new document (drawing)
    # Ensure the path to the template is correct and accessible
    template_path = 'C:\\ProgramData\\SolidWorks\\SOLIDWORKS 2021\\templates\\Drawing.drwdot'
    swDoc = solidworks_app.NewDocument(template_path, 0, 0, 0)
    if swDoc is None:
        print("Failed to create a new drawing document.")
        return

    # Handling DWG file import into the drawing
    # You need to find the appropriate method for importing DWG files here

    # Attempt to save the current document as an image
    # This part needs adjustment based on SolidWorks API for exporting to PNG
    print("Adjust the script to use the correct SolidWorks method to save or export as PNG.")

    # Close the document
    solidworks_app.CloseDoc(swDoc.GetTitle())

# Main code
if __name__ == "__main__":
    try:
        # Attempt to connect to an existing instance of SolidWorks
        swApp = win32com.client.Dispatch("SldWorks.Application")
    except Exception as e:
        print("Error: Could not connect to SolidWorks.")
        print(e)
        exit()

    # Prompt for the DWG file path
    dwg_file = input("Enter the path to the DWG file: ")

    # Construct the output image file path in the same folder as the DWG file
    image_file = os.path.splitext(dwg_file)[0] + ".png"

    # Call the function to attempt the conversion
    save_dwg_as_image(swApp, dwg_file, image_file)
