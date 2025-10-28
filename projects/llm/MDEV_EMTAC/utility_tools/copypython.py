import shutil
import os

def copy_python_environment(source_directory, destination_directory):
    if not os.path.exists(source_directory):
        print(f"Source directory not found: {source_directory}")
        return
    
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    base_name = os.path.basename(source_directory)
    destination_path = os.path.join(destination_directory, base_name)

    # Check if destination directory already exists, and create a unique name if so
    counter = 1
    while os.path.exists(destination_path):
        destination_path = os.path.join(destination_directory, f"{base_name}_{counter}")
        counter += 1

    print(f"Starting to copy Python environment from {source_directory} to {destination_path}...")
    
    # Custom function to print the status during the copy
    def _copy_progress(src, dst):
        print(f"Copying {src} to {dst}")
    
    # Copy the entire directory and show progress
    shutil.copytree(source_directory, destination_path, copy_function=_copy_progress)
    
    print(f"Successfully copied Python environment to {destination_path}")

# Your specific Python environment directory
source_directory = "C:\\Users\\10169062\\AppData\\Local\\Programs\\Python\\Python311"

# Replace 'D:\\Python' with the path to your desired destination directory
destination_directory = "D:\\Python"

copy_python_environment(source_directory, destination_directory)
