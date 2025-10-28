import shutil

# Prompt the user for the source and destination paths
print("Please enter the file location and destination.")
source_path = input("Enter the source file path: ")
destination_path = input("Enter the destination file path: ")

# Copy the file
try:
    shutil.copy(source_path, destination_path)
    print(f'File copied successfully to {destination_path}')
except Exception as e:
    print(f'Error copying file: {e}')
