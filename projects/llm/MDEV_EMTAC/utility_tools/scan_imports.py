import os
import re
import sys

# Get the current script directory path
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Navigate one level up to locate the requirements.txt file
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Add PARENT_DIR to the Python path
sys.path.append(PARENT_DIR)

def find_imports(folder_path):
    # Initialize a set to store unique package names
    packages = set()
    
    # Regular expression pattern to match Python import statements
    pattern = re.compile(r"^\s*import (\w+)|^\s*from (\w+) import")

    # Loop through each file in the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        match = pattern.match(line)
                        if match:
                            # Add the package name to the set
                            # Group 1 is for "import package"
                            # Group 2 is for "from package import"
                            package = match.group(1) or match.group(2)
                            packages.add(package)

    return packages

if __name__ == "__main__":
    folder_path = SCRIPT_DIR  # The folder containing your .py files; set to the current script directory
    packages = find_imports(folder_path)
    
    # Print out the packages
    print("You may need to install the following packages:")
    for package in packages:
        print(package)

    # Create a requirements.txt file
    requirements_file = os.path.join(PARENT_DIR, "requirements.txt")
    with open(requirements_file, "w") as f:
        for package in packages:
            f.write(f"{package}\n")

    print("A requirements.txt file has been created at:", requirements_file)
