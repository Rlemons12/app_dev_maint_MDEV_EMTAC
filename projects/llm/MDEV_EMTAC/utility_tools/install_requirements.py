import subprocess
import os
import sys
import logging

# Configure logging
logging.basicConfig(filename='install_requirements.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def install_requirements(requirements_path):
    """
    Install packages from a requirements.txt file.
    """
    print(f"Attempting to install from: {requirements_path}")
    if not os.path.exists(requirements_path):
        logging.error(f"Requirements file does not exist: {requirements_path}")
        print("Requirements file does not exist.")
        return

    try:
        # Execute pip install command and display output directly
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path], stdout=sys.stdout, stderr=sys.stderr)
        logging.info(f"Successfully installed packages from {requirements_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install packages from {requirements_path}: {e}")
        print(f"Failed to install packages: {e}")



# Configure logging
logging.basicConfig(filename='generate_requirements.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def find_python_files(directory):
    """
    Recursively find all Python files in the given directory.
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
                logging.info(f"Found Python file: {file_path}")
    return python_files

def extract_imports(file_path):
    """
    Extract import statements from a Python file.
    """
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    parts = line.split()
                    if len(parts) > 1:
                        imports.add(parts[1].split('.')[0])
        logging.info(f"Extracted imports from {file_path}")
    except UnicodeDecodeError:
        logging.error(f"Failed to read {file_path} due to encoding error")
    except Exception as e:
        logging.error(f"An error occurred while processing {file_path}: {e}")
    return imports

def generate_requirements(source_dir, output_dir):
    """
    Generate a requirements list by analyzing import statements in Python files.
    """
    all_imports = set()
    python_files = find_python_files(source_dir)
    for python_file in python_files:
        all_imports.update(extract_imports(python_file))
    
    logging.info("Collected all imports, now generating requirements.txt")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Path to the final requirements file
    final_requirements_path = os.path.join(output_dir, 'requirements.txt')

    # Capture the output of pip freeze
    try:
        pip_freeze_output = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')
        installed_packages = dict(line.split('==') for line in pip_freeze_output.split('\n') if '==' in line)
        logging.info("Successfully captured pip freeze output")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to run pip freeze: {e}")
        return

if __name__ == '__main__':
    logging.info("Installation script started")
    # Define the path to the requirements file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Correct the path to go back to AuMaintdb and then into static/project_requirements
    project_directory = os.path.abspath(os.path.join(script_directory, '..'))  # Go up one level to AuMaintdb
    requirements_path = os.path.join(project_directory, 'static', 'project_requirements', 'requirements.txt')

    # Install the packages from the requirements file
    install_requirements(requirements_path)
    logging.info("Installation script completed")
    print("Installation completed.")

