import os
import subprocess
import sys
import logging

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

    # Write the filtered imports with versions to the requirements file
    with open(final_requirements_path, 'w') as final_file:
        for imp in sorted(all_imports):
            if imp in installed_packages:
                final_file.write(f"{imp}=={installed_packages[imp]}\n")
        logging.info(f"Requirements file generated at: {final_requirements_path}")

if __name__ == '__main__':
    logging.info("Script started")
    # Define paths relative to the script's location
    script_directory = os.path.dirname(os.path.abspath(__file__))
    source_directory = os.path.abspath(os.path.join(script_directory, '..'))
    output_directory = os.path.abspath(os.path.join(source_directory, 'static', 'project_requirements'))

    # Run the requirements generation
    generate_requirements(source_directory, output_directory)
    logging.info("Script completed successfully")
