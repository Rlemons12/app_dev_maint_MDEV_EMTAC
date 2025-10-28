# blueprints/generate_requirements.py
import os
import subprocess
import sys
from flask import Blueprint, jsonify

# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))



generate_requirements_bp = Blueprint('generate_requirements_bp', __name__)

@generate_requirements_bp.route('/generate_requirements')
def find_python_files(directory):
    """
    Recursively find all Python files in the given directory.
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_imports(file_path):
    """
    Extract import statements from a Python file.
    """
    imports = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('import '):
                parts = line.split()
                if len(parts) > 1:
                    imports.add(parts[1].split('.')[0])
            elif line.startswith('from '):
                parts = line.split()
                if len(parts) > 1:
                    imports.add(parts[1].split('.')[0])
    return imports

def generate_requirements(directory, output_dir):
    """
    Generate a requirements list by analyzing import statements in Python files.
    """
    all_imports = set()
    python_files = find_python_files(directory)
    for python_file in python_files:
        all_imports.update(extract_imports(python_file))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Path to the temporary and final requirements files
    temp_requirements_path = os.path.join(output_dir, 'temp_requirements.txt')
    final_requirements_path = os.path.join(output_dir, 'requirements.txt')

    # Write these imports to a temporary requirements file
    with open(temp_requirements_path, 'w') as temp_file:
        for imp in all_imports:
            temp_file.write(imp + '\n')

    # Use pip to generate the full requirements file
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=open(final_requirements_path, 'w'))

    # Clean up temporary file
    os.remove(temp_requirements_path)

    print(f"Requirements file generated at: {final_requirements_path}")

@generate_requirements_bp.route('/generate_requirements', methods=['POST'])
def generate_requirements_route():
    project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Project root directory
    output_directory = os.path.join(project_directory, 'static')  # Save requirements.txt in the static folder
    generate_requirements(project_directory, output_directory)
    return jsonify({'status': 'Requirements file generated', 'path': output_directory}), 200
