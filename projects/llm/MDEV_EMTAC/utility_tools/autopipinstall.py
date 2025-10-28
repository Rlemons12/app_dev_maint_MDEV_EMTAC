import subprocess
import importlib
import os
import sys

# Get the current script directory path
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Navigate one level up to locate the requirements.txt file
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Add PARENT_DIR to the Python path
sys.path.append(PARENT_DIR)

# Function to install a package using pip
def install_package(package):
    subprocess.call([sys.executable, '-m', 'pip', 'install', package])

# Function to check if a package is installed
def is_installed(package):
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False

# List of required packages
packages = [
    'flask',
    'spacy',
    'comtypes',
    'Levenshtein',
    'sqlalchemy',
    'openai==0.28',  # Specific version of OpenAI package
    'docx2pdf',
    'nltk',
    'requests',
    'simplejson',
    'pdfplumber',
    'fuzzywuzzy==0.18.0',  # Specific version
    'pywin32',
    'python-docx',
    'python-pptx',
    'pandas',  # Already includes the latest version
    'openpyxl',
    'flask-bcrypt',
    'pyarrow',  # Already includes the latest version
    'PyMuPDF',
    'fitz',
    'fuzzywuzzy[speedup]',
    'frontend',
    'tools',
    'PyPDF2',
    'flask_sqlalchemy',
    'numpy==1.26.4',
    'docx'
]

# Check and install required packages
for package in packages:
    package_name = package.split('==')[0]  # Get the package name without the version specifier
    if not is_installed(package_name):
        print(f"Installing {package}...")
        install_package(package)
        print(f"{package} installed successfully!")
    else:
        print(f"{package} is already installed.")

# Check if the spaCy model is installed, if not, download it
try:
    import spacy
    spacy.load('en_core_web_sm')
    print("spaCy model 'en_core_web_sm' is already installed.")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    subprocess.call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    print("spaCy model 'en_core_web_sm' downloaded successfully!")

print("All required packages are installed.")
