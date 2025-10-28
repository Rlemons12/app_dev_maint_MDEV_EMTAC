import os
import sys
from sqlalchemy import create_engine, inspect
# Get the current script directory path
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Navigate one level up to locate the requirements.txt file
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Add PARENT_DIR to the Python path
sys.path.append(PARENT_DIR)
from modules.configuration.config import DATABASE_PATH, DATABASE_URL

# Check if the database file exists in the specified directory and create it if not
if not os.path.exists(DATABASE_PATH):
    with open(DATABASE_PATH, 'w'):
        pass

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Reflect the database tables
inspector = inspect(engine)
table_names = inspector.get_table_names()

# Print relationships
for table_name in table_names:
    print("Table:", table_name)
    foreign_keys = inspector.get_foreign_keys(table_name)
    for foreign_key in foreign_keys:
        print("  Foreign key:", foreign_key['referred_table'], "->", foreign_key['referred_columns'])
