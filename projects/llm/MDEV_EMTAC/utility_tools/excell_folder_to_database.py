import os
import sqlite3
import pandas as pd
import glob  # for file handling

# Your defined paths
BASE_DIR = r'C:\Users\10169062\Desktop\AI_CETACr10m5'
DOCS_PATH = os.path.join(BASE_DIR, 'Database', 'MS_Access')  # Excel files location
DATABASE_DIR = os.path.join(BASE_DIR, 'Database')  # Database directory
DATABASE_PATH = os.path.join(DATABASE_DIR, 'chatbot_data.db')  # Database file

# Check if database directory exists, create if not
if not os.path.exists(DATABASE_DIR):
    os.makedirs(DATABASE_DIR)

# Connect to SQLite database
conn = sqlite3.connect(DATABASE_PATH)

# Iterate over all Excel files in the specified folder
for excel_path in glob.glob(os.path.join(DOCS_PATH, '*.xlsx')):
    # Get all sheet names in the Excel file
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    # Iterate over all sheet names
    for sheet_name in sheet_names:
        # Read Excel sheet into a DataFrame
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        # Set the table name to be the same as the sheet name
        table_name = sheet_name  # Setting table name as the sheet name

        # Create table
        df.to_sql(table_name, conn, if_exists='replace', index=False)

# Commit changes and close connection
conn.commit()
conn.close()

print("All Excel files and sheets have been processed.")
