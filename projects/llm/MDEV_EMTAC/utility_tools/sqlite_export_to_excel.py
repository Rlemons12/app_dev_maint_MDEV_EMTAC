import sqlite3
import pandas as pd
from modules.configuration.config import DATABASE_PATH  # Ensure DATABASE_PATH is defined in config.py


# Connect to the SQLite database
conn = sqlite3.connect(DATABASE_PATH)

# List all tables in the database
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Create a writer object for Excel
with pd.ExcelWriter("output_multisheet.xlsx") as writer:
    for table in tables:
        table_name = table[0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df.to_excel(writer, sheet_name=table_name, index=False)

# Close the SQLite database connection
conn.close()
