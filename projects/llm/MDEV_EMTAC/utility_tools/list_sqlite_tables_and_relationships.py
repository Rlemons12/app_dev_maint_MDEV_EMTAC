import sqlite3
import re

# Connect to SQLite database
conn = sqlite3.connect("chatbot_data.db")
cursor = conn.cursor()

# Open a text file for writing output
with open("sqlite_table_relationships.txt", "w") as f:

    # Fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Fetch and parse table creation SQL to find foreign keys
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        create_sql = cursor.fetchone()[0]
        
        f.write(f"Table: {table_name}\n")
        
        foreign_keys = re.findall("FOREIGN KEY\s*\(.*\)", create_sql)
        
        if foreign_keys:
            for fk in foreign_keys:
                f.write(f"  {fk}\n")
        else:
            f.write("  No foreign keys found.\n")

    # Close the database connection
    conn.close()

    print("Output written to sqlite_table_relationships.txt")
