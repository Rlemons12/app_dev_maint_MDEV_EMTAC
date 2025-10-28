import pandas as pd
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import os
import sys

# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configurations
from modules.configuration.config import DB_LOADSHEET, DATABASE_URL

# Define the ORM class
Base = declarative_base()

class KeywordAction(Base):
    __tablename__ = 'keyword_actions'

    id = Column(Integer, primary_key=True)
    keyword = Column(String, unique=True)
    action = Column(String)

# Function to load keywords from an Excel file
def load_keywords_from_excel(excel_path, database_url):
    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Connect to the database
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)  # Create tables if they don't exist
    Session = sessionmaker(bind=engine)
    session = Session()

    # Iterate over the rows of the DataFrame and create KeywordAction instances
    for index, row in df.iterrows():
        try:
            keyword_action = KeywordAction(keyword=row['keyword'], action=row['action'])
            session.add(keyword_action)
            session.commit()
        except IntegrityError:
            # If a keyword is already present, the unique constraint is violated.
            # You can choose to ignore, update, or raise.
            session.rollback()
            print(f"Keyword '{row['keyword']}' already exists. Skipping.")

    # Close the session
    session.close()
    print("All new keywords and actions have been loaded into the database.")

# Usage
if __name__ == "__main__":
    # Path to the keywords Excel file
    excel_file_path = os.path.join(DB_LOADSHEET, "load_keywords_file.xlsx")

    # Load the keywords into the database
    load_keywords_from_excel(excel_file_path, DATABASE_URL)
