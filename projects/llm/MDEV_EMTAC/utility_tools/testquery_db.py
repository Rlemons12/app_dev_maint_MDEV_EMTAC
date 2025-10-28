from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from modules.configuration.config import REVISION_CONTROL_DB_PATH
# Revision control database configuration

revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')
RevisionControlBase = declarative_base()
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))  # Use distinct name
revision_control_session = RevisionControlSession()

# Inspect the database for tables
inspector = inspect(revision_control_engine)
tables = inspector.get_table_names()

# Print the tables
print("Tables in the revision control database:")
for table in tables:
    print(table)
