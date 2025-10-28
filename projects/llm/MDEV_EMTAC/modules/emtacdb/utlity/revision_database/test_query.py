# test_query.py
from sqlalchemy import inspect
from modules.configuration.config_env import DatabaseConfig

def list_tables(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return tables

def main():
    # Instantiate the database configuration
    db_config = DatabaseConfig()

    # Query tables in the main database
    print("Tables in the Main Database:")
    main_tables = list_tables(db_config.main_engine)
    for table in main_tables:
        print(f" - {table}")

    # Query tables in the revision control database
    print("\nTables in the Revision Control Database:")
    revision_tables = list_tables(db_config.revision_control_engine)
    for table in revision_tables:
        print(f" - {table}")

if __name__ == "__main__":
    main()
