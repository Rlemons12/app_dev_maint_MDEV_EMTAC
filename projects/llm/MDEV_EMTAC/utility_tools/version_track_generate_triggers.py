from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL

# Define your tables and columns here
tables_and_columns = {
    'drawings':['id', 'drw_equipment_name', 'drw_number','drw_name', 'drw_revision', 'drw_spare_part_number'],
    'parts': ['id', 'part_number', 'name', 'description', 'documentation','rev'],
    'multi_entity_document': ['id', 'title', 'area_id', 'equipment_group_id', 'model_id', 'asset_number_id', 'location_id','rev'],
    'area': ['id', 'name', 'description','rev'],
    'equipment_group': ['id', 'name', 'area_id', 'rev'],
    'model': ['id', 'name', 'description','rev'],
    'asset_number': ['id', 'number', 'model_id', 'description', 'rev'],
    'location': ['id', 'name', 'equipment_group_id', 'model_id','rev'],
    'problem': ['id', 'name', 'description', 'location_id', 'model_id', 'asset_number_id','rev'],
    'complete_document': ['id', 'title', 'file_path', 'content', 'area_id', 'equipment_group_id', 'model_id', 'asset_number_id', 'location_id','rev'],
    'solution': ['id', 'description', 'problem_id', 'document_id', 'image_id','rev'],
    'images': ['id', 'title', 'description', 'image_blob', 'area_id', 'equipment_group_id', 'model_id', 'asset_number_id', 'complete_document_id', 'location_id', 'problem_id', 'part_id','rev'],
    'documents': ['id', 'title', 'area_id', 'equipment_group_id', 'model_id', 'asset_number_id', 'location_id', 'content', 'complete_document_id', 'embedding','rev'],
    'powerpoints': ['id', 'title', 'area', 'equipment_group', 'model', 'asset_number', 'ppt_file_path', 'pdf_file_path', 'description', 'complete_document_id','rev'],
    'users': ['id', 'employee', 'first_name', 'last_name', 'current_shift', 'primary_area', 'age', 'education_level', 'start_date', 'hashed_password','rev']
}

# Database setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=True)  # Logging SQL statements executed by SQLAlchemy
Session = sessionmaker(bind=engine)
session = scoped_session(Session)

def create_audit_table():
    """
    Create the audit table if it doesn't exist.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY,
        table_name TEXT NOT NULL,
        record_id INTEGER NOT NULL,
        action TEXT NOT NULL,
        previous_values TEXT,
        user_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with engine.connect() as connection:
        connection.execute(text(create_table_sql))

def drop_audit_table():
    """
    Drop the audit table if it exists.
    """
    drop_table_sql = """
    DROP TABLE IF EXISTS audit_log;
    """
    with engine.connect() as connection:
        connection.execute(text(drop_table_sql))

def drop_existing_triggers():
    """
    Drop existing triggers if they exist.
    """
    with engine.connect() as connection:
        for table in tables_and_columns:
            update_trigger_name = f"{table}_update_audit"
            delete_trigger_name = f"{table}_delete_audit"
            connection.execute(text(f"DROP TRIGGER IF EXISTS {update_trigger_name};"))
            connection.execute(text(f"DROP TRIGGER IF EXISTS {delete_trigger_name};"))

# Drop existing audit table and triggers
drop_audit_table()
drop_existing_triggers()

# Create the audit table
create_audit_table()

def trigger_exists(trigger_name):
    """
    Check if a trigger with the given name already exists in the database.
    """
    query = f"SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='{trigger_name}';"
    with engine.connect() as connection:
        result = connection.execute(text(query))
        return bool(result.fetchone())

def generate_audit_triggers(table_name, columns):
    """
    Generate SQL for triggers that audit updates, deletions, and insertions of records in a table.
    
    Args:
    table_name: Name of the database table.
    columns: A list of column names to be included in the audit.
    
    Returns:
    A tuple of three strings, containing the SQL commands for the update, delete, and insert triggers.
    """
    print(f"Generating triggers for table: {table_name}")
    print("Columns:", columns)

    update_trigger_name = f"{table_name}_update_audit"
    delete_trigger_name = f"{table_name}_delete_audit"
    insert_trigger_name = f"{table_name}_insert_audit"
    
    update_trigger_sql = None
    delete_trigger_sql = None
    insert_trigger_sql = None

    if not trigger_exists(update_trigger_name):
        update_trigger_sql = f"""
        CREATE TRIGGER {update_trigger_name}
        AFTER UPDATE ON {table_name}
        FOR EACH ROW
        BEGIN
            UPDATE {table_name}
            SET rev = CASE
                            WHEN OLD.rev IS NULL THEN 1
                            ELSE CAST(OLD.rev AS INTEGER) + 1
                       END
            WHERE id = NEW.id;

            INSERT INTO audit_log (table_name, record_id, action, previous_values)
            VALUES (
                '{table_name}',
                NEW.id,
                'UPDATE',
                {json_object(columns, 'OLD')}
            );
        END;
        """
        print("Update Trigger SQL:", update_trigger_sql)

    if not trigger_exists(delete_trigger_name):
        delete_trigger_sql = f"""
        CREATE TRIGGER {delete_trigger_name}
        BEFORE DELETE ON {table_name}
        FOR EACH ROW
        BEGIN
            INSERT INTO audit_log (table_name, record_id, action, previous_values)
            VALUES (
                '{table_name}',
                OLD.id,
                'DELETE',
                {json_object(columns, 'OLD')}
            );
        END;
        """
        print("Delete Trigger SQL:", delete_trigger_sql)
    
    if not trigger_exists(insert_trigger_name):
        insert_trigger_sql = f"""
        CREATE TRIGGER {insert_trigger_name}
        AFTER INSERT ON {table_name}
        FOR EACH ROW
        BEGIN
            INSERT INTO audit_log (table_name, record_id, action)
            VALUES (
                '{table_name}',
                NEW.id,
                'INSERT'
            );
        END;
        """
        print("Insert Trigger SQL:", insert_trigger_sql)

    return update_trigger_sql, delete_trigger_sql, insert_trigger_sql



def json_object(columns, row_alias):
    """
    Generate a string representing a JSON object with column names and values from a row.
    
    Args:
    columns: A list of column names.
    row_alias: The alias for the row, typically 'OLD' or 'NEW'.
    
    Returns:
    A string representing the JSON object for inclusion in an SQL statement.
    """
    return ' || \', \' || '.join([f"quote('{col}: ' || {row_alias}.{col})" for col in columns])

# Generate new triggers
triggers_sql = []
for table, columns in tables_and_columns.items():
    print(f"Generating triggers for table: {table}")
    update_trigger, delete_trigger, insert_trigger = generate_audit_triggers(table, columns)  # Unpack three values
    if update_trigger:
        triggers_sql.append(update_trigger)
    if delete_trigger:
        triggers_sql.append(delete_trigger)
    if insert_trigger:
        triggers_sql.append(insert_trigger)


# Apply new triggers
with engine.connect() as connection:
    for sql_command in triggers_sql:
        if sql_command:
            connection.execute(text(sql_command))

print("Triggers and audit table have been replaced or created.")
