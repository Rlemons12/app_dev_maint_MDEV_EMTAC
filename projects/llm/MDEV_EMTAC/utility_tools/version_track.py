from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL

# Database setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=True)  # Logging SQL statements executed by SQLAlchemy
Session = sessionmaker(bind=engine)
session = scoped_session(Session)

# Define the audit_log table with SQLite-compatible syntax
create_audit_log_table = text("""
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT,
    record_id INTEGER,
    action TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    previous_values TEXT
)
""")

# Creating the audit_log table
try:
    print("Creating the audit_log table...")
    session.execute(create_audit_log_table)
    session.commit()
    print("Audit_log table created successfully.")
except Exception as e:
    session.rollback()
    print(f"Error while creating the audit_log table: {e}")

# Define tables to audit
tables_to_audit = [
    "area", "asset_number", "chat_sessions", "complete_document",
    "complete_document_multi_entity_document_association", "documents",
    "drawing_part_image_model_location_association", "drawings",
    "equipment_group", "file_logs", "image_problem", "image_solution",
    "images", "location", "model", "multi_entity_document",
    "parts", "powerpoints", "problem", "problem_complete_document",
    "qanda", "solution", "solution_complete_document", "users"
]

# Define and create triggers for each table to be audited
try:
    for table in tables_to_audit:
        print(f"Setting up triggers for {table}...")
        # Drop existing triggers if they exist
        session.execute(text(f"DROP TRIGGER IF EXISTS audit_{table}_insert;"))
        session.execute(text(f"DROP TRIGGER IF EXISTS audit_{table}_update;"))
        session.execute(text(f"DROP TRIGGER IF EXISTS audit_{table}_delete;"))

        # Recreate triggers
        session.execute(text(f"""
        CREATE TRIGGER audit_{table}_insert AFTER INSERT ON {table}
        BEGIN
            INSERT INTO audit_log (table_name, record_id, action, user_id, previous_values)
            VALUES ('{table}', NEW.id, 'INSERT', 'user_id_placeholder', '');
        END;
        """))
        session.execute(text(f"""
        CREATE TRIGGER audit_{table}_update AFTER UPDATE ON {table}
        BEGIN
            INSERT INTO audit_log (table_name, record_id, action, user_id, previous_values)
            VALUES ('{table}', NEW.id, 'UPDATE', 'user_id_placeholder', 'Old Value: ' || OLD.column_name);
        END;
        """))
        session.execute(text(f"""
        CREATE TRIGGER audit_{table}_delete BEFORE DELETE ON {table}
        BEGIN
            INSERT INTO audit_log (table_name, record_id, action, user_id, previous_values)
            VALUES ('{table}', OLD.id, 'DELETE', 'user_id_placeholder', 'Old Value: ' || OLD.column_name);
        END;
        """))
    session.commit()
    print("Triggers have been created successfully for all tables.")
except Exception as e:
    session.rollback()
    print(f"Error while creating triggers: {e}")
finally:
    session.close()
    print("Database session closed.")
