"""
create_main_tables_only.py
--------------------------------------------------------
Creates only the main EMTAC database tables defined in
modules.emtacdb.emtacdb_fts.Base using SQLAlchemy.

Uses the custom EMTAC logger for structured, request-ID-aware logging.

Usage:
> E:\emtac\projects\llm\MDEV_EMTAC\.venv\Scripts\python.exe modules\initial_setup\create_main_tables_only.py
"""

import os
from sqlalchemy import inspect
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Base as MainBase
from modules.configuration.log_config import info_id, warning_id, error_id, get_request_id

def create_main_tables():
    """
    Create all tables for the main EMTAC database only.
    """
    request_id = get_request_id()
    info_id("===============================================", request_id)
    info_id(" EMTAC MAIN DATABASE TABLE CREATION UTILITY", request_id)
    info_id("===============================================", request_id)

    try:
        # Initialize DB engine using your DatabaseConfig class
        db_config = DatabaseConfig()
        main_engine = db_config.main_engine

        info_id("Connected to main database.", request_id)
        info_id(f"Database URL: {main_engine.url}", request_id)

        # Create all tables declared in emtacdb_fts
        info_id("Creating EMTAC tables...", request_id)
        MainBase.metadata.create_all(main_engine)
        info_id("All EMTAC tables created successfully.", request_id)

        # Verify what was created
        inspector = inspect(main_engine)
        tables = inspector.get_table_names()
        info_id(f"Verified {len(tables)} tables in main database.", request_id)
        for t in tables:
            info_id(f" ├─ {t}", request_id)

        info_id("✅ EMTAC main database tables are ready.", request_id)
        info_id("Revision control tables were intentionally skipped.", request_id)

    except Exception as e:
        error_id(f"❌ Failed to create EMTAC tables: {e}", request_id)
        raise  # Let higher-level scripts catch or log further if needed

if __name__ == "__main__":
    create_main_tables()
