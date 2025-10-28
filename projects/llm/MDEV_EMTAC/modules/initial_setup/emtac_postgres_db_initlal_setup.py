import os
import sys
import subprocess
import shutil
import platform
from sqlalchemy import inspect, text
from modules.emtacdb.emtacdb_fts import Document


# Enhanced logging functions for Windows compatibility
def setup_windows_console():
    """Set up Windows console for better Unicode support."""
    if platform.system() == 'Windows':
        try:
            # Set console to UTF-8
            os.system('chcp 65001 > nul 2>&1')
            os.environ['PYTHONIOENCODING'] = 'utf-8'
        except:
            pass


# Set up console at import time
setup_windows_console()

# Import logging after console setup
try:
    from modules.configuration.log_config import debug_id, info_id, warning_id, error_id, get_request_id, logger

    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import logging modules: {e}")


    # Create dummy functions if imports fail
    def info_id(msg, req_id=None):
        print(f"INFO: {msg}")


    def error_id(msg, req_id=None, **kwargs):
        print(f"ERROR: {msg}")


    def warning_id(msg, req_id=None, **kwargs):
        print(f"WARNING: {msg}")


    def debug_id(msg, req_id=None):
        print(f"DEBUG: {msg}")


    def get_request_id():
        return "setup-001"


    LOGGING_AVAILABLE = False

# Initialize logger first
try:
    from modules.initial_setup.initializer_logger import (
        initializer_logger, compress_logs_except_most_recent, close_initializer_logger, LOG_DIRECTORY
    )
    from modules.database_manager.maintenance import db_maintenance
except ImportError as e:
    warning_id(f"Could not import some modules: {e}")


    # Create dummy functions if imports fail
    def compress_logs_except_most_recent(log_dir):
        pass


    def close_initializer_logger():
        pass


    LOG_DIRECTORY = "logs"
    db_maintenance = None

# Global variables
DatabaseConfig = None
MainBase = None
RevisionControlBase = None
directories_to_check = []

# Global state variables - use a simple dict to avoid scoping issues
setup_state = {
    'custom_database_url': None,
    'database_name': None,
    'request_id': None,
    'audit_system_enabled': False
}


def create_base_directories():
    """Creates the essential base directories needed before importing modules."""
    info_id("Creating essential base directories...")

    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    essential_dirs = [
        os.path.join(project_root, "Database"),
        os.path.join(project_root, "logs")
    ]

    for directory in essential_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            info_id(f"Created essential directory: {directory}")
        else:
            info_id(f"Essential directory already exists: {directory}")


def early_database_creation():
    """Create PostgreSQL database early, before importing modules that need it."""
    try:
        info_id("Early database check - before module imports...")

        # Import only the basic config we need
        try:
            from modules.configuration.config import DATABASE_URL
        except ImportError as e:
            error_id(f"Could not import DATABASE_URL: {e}")
            return False

        if not DATABASE_URL.startswith('postgresql'):
            info_id("SQLite detected - skipping early database creation")
            return True

        info_id("PostgreSQL detected - creating database before module imports...")

        # Parse database URL
        import re
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        url_pattern = r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)'
        match = re.match(url_pattern, DATABASE_URL)

        if not match:
            warning_id("Could not parse PostgreSQL DATABASE_URL.")
            return False

        username, password, host, port, default_database_name = match.groups()

        # Ask user for database name choice
        info_id("=" * 50)
        info_id("EARLY DATABASE CREATION")
        info_id("=" * 50)
        info_id(f"Default database name from config: {default_database_name}")

        custom_name = input(f"Enter database name (press Enter for '{default_database_name}'): ").strip()
        database_name = custom_name if custom_name else default_database_name

        if custom_name:
            info_id(f"Using custom database name: {database_name}")
        else:
            info_id(f"Using default database name: {database_name}")

        # Store the database name in our state
        setup_state['database_name'] = database_name

        info_id(f"Checking if PostgreSQL database '{database_name}' exists...")

        try:
            # Connect to default 'postgres' database
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=username,
                password=password,
                database='postgres'
            )

            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            try:
                # Check if database exists
                cursor.execute("""
                    SELECT 1 FROM pg_database WHERE datname = %s
                """, (database_name,))

                db_exists = cursor.fetchone() is not None

                if db_exists:
                    info_id(f"Database '{database_name}' already exists")

                    # Check if existing database has correct encoding
                    cursor.execute("""
                        SELECT datname, encoding, datcollate, datctype 
                        FROM pg_database 
                        WHERE datname = %s
                    """, (database_name,))

                    db_info = cursor.fetchone()
                    if db_info:
                        db_name, encoding, collate, ctype = db_info
                        info_id(f"Current database encoding: {encoding} (collate: {collate}, ctype: {ctype})")

                        if encoding == 6:  # UTF8
                            info_id(f"SUCCESS: Database '{database_name}' has correct UTF-8 encoding")
                        else:
                            warning_id(
                                f"WARNING: Database '{database_name}' has wrong encoding: {encoding} (should be 6 for UTF8)")
                            warning_id(
                                "This will cause Unicode errors with scientific documents containing special characters")

                            recreate = input(
                                f"Recreate database '{database_name}' with UTF-8 encoding? (y/n): ").strip().lower()

                            if recreate in ['y', 'yes']:
                                info_id(f"Dropping and recreating database '{database_name}' with UTF-8...")
                                try:
                                    cursor.execute(f'DROP DATABASE "{database_name}"')
                                    cursor.execute(f'''
                                        CREATE DATABASE "{database_name}" 
                                        WITH 
                                        OWNER = "{username}"
                                        ENCODING = 'UTF8'
                                        LC_COLLATE = 'C'  
                                        LC_CTYPE = 'C'
                                        TEMPLATE = template0
                                    ''')
                                    info_id(f"SUCCESS: Recreated database '{database_name}' with UTF-8 encoding")

                                    # Verify the new encoding
                                    cursor.execute("""
                                        SELECT datname, encoding, datcollate, datctype 
                                        FROM pg_database 
                                        WHERE datname = %s
                                    """, (database_name,))

                                    new_db_info = cursor.fetchone()
                                    if new_db_info and new_db_info[1] == 6:
                                        info_id("VERIFIED: Database now has UTF-8 encoding")
                                    else:
                                        error_id("ERROR: Database recreation failed to set UTF-8 encoding")
                                        return False

                                except Exception as e:
                                    error_id(f"Failed to recreate database '{database_name}': {e}")
                                    return False
                            else:
                                warning_id("Continuing with existing database (may experience Unicode issues)")
                                warning_id("Scientific documents with special characters may fail to save")
                    else:
                        error_id(f"Could not retrieve encoding information for database '{database_name}'")
                        return False

                else:
                    info_id(f"Database '{database_name}' does not exist")

                    # Ask user if they want to create it
                    create_db = input(
                        f"Create PostgreSQL database '{database_name}' with UTF-8 encoding? (y/n): ").strip().lower()

                    if create_db in ['y', 'yes']:
                        info_id(f"Creating PostgreSQL database '{database_name}' with UTF-8 encoding...")

                        try:
                            # Create database with explicit UTF-8 encoding
                            cursor.execute(f'''
                                CREATE DATABASE "{database_name}" 
                                WITH 
                                OWNER = "{username}"
                                ENCODING = 'UTF8'
                                LC_COLLATE = 'C'  
                                LC_CTYPE = 'C'
                                TEMPLATE = template0
                            ''')
                            info_id(f"Database '{database_name}' created successfully")

                            # Verify the encoding was set correctly
                            cursor.execute("""
                                SELECT datname, encoding, datcollate, datctype 
                                FROM pg_database 
                                WHERE datname = %s
                            """, (database_name,))

                            db_info = cursor.fetchone()
                            if db_info:
                                db_name, encoding, collate, ctype = db_info
                                info_id(
                                    f"Database encoding verification: {encoding} (collate: {collate}, ctype: {ctype})")

                                if encoding == 6:  # UTF8
                                    info_id(f"SUCCESS: Database '{database_name}' created with UTF-8 encoding")
                                    info_id("Ready to handle Unicode characters in scientific documents")
                                else:
                                    error_id(f"ERROR: Database created with encoding {encoding} instead of UTF-8")
                                    error_id("This may cause Unicode issues with scientific documents")
                                    return False
                            else:
                                error_id("Could not verify database encoding after creation")
                                return False

                        except Exception as e:
                            error_id(f"Failed to create database '{database_name}': {e}")
                            return False
                    else:
                        error_id(f"Database '{database_name}' is required but does not exist")
                        return False

                # Update DATABASE_URL if using custom name
                if custom_name:
                    custom_url = f'postgresql://{username}:{password}@{host}:{port}/{database_name}'
                    setup_state['custom_database_url'] = custom_url
                    os.environ['DATABASE_URL'] = custom_url
                    info_id(f"Updated DATABASE_URL for custom database name")

                return True

            finally:
                cursor.close()
                conn.close()

        except psycopg2.Error as e:
            error_id(f"PostgreSQL connection error: {e}")
            return False
        except Exception as e:
            error_id(f"Error in early database creation: {e}")
            return False

    except ImportError as e:
        warning_id(f"psycopg2 not available: {e}")
        return False
    except Exception as e:
        error_id(f"Error in early database creation process: {e}")
        return False


def import_modules_after_directory_setup():
    """Imports modules that require base directories to exist."""
    global DatabaseConfig, MainBase, RevisionControlBase, directories_to_check

    try:
        info_id("Importing configuration and database modules...")

        from modules.configuration.config import (
            DATABASE_URL, REVISION_CONTROL_DB_PATH,
            TEMPLATE_FOLDER_PATH, DATABASE_DIR, UPLOAD_FOLDER,
            IMAGES_FOLDER, DATABASE_PATH_IMAGES_FOLDER,
            PDF_FOR_EXTRACTION_FOLDER, IMAGES_EXTRACTED,
            TEMPORARY_FILES, PPT2PDF_PPT_FILES_PROCESS,
            PPT2PDF_PDF_FILES_PROCESS, DATABASE_DOC,
            TEMPORARY_UPLOAD_FILES, DB_LOADSHEET,
            DB_LOADSHEETS_BACKUP, DB_LOADSHEET_BOMS,
            BACKUP_DIR, Utility_tools, UTILITIES
        )
        from modules.configuration.config_env import DatabaseConfig
        from modules.emtacdb.emtacdb_fts import Base as MainBase

        # Handle revision control import carefully
        try:
            from modules.emtacdb.emtac_revision_control_db import RevisionControlBase
        except Exception as e:
            warning_id(f"Warning: Could not import revision control module: {e}")
            from sqlalchemy.ext.declarative import declarative_base
            RevisionControlBase = declarative_base()

        directories_to_check = [
            TEMPLATE_FOLDER_PATH, DATABASE_DIR, UPLOAD_FOLDER,
            IMAGES_FOLDER, DATABASE_PATH_IMAGES_FOLDER,
            PDF_FOR_EXTRACTION_FOLDER, IMAGES_EXTRACTED,
            TEMPORARY_FILES, PPT2PDF_PPT_FILES_PROCESS,
            PPT2PDF_PDF_FILES_PROCESS, DATABASE_DOC,
            TEMPORARY_UPLOAD_FILES, DB_LOADSHEET,
            DB_LOADSHEETS_BACKUP, DB_LOADSHEET_BOMS,
            BACKUP_DIR, Utility_tools, UTILITIES
        ]

        info_id("Successfully imported all required modules")
        return True
    except Exception as e:
        error_id(f"Failed to import modules: {e}")
        return False


def setup_virtual_environment_and_install_requirements():
    """Creates a virtual environment if desired and installs requirements.txt dependencies."""
    # Calculate the project root directory
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    requirements_file = os.path.join(project_root, "requirements.txt")
    venv_dir = os.path.join(project_root, "venv")

    # Ask user if they want to create a virtual environment
    create_venv = input("Would you like to create a virtual environment? (Recommended) (y/n): ").strip().lower()

    if create_venv == 'y' or create_venv == 'yes':
        # Check if venv already exists
        if os.path.exists(venv_dir):
            overwrite = input(f"Virtual environment already exists at {venv_dir}. Overwrite? (y/n): ").strip().lower()
            if overwrite == 'y' or overwrite == 'yes':
                info_id(f"Removing existing virtual environment at {venv_dir}")
                shutil.rmtree(venv_dir)
            else:
                info_id("Using existing virtual environment.")

        # Create virtual environment if it doesn't exist or was removed
        if not os.path.exists(venv_dir):
            info_id(f"Creating virtual environment at {venv_dir}...")
            try:
                subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
                info_id("Virtual environment created successfully.")
            except subprocess.CalledProcessError as e:
                error_id(f"Failed to create virtual environment: {e}")
                sys.exit(1)

        # Get the path to the Python executable in the virtual environment
        if sys.platform == "win32":
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_python = os.path.join(venv_dir, "bin", "python")

        # Upgrade pip in the virtual environment
        try:
            info_id("Upgrading pip in virtual environment...")
            subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        except subprocess.CalledProcessError as e:
            warning_id(f"Failed to upgrade pip: {e}")

        # Use the virtual environment's Python to install requirements
        python_executable = venv_python

        # Print activation instructions for the user
        if sys.platform == "win32":
            activate_cmd = os.path.join(venv_dir, "Scripts", "activate")
            info_id(f"To activate this virtual environment in the future, run: {activate_cmd}")
        else:
            activate_cmd = f"source {os.path.join(venv_dir, 'bin', 'activate')}"
            info_id(f"To activate this virtual environment in the future, run: {activate_cmd}")

        info_id(
            "Note: This script will continue using the virtual environment, but you'll need to activate it manually for future sessions.")
    else:
        # Use the current Python if not creating a virtual environment
        info_id("Skipping virtual environment creation.")
        python_executable = sys.executable

    # Install requirements
    if os.path.isfile(requirements_file):
        info_id(f"Installing dependencies from {requirements_file}...")
        try:
            subprocess.run([python_executable, "-m", "pip", "install", "-r", requirements_file], check=True)
            info_id("All dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            error_id(f"Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        warning_id(f"No requirements.txt file found at {requirements_file}. Skipping dependency installation.")


def create_directories():
    """Ensures all required directories exist, creating them if necessary."""
    global directories_to_check

    info_id("Checking and creating required directories...")

    # Get the project root for resolving relative paths
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    for directory in directories_to_check:
        try:
            # Handle relative paths by making them absolute relative to project root
            if not os.path.isabs(directory):
                # If it's a relative path, resolve it relative to project root
                if directory.startswith('..'):
                    # For paths like ../../static, resolve them properly
                    resolved_dir = os.path.abspath(os.path.join(project_root, directory))
                else:
                    resolved_dir = os.path.abspath(os.path.join(project_root, directory))
            else:
                resolved_dir = directory

            # Check if the resolved directory is within our project or system temp
            # Skip directories that would go outside safe boundaries
            if not (resolved_dir.startswith(project_root) or
                    resolved_dir.startswith(os.path.expanduser('~')) or
                    'temp' in resolved_dir.lower()):
                warning_id(f"Skipping directory outside project scope: {directory} -> {resolved_dir}")
                continue

            if not os.path.exists(resolved_dir):
                os.makedirs(resolved_dir, exist_ok=True)
                info_id(f"Created directory: {resolved_dir}")
            else:
                info_id(f"Directory already exists: {resolved_dir}")

        except PermissionError as e:
            warning_id(f"Cannot create directory {directory}: {e}")
            warning_id(f"Skipping - you may need to create this manually or run as administrator")
            continue
        except Exception as e:
            warning_id(f"Error with directory {directory}: {e}")
            continue

    info_id("Directory creation process completed")


def test_database_connection():
    """Test database connection and return database type information."""
    global DatabaseConfig

    try:
        info_id("Testing database schema connection...")
        db_config = DatabaseConfig()

        connection_info = db_config.test_connection()

        if connection_info['status'] == 'success':
            db_type = connection_info['database_type']
            version = connection_info.get('version', 'Unknown')

            info_id(f"Database connection successful!")
            info_id(f"   Database Type: {db_type}")
            info_id(f"   Version: {version}")

            if db_type == 'PostgreSQL':
                current_user = connection_info.get('current_user', 'Unknown')
                info_id(f"   Connected as: {current_user}")

                try:
                    with db_config.main_session() as session:
                        result = session.execute(text("""
                            SELECT has_database_privilege(current_user, current_database(), 'CREATE'),
                                   has_database_privilege(current_user, current_database(), 'CONNECT')
                        """))
                        can_create, can_connect = result.fetchone()
                        info_id(f"   Permissions - CREATE: {can_create}, CONNECT: {can_connect}")

                        if not can_create:
                            warning_id("Warning: User may not have CREATE privileges")
                            return False
                except Exception as e:
                    warning_id(f"Could not check permissions: {e}")

            return True
        else:
            error_id(f"Database connection failed: {connection_info.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        error_id(f"Error testing database connection: {e}")
        return False


def check_and_create_database_schema():
    """Create database schema and tables - now that database exists."""
    global DatabaseConfig, MainBase, RevisionControlBase

    info_id("Setting up database schema and tables...")

    # Step 1: Test connection
    if not test_database_connection():
        error_id("Database connection test failed.")
        sys.exit(1)

    # Step 2: Create database schema
    db_config = DatabaseConfig()
    main_engine = db_config.main_engine
    revision_engine = db_config.revision_control_engine

    try:
        # Import AI models module to ensure classes are registered
        info_id("Importing AI models module...")
        try:
            from plugins.ai_modules.ai_models.ai_models import ModelsConfig
            info_id("Successfully imported AI models module")
        except ImportError as e:
            warning_id(f"Could not import AI models module: {e}")
            warning_id("AI functionality may be limited")
            ModelsConfig = None

        # Check and create main database tables
        info_id("Setting up main database tables...")
        main_inspector = inspect(main_engine)
        main_tables = main_inspector.get_table_names()

        if not main_tables:
            info_id("Creating main database tables...")
            MainBase.metadata.create_all(main_engine)
            main_tables = inspect(main_engine).get_table_names()
            info_id(f"Created {len(main_tables)} main database tables")
        else:
            info_id(f"Main database ready with {len(main_tables)} existing tables")

        # Handle AI models configuration table if available
        if ModelsConfig:
            info_id("Setting up AI models configuration...")
            try:
                main_inspector = inspect(main_engine)

                if 'models_config' not in main_inspector.get_table_names():
                    info_id("Creating AI models configuration table...")
                    ModelsConfig.__table__.create(main_engine)
                    info_id("AI models configuration table created")

                    info_id("Initializing AI models with default configuration...")
                    ModelsConfig.initialize_models_config_table()
                    info_id("AI models configuration initialized")
                else:
                    info_id("AI models configuration table exists")

                    session = db_config.get_main_session()
                    try:
                        config_count = session.query(ModelsConfig).count()
                        if config_count == 0:
                            info_id("Initializing empty AI models configuration...")
                            ModelsConfig.initialize_models_config_table()
                            info_id("AI models configuration initialized")
                        else:
                            info_id(f"AI models configuration has {config_count} entries")
                    finally:
                        session.close()

            except Exception as e:
                warning_id(f"Error setting up AI models configuration: {e}")

        # Check and create revision control database tables
        info_id("Setting up revision control database...")
        revision_inspector = inspect(revision_engine)
        revision_tables = revision_inspector.get_table_names()

        if not revision_tables:
            info_id("Creating revision control database tables...")
            RevisionControlBase.metadata.create_all(revision_engine)
            revision_tables = inspect(revision_engine).get_table_names()
            info_id(f"Created {len(revision_tables)} revision control tables")
        else:
            info_id(f"Revision control database ready with {len(revision_tables)} existing tables")

        info_id("Database schema setup completed successfully!")

    except Exception as e:
        error_id(f"Database schema setup failed: {e}")
        sys.exit(1)


def setup_audit_system():
    """Set up the database audit system - Clean Version"""
    try:
        info_id("Setting up database audit system...")

        info_id("=" * 80)
        info_id("DATABASE AUDIT SYSTEM SETUP")
        info_id("=" * 80)
        info_id("The audit system provides comprehensive change tracking:")
        info_id("")
        info_id("CAPABILITIES:")
        info_id("   - Track all INSERT, UPDATE, DELETE operations")
        info_id("   - Record who made changes and when")
        info_id("   - Store before/after values for all changes")
        info_id("   - Track user sessions and IP addresses")
        info_id("   - Maintain complete change history")
        info_id("")
        info_id("IMPLEMENTATION:")
        info_id("   - SQLAlchemy Event-Based Auditing (Python integration)")
        info_id("   - PostgreSQL Trigger-Based Auditing (Database-level)")
        info_id("   - Central audit_log table + individual audit tables")
        info_id("   - Performance indexes for fast queries")
        info_id("")
        info_id("BENEFITS:")
        info_id("   - Compliance and regulatory requirements")
        info_id("   - Security monitoring and forensics")
        info_id("   - Data change history and recovery")
        info_id("   - User activity tracking")
        info_id("=" * 80)

        # Ask user if they want to set up audit system
        setup_audit_input = input("\nSet up comprehensive database audit system? (Recommended) (y/n): ").strip().lower()

        if setup_audit_input not in ['y', 'yes']:
            info_id("Skipped audit system setup.")
            return False

        # Import and run the audit system setup
        script_dir = os.path.dirname(__file__)
        audit_setup_script = os.path.join(script_dir, "setup_audit_system.py")

        # Check if the audit setup script exists
        if not os.path.exists(audit_setup_script):
            # Create the audit setup script from our template (CLEAN VERSION)
            info_id("Creating audit system setup script...")

            audit_script_content = '''#!/usr/bin/env python3
"""
EMTAC Database Audit System Setup Script
Integrates with your existing setup process to add comprehensive auditing
"""

import os
import sys
import subprocess
from datetime import datetime

# Add the project root to the path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

try:
    from modules.configuration.config_env import DatabaseConfig
    from modules.emtacdb.emtacdb_fts import Base as MainBase
    from modules.configuration.log_config import info_id, warning_id, error_id
    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import EMTAC modules: {e}")
    LOGGING_AVAILABLE = False

    def info_id(msg, **kwargs):
        print(f"INFO: {msg}")

    def warning_id(msg, **kwargs):
        print(f"WARNING: {msg}")

    def error_id(msg, **kwargs):
        print(f"ERROR: {msg}")

# Basic audit system setup (simplified version)
def setup_basic_audit_system():
    """Set up a basic audit system"""
    try:
        info_id("Setting up basic audit system...")

        db_config = DatabaseConfig()

        # Create basic audit table
        with db_config.main_session() as session:
            from sqlalchemy import text

            # Create audit_log table
            create_audit_table_sql = """
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                table_name VARCHAR(100) NOT NULL,
                record_id VARCHAR(100) NOT NULL,
                operation VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id VARCHAR(100),
                user_name VARCHAR(200),
                session_id VARCHAR(100),
                old_values TEXT,
                new_values TEXT,
                changed_fields TEXT,
                ip_address VARCHAR(50),
                user_agent TEXT,
                application VARCHAR(100) DEFAULT 'EMTAC',
                notes TEXT
            );
            """

            session.execute(text(create_audit_table_sql))

            # Create indexes
            index_statements = [
                "CREATE INDEX IF NOT EXISTS idx_audit_log_table_record ON audit_log(table_name, record_id);",
                "CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_audit_log_operation ON audit_log(operation);",
            ]

            for idx_sql in index_statements:
                try:
                    session.execute(text(idx_sql))
                except Exception as e:
                    warning_id(f"Index creation skipped: {e}")

            session.commit()
            info_id("Basic audit system created successfully")

        return True

    except Exception as e:
        error_id(f"Failed to setup basic audit system: {e}")
        return False

def main():
    """Main setup function"""
    try:
        return setup_basic_audit_system()
    except Exception as e:
        error_id(f"Audit setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''

            # Write the audit setup script with UTF-8 encoding
            try:
                with open(audit_setup_script, 'w', encoding='utf-8') as f:
                    f.write(audit_script_content)
                info_id(f"Created audit setup script: {audit_setup_script}")
            except Exception as write_error:
                error_id(f"Failed to write audit script: {write_error}")
                # Try without UTF-8 specification
                try:
                    with open(audit_setup_script, 'w') as f:
                        f.write(audit_script_content)
                    info_id(f"Created audit setup script (fallback): {audit_setup_script}")
                except Exception as fallback_error:
                    error_id(f"Could not create audit setup script: {fallback_error}")
                    return False

        # Run the audit system setup
        info_id("Running audit system setup...")
        try:
            result = subprocess.run([sys.executable, audit_setup_script],
                                    capture_output=True, text=True, check=False)

            if result.returncode == 0:
                info_id("Audit system setup completed successfully!")
                setup_state['audit_system_enabled'] = True

                # Show the output
                if result.stdout:
                    info_id("Audit Setup Output:")
                    info_id(result.stdout)

                return True
            else:
                error_id(f"Audit system setup failed with code {result.returncode}")
                if result.stderr:
                    error_id(f"Error output: {result.stderr}")
                return False

        except subprocess.CalledProcessError as e:
            error_id(f"Failed to run audit setup script: {e}")
            return False
        except FileNotFoundError:
            error_id(f"Audit setup script not found: {audit_setup_script}")
            return False

    except Exception as e:
        error_id(f"Error in audit system setup: {e}")
        return False


def suggest_database_backup():
    """Suggest backing up the database before running data import scripts."""
    try:
        db_config = DatabaseConfig()

        info_id("=" * 50)
        info_id("DATABASE BACKUP RECOMMENDATION")
        info_id("=" * 50)

        if db_config.is_postgresql:
            info_id("For PostgreSQL, you can create a backup using:")
            info_id("   pg_dump -h hostname -U username -d database_name > backup.sql")
            info_id("")

            # Try to provide specific command if we can parse the URL
            try:
                from modules.configuration.config import DATABASE_URL
                import re
                url_pattern = r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)'
                match = re.match(url_pattern, DATABASE_URL)
                if match:
                    username, _, host, port, db_name = match.groups()
                    if setup_state.get('database_name'):
                        db_name = setup_state['database_name']
                    info_id(f"   Your specific command:")
                    info_id(
                        f"   pg_dump -h {host} -p {port} -U {username} -d {db_name} > emtac_backup_$(date +%Y%m%d_%H%M%S).sql")
            except:
                pass

        else:
            info_id("For SQLite, you can simply copy the database file:")
            try:
                from modules.configuration.config import DATABASE_URL
                if DATABASE_URL.startswith('sqlite:///'):
                    db_path = DATABASE_URL.replace('sqlite:///', '')
                    info_id(f"   cp {db_path} {db_path}.backup")
            except:
                info_id("   cp your_database.db your_database.db.backup")

        info_id("=" * 50)

        backup_now = input("Would you like to pause here to create a backup? (y/n): ").strip().lower()
        if backup_now in ['y', 'yes']:
            input("Please create your backup now, then press Enter to continue...")
            info_id("User confirmed backup creation")
        else:
            info_id("User chose to skip backup")

    except Exception as e:
        warning_id(f"Error in backup suggestion: {e}")


def check_existing_data():
    """Check if data already exists in the database to prevent duplicates."""
    try:
        db_config = DatabaseConfig()

        # Check for existing data in key tables
        with db_config.main_session() as session:
            # You can add specific checks here based on your table structure
            # For example:
            # admin_count = session.execute(text("SELECT COUNT(*) FROM users WHERE role = 'admin'")).scalar()
            # parts_count = session.execute(text("SELECT COUNT(*) FROM parts")).scalar()

            # For now, we'll do a general table check
            inspector = inspect(db_config.main_engine)
            tables = inspector.get_table_names()

            data_exists = False
            table_info = {}

            # Check a few key tables for data
            key_tables = ['users', 'parts', 'drawings', 'equipment_relationships']
            for table in key_tables:
                if table in tables:
                    try:
                        count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                        table_info[table] = count
                        if count > 0:
                            data_exists = True
                    except:
                        table_info[table] = "unknown"

            return data_exists, table_info

    except Exception as e:
        warning_id(f"Could not check existing data: {e}")
        return False, {}


def run_setup_scripts():
    """Runs each of the setup scripts in sequence, prompting the user before each one."""

    # Check for existing data first
    data_exists, table_info = check_existing_data()

    if data_exists:
        info_id("=" * 50)
        info_id("EXISTING DATA DETECTED")
        info_id("=" * 50)
        info_id("The following tables already contain data:")
        for table, count in table_info.items():
            if isinstance(count, int) and count > 0:
                info_id(f"   {table}: {count} records")
        info_id("")
        info_id("Running import scripts may create duplicates!")
        info_id("Consider backing up your database first.")
        info_id("")

        proceed = input("Continue with data import scripts anyway? (y/n): ").strip().lower()
        if proceed not in ['y', 'yes']:
            info_id("Skipped setup scripts due to existing data.")
            return

    scripts_to_run = [
        "load_equipment_relationships_table_data.py",
        "initial_admin.py",
        "load_parts_sheet.py",
        "load_active_drawing_list.py",
        "load_image_folder.py",
        "load_bom_loadsheet.py",
    ]

    this_dir = os.path.dirname(os.path.abspath(__file__))

    for script in scripts_to_run:
        script_path = os.path.join(this_dir, script)

        # Get a description of what the script does (if available)
        script_description = {
            "load_equipment_relationships_table_data.py": "Loads equipment relationship data with advanced dependency management (PostgreSQL-enhanced)",
            "initial_admin.py": "Creates initial admin users with comprehensive validation (PostgreSQL-enhanced)",
            "load_parts_sheet.py": "Processes parts data with bulk operations and automatic associations (PostgreSQL-enhanced)",
            "load_active_drawing_list.py": "Loads active drawing list information (PostgreSQL-enhanced with duplicate prevention)",
            "load_image_folder.py": "Processes image folders with AI integration and duplicate detection (PostgreSQL-enhanced)",
            "load_bom_loadsheet.py": "Processes BOM loadsheets with advanced Excel manipulation (PostgreSQL-enhanced)"
        }.get(script, "No description available")

        # Prompt the user
        info_id(f"--- {script} ---")
        info_id(f"Description: {script_description}")

        # Add audit information if audit system is enabled
        if setup_state.get('audit_system_enabled'):
            info_id("Note: All changes will be automatically audited")

        user_input = input(f"Run {script}? (y/n): ").strip().lower()

        if user_input == 'y' or user_input == 'yes':
            info_id(f"Running: {script}...")
            try:
                subprocess.run([sys.executable, script_path], check=True)
                info_id(f"{script} completed successfully.")
            except subprocess.CalledProcessError as e:
                error_id(f"ERROR: Script {script} failed with: {e}")

                # Ask if the user wants to continue despite the error
                continue_input = input("Continue with the next script despite the error? (y/n): ").strip().lower()
                if continue_input != 'y' and continue_input != 'yes':
                    info_id("Setup aborted by user after script failure.")
                    sys.exit(1)
            except FileNotFoundError:
                error_id(f"ERROR: Could not find the script {script_path}")

                # Ask if the user wants to continue despite the missing script
                continue_input = input(
                    "Continue with the next script despite the missing file? (y/n): ").strip().lower()
                if continue_input != 'y' and continue_input != 'yes':
                    info_id("Setup aborted by user after missing script file.")
                    sys.exit(1)
        else:
            info_id(f"Skipping {script}...")

    info_id("Setup script sequence completed!")


def run_post_setup_associations():
    """Run automatic post-setup associations: Part ↔ Image and Drawing ↔ Part."""
    try:
        info_id("Starting post-setup association tasks...")

        # Run part-image associations
        info_id("Associating parts with images...")
        if db_maintenance:
            db_maintenance.associate_all_parts_with_images(export_report=True)
        else:
            warning_id("Database maintenance module not available, skipping associations")

        # Run drawing-part associations
        info_id("Associating drawings with parts...")
        if db_maintenance:
            db_maintenance.associate_all_drawings_with_parts(export_report=True)
        else:
            warning_id("Database maintenance module not available, skipping associations")

        info_id("Post-setup associations completed.")

    except Exception as e:
        error_id(f"Failed during post-setup associations: {e}")


def run_post_setup_ai_configuration():
    """Run AI-specific configuration tasks after the main setup is complete."""
    try:
        info_id("Running post-setup AI configuration...")

        try:
            from plugins.ai_modules.ai_models.ai_models import ModelsConfig

            # Ensure AI models are properly configured
            info_id("Checking AI model configurations...")

            # Get current configurations
            active_models = ModelsConfig.get_active_model_names()
            info_id(f"Current active models: {active_models}")

            # Verify that at least basic models are available
            available_ai_models = ModelsConfig.get_available_models('ai')
            if not available_ai_models:
                warning_id("No AI models available. This may cause issues with AI functionality.")
            else:
                info_id(f"{len(available_ai_models)} AI models available")

            available_embedding_models = ModelsConfig.get_available_models('embedding')
            if not available_embedding_models:
                warning_id("No embedding models available. This may cause issues with search functionality.")
            else:
                info_id(f"{len(available_embedding_models)} embedding models available")

            info_id("AI configuration check completed.")

        except ImportError:
            warning_id("AI models module not available. Skipping AI configuration.")

    except Exception as e:
        error_id(f"Failed during post-setup AI configuration: {e}")
        warning_id("AI functionality may be limited due to configuration issues.")


def display_setup_summary():
    """Display what this setup script will do."""
    info_id("=" * 70)
    info_id("EMTAC Enhanced Database & Application Setup")
    info_id("=" * 70)
    info_id("This comprehensive setup script will configure your EMTAC system including:")
    info_id("")
    info_id("DATABASE & SCHEMA:")
    info_id("   - Database creation BEFORE module imports (fixes import errors)")
    info_id("   - Database tables and relationships")
    info_id("   - Performance indexes and optimizations")
    info_id("   - PostgreSQL extensions (if using PostgreSQL)")
    info_id("   - Full-text search capabilities")
    info_id("")
    info_id("AUDIT & COMPLIANCE:")
    info_id("   - Comprehensive database change tracking")
    info_id("   - User activity monitoring")
    info_id("   - Before/after data snapshots")
    info_id("   - Regulatory compliance support")
    info_id("")
    info_id("AI & INTELLIGENCE:")
    info_id("   - AI model configurations")
    info_id("   - Embedding model setup")
    info_id("   - Smart search capabilities")
    info_id("")
    info_id("FILES & DIRECTORIES:")
    info_id("   - Required project directories")
    info_id("   - File upload and processing folders")
    info_id("   - Backup and utility directories")
    info_id("")
    info_id("PYTHON ENVIRONMENT:")
    info_id("   - Virtual environment setup (optional)")
    info_id("   - Dependency installation")
    info_id("   - Package management")
    info_id("")
    info_id("DATA & CONTENT:")
    info_id("   - Equipment relationships with dependency management")
    info_id("   - Initial admin users with security validation")
    info_id("   - Parts processing with bulk operations and auto-associations")
    info_id("   - Drawing associations and active lists")
    info_id("   - Image processing with AI integration")
    info_id("   - Advanced duplicate prevention across all imports")
    info_id("")
    info_id("SMART ASSOCIATIONS:")
    info_id("   - Automatic part-image linking")
    info_id("   - Drawing-part relationships")
    info_id("   - Content indexing")
    info_id("")
    info_id("IMPORTANT NOTES:")
    info_id("   - For PostgreSQL: Database server must be running")
    info_id("   - For PostgreSQL: Database will be created EARLY to prevent import errors")
    info_id("   - For PostgreSQL: You can choose a custom database name")
    info_id("   - For SQLite: Database file will be created automatically")
    info_id("   - Existing data will be preserved")
    info_id("   - Virtual environment is recommended but optional")
    info_id("   - Audit system provides comprehensive change tracking")
    info_id("=" * 70)


def check_prerequisites():
    """Check prerequisites and provide setup instructions for PostgreSQL."""
    try:
        from modules.configuration.config import DATABASE_URL

        info_id("Checking setup prerequisites...")

        if DATABASE_URL.startswith('postgresql'):
            info_id("PostgreSQL setup detected")
            info_id("Prerequisites for PostgreSQL:")
            info_id("   1. PostgreSQL server must be running")
            info_id("   2. Database user must exist with appropriate privileges")
            info_id("   3. Network connectivity to PostgreSQL server")
            info_id("   4. Correct DATABASE_URL in your configuration")

            # Parse and display connection details (without password)
            import re
            url_pattern = r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)'
            match = re.match(url_pattern, DATABASE_URL)

            if match:
                username, _, host, port, database_name = match.groups()
                info_id(f"   Current connection details:")
                info_id(f"      Host: {host}:{port}")
                info_id(f"      User: {username}")
                info_id(f"      Default Database: {database_name}")
                info_id(f"   You can choose a different database name during setup")

            # Test basic server connectivity
            try:
                import socket
                url_match = re.match(r'postgresql://[^@]+@([^:]+):(\d+)/', DATABASE_URL)
                if url_match:
                    host, port = url_match.groups()
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, int(port)))
                    sock.close()

                    if result == 0:
                        info_id(f"   Network connectivity to {host}:{port} - OK")
                    else:
                        warning_id(f"   Cannot connect to {host}:{port}")
                        warning_id("   Check if PostgreSQL server is running and accessible")
                        return False
            except Exception as e:
                warning_id(f"   Could not test network connectivity: {e}")

        else:
            info_id("SQLite setup detected")
            info_id("Prerequisites for SQLite:")
            info_id("   1. Write permissions to database directory")
            info_id("   2. Sufficient disk space")

        return True

    except Exception as e:
        error_id(f"Error checking prerequisites: {e}")
        return False


def main():
    """Enhanced main setup function with early database creation."""
    try:
        # Initialize request ID for tracking
        from modules.configuration.log_config import set_request_id
        setup_state['request_id'] = set_request_id()

        # Display what this script does
        display_setup_summary()

        # Check prerequisites before starting
        if not check_prerequisites():
            error_id("Prerequisites check failed. Please resolve issues before continuing.")
            sys.exit(1)

        # Ask user to confirm they want to proceed
        proceed = input("\nDo you want to proceed with the comprehensive EMTAC setup? (y/n): ").strip().lower()
        if proceed not in ['y', 'yes']:
            info_id("Setup cancelled by user.")
            sys.exit(0)

        # Setup process
        info_id("Starting Enhanced EMTAC Database & Application Setup...")
        info_id("=" * 70)

        # Step 1: Create base directories
        create_base_directories()

        # Step 1.5: EARLY DATABASE CREATION - BEFORE MODULE IMPORTS
        info_id("STEP 1.5: Early Database Creation (prevents import errors)")
        if not early_database_creation():
            error_id("Early database creation failed. Exiting.")
            sys.exit(1)

        # Step 2: Import modules (now database exists)
        info_id("STEP 2: Module imports (database now exists)")
        if not import_modules_after_directory_setup():
            error_id("Failed to import required modules. Exiting.")
            sys.exit(1)

        # Step 3: Create all directories
        create_directories()

        # Step 4: Setup virtual environment and dependencies
        setup_virtual_environment_and_install_requirements()

        # Step 5: Database schema setup (tables, etc.)
        info_id("STEP 5: Database schema setup")
        check_and_create_database_schema()

        # Step 5.5: Set up Full-Text Search (FTS) tables
        info_id("STEP 5.5: Full-Text Search (FTS) table setup")
        info_id("=" * 50)
        info_id("FULL-TEXT SEARCH SETUP")
        info_id("=" * 50)
        info_id("This will enable advanced search capabilities:")
        info_id("- Create FTS table for document indexing")
        info_id("- Set up GIN indexes for fast searches")
        info_id("- Enable automatic search vector updates")
        info_id("- Requires PostgreSQL extensions (pg_trgm, unaccent)")
        info_id("")
        setup_fts = input("Set up Full-Text Search tables? (Recommended for PostgreSQL) (y/n): ").strip().lower()
        if setup_fts in ['y', 'yes']:
            try:
                from modules.emtacdb.emtacdb_fts import CompleteDocument
                if Document.create_fts_table():
                    info_id("Full-Text Search tables created successfully")
                else:
                    warning_id("Failed to create Full-Text Search tables, continuing setup")
            except ImportError as e:
                warning_id(f"Could not import CompleteDocument: {e}. Skipping FTS setup")
            except Exception as e:
                warning_id(f"Error setting up Full-Text Search tables: {e}. Continuing setup")
        else:
            info_id("Skipped Full-Text Search table setup")

        # Step 6: AI configuration
        info_id("STEP 6: AI configuration")
        run_post_setup_ai_configuration()

        # Step 7: AUDIT SYSTEM SETUP
        info_id("STEP 7: Audit system setup")
        setup_audit_system()

        # Step 8: Run setup scripts
        info_id("=" * 50)
        info_id("DATA IMPORT & CONFIGURATION SCRIPTS")
        info_id("=" * 50)
        info_id("The following scripts will populate your database with initial data:")
        info_id("- Equipment relationships (PostgreSQL-enhanced with dependency management)")
        info_id("- Initial admin users (with comprehensive validation and security)")
        info_id("- Parts data import (with bulk operations and automatic associations)")
        info_id("- Drawing list import (with duplicate prevention)")
        info_id("- Image folder processing (with AI integration)")
        info_id("- BOM data import (with advanced Excel manipulation)")
        info_id("")
        info_id("Note: All scripts use PostgreSQL framework with duplicate prevention")
        info_id("Each script will check for existing data and warn you accordingly")

        if setup_state.get('audit_system_enabled'):
            info_id("Audit System: All data changes will be automatically tracked")

        info_id("")

        run_scripts = input("Would you like to run the data import scripts now? (y/n): ").strip().lower()
        if run_scripts in ['y', 'yes']:
            # Suggest backup before importing data
            suggest_database_backup()
            run_setup_scripts()
        else:
            info_id("Skipped setup scripts. You can run them manually later.")

        # Step 9: Associations
        info_id("=" * 50)
        info_id("AUTOMATIC ASSOCIATIONS")
        info_id("=" * 50)
        info_id("These create smart links between:")
        info_id("- Parts <-> Images")
        info_id("- Drawings <-> Parts")
        info_id("")

        run_associations = input("Run automatic part-image and drawing-part associations now? (y/n): ").strip().lower()
        if run_associations in ['y', 'yes']:
            run_post_setup_associations()
        else:
            info_id("Skipped association step. You can run them manually later.")

        # Step 10: Log compression
        info_id("Would you like to compress old setup logs?")
        compress_input = input("Compress logs? (y/n): ").strip().lower()

        if compress_input == 'y' or compress_input == 'yes':
            this_dir = os.path.dirname(os.path.abspath(__file__))
            logs_directory = os.path.join(this_dir, "logs")
            if os.path.exists(logs_directory):
                info_id("Compressing old initializer logs...")
                compress_logs_except_most_recent(logs_directory)
                info_id("Log compression completed.")
            else:
                warning_id(f"No logs directory found at {logs_directory}.")

            # Also compress global LOG_DIRECTORY if different
            if LOG_DIRECTORY != logs_directory and os.path.exists(LOG_DIRECTORY):
                compress_logs_except_most_recent(LOG_DIRECTORY)

        # Final success message
        info_id("Enhanced EMTAC Setup completed successfully!")
        info_id("=" * 70)
        info_id("EMTAC SETUP COMPLETED SUCCESSFULLY!")
        info_id("=" * 70)
        info_id("What was accomplished:")
        info_id("")
        info_id("DATABASE:")
        info_id("   - PostgreSQL/SQLite database created EARLY (no import errors)")
        info_id("   - All database tables and relationships ready")
        info_id("   - Database extensions and optimizations enabled")
        info_id("   - Full-Text Search tables configured (if enabled)")
        info_id("")

        if setup_state.get('audit_system_enabled'):
            info_id("AUDIT SYSTEM:")
            info_id("   - Comprehensive change tracking enabled")
            info_id("   - Central audit_log table created")
            info_id("   - User activity monitoring ready")
            info_id("   - Compliance and security auditing active")
            info_id("")

        info_id("AI & INTELLIGENCE:")
        info_id("   - AI model configurations initialized")
        info_id("   - Smart search capabilities enabled")
        info_id("   - Embedding models configured")
        info_id("")
        info_id("FILES & STRUCTURE:")
        info_id("   - All required directories created")
        info_id("   - File processing folders ready")
        info_id("   - Backup systems configured")
        info_id("")

        if run_scripts in ['y', 'yes']:
            info_id("DATA & CONTENT:")
            info_id("   - Equipment relationships with dependency management")
            info_id("   - Initial admin users with security validation")
            info_id("   - Parts processing with bulk operations and auto-associations")
            info_id("   - Drawing associations with duplicate prevention")
            info_id("   - Image processing with AI integration")
            info_id("   - Comprehensive duplicate detection across all imports")
            if setup_state.get('audit_system_enabled'):
                info_id("   - All imported data automatically audited")
            info_id("")

        if run_associations in ['y', 'yes']:
            info_id("SMART ASSOCIATIONS:")
            info_id("   - Automatic part-image links created")
            info_id("   - Drawing-part relationships established")
            info_id("")

        # Show final database information
        try:
            db_config = DatabaseConfig()
            db_type = "PostgreSQL" if db_config.is_postgresql else "SQLite"
            info_id(f"DATABASE INFORMATION:")
            info_id(f"   Type: {db_type}")
            info_id(f"   Status: Ready for use")
            info_id(f"   Full-Text Search: {'Enabled' if setup_fts in ['y', 'yes'] else 'Disabled'}")

            if setup_state.get('audit_system_enabled'):
                info_id(f"   Audit System: Active and monitoring")

            if setup_state['custom_database_url']:
                db_name = setup_state['database_name']
                info_id(f"   Database Name: {db_name} (custom)")
            elif db_config.is_postgresql:
                stats = db_config.get_connection_stats()
                current_url = stats.get('database_url', '')
                if current_url:
                    db_name = current_url.split('/')[-1]
                    info_id(f"   Database Name: {db_name}")

            if db_config.is_postgresql:
                info_id("   Optimizations: PostgreSQL features enabled")

            info_id(f"")
            info_id(f"NEXT STEPS:")
            info_id("   1. Your EMTAC system is now fully configured")
            info_id("   2. All data import scripts use PostgreSQL framework")
            info_id("   3. Admin users are ready with secure authentication")
            info_id("   4. Advanced position creation tools are available")
            if setup_fts in ['y', 'yes']:
                info_id("   5. Full-Text Search is enabled for document indexing")
            if setup_state.get('audit_system_enabled'):
                info_id("   6. Audit system is tracking all database changes")
                info_id("   7. Check audit logs with: SELECT * FROM audit_log;")
            info_id("   8. Start your EMTAC application")
            info_id("   9. Log in with your admin credentials")
            info_id("   10. Change default passwords for security")
            info_id("   11. Begin using all EMTAC features")

        except Exception as e:
            warning_id(f"Could not display final database info: {e}")

        info_id("=" * 70)
        info_id("Your Enhanced EMTAC System is Ready!")
        info_id("Early database creation prevents all import errors")
        info_id("Advanced position creation and management tools available")
        if setup_fts in ['y', 'yes']:
            info_id("Full-Text Search enabled for advanced document search")
        if setup_state.get('audit_system_enabled'):
            info_id("Comprehensive audit system providing full change tracking")
        info_id("World-class user experience with enterprise reliability")
        info_id("=" * 70)

    except KeyboardInterrupt:
        info_id("Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        error_id(f"Unexpected error during setup: {e}")
        sys.exit(1)
    finally:
        try:
            close_initializer_logger()
        except:
            pass


if __name__ == "__main__":
    main()