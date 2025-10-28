
# --- Load environment variables from .env automatically ---
from dotenv import load_dotenv
import os

# Resolve path to the project root .env (two levels up from this file)
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')

# Load it if it exists
if os.path.isfile(env_path):
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")
else:
    print(f"No .env file found at {env_path}")
import os
import sys
import subprocess
import shutil
from sqlalchemy import inspect
from modules.configuration.log_config import debug_id, info_id, warning_id, error_id, get_request_id, logger
from modules.database_manager.maintenance import db_maintenance

# Initialize logger first
from modules.initial_setup.initializer_logger import (
    initializer_logger, compress_logs_except_most_recent, close_initializer_logger, LOG_DIRECTORY
)

# Global variables that will be populated after directory creation
DatabaseConfig = None
MainBase = None
RevisionControlBase = None
directories_to_check = []

print("Database URL:", os.getenv("DATABASE_URL"))

def create_base_directories():
    """
    Creates the essential base directories needed before importing modules
    """
    info_id("Creating essential base directories...")

    # Create the project root directory path
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Essential directories that must exist before importing modules
    essential_dirs = [
        os.path.join(project_root, "Database"),
        os.path.join(project_root, "logs")
    ]

    for directory in essential_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            info_id(f" Created essential directory: {directory}")
        else:
            info_id(f"️ Essential directory already exists: {directory}")


def import_modules_after_directory_setup():
    """
    Imports modules that require base directories to exist
    """
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
        from modules.emtacdb.emtac_revision_control_db import RevisionControlBase

        # List of directories to check and create
        directories_to_check = [
            TEMPLATE_FOLDER_PATH,
            DATABASE_DIR,
            UPLOAD_FOLDER,
            IMAGES_FOLDER,
            DATABASE_PATH_IMAGES_FOLDER,
            PDF_FOR_EXTRACTION_FOLDER,
            IMAGES_EXTRACTED,
            TEMPORARY_FILES,
            PPT2PDF_PPT_FILES_PROCESS,
            PPT2PDF_PDF_FILES_PROCESS,
            DATABASE_DOC,
            TEMPORARY_UPLOAD_FILES,
            DB_LOADSHEET,
            DB_LOADSHEETS_BACKUP,
            DB_LOADSHEET_BOMS,
            BACKUP_DIR,
            Utility_tools,
            UTILITIES
        ]

        info_id(" Successfully imported all required modules")
        return True
    except Exception as e:
        error_id(f" Error importing modules: {e}", exc_info=True)
        return False


def setup_virtual_environment_and_install_requirements():
    """
    Creates or uses an existing virtual environment (.venv or venv)
    and installs requirements.txt dependencies.
    Offers the user control to skip creation or dependency installation.
    """

    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    requirements_file = os.path.join(project_root, "requirements.txt")

    # Accept both .venv and venv
    venv_dir_candidates = [
        os.path.join(project_root, ".venv"),
        os.path.join(project_root, "venv")
    ]
    venv_dir = next((v for v in venv_dir_candidates if os.path.exists(v)), venv_dir_candidates[0])

    info_id("Checking for existing virtual environment (.venv or venv)...")

    # 🧩 Detect if already inside an active virtual environment
    if hasattr(sys, "real_prefix") or sys.base_prefix != sys.prefix:
        info_id("Detected active virtual environment — skipping venv creation.")
        # Optional: still install dependencies if desired
        install_reqs = input("Install dependencies in this environment? (y/n): ").strip().lower()
        if install_reqs in ["y", "yes"]:
            install_requirements(sys.executable, requirements_file)
        else:
            info_id("Skipping dependency installation entirely.")
        return  # Exit early since we don't want to rebuild venv

    # 🧱 Normal behavior: user is not in a venv yet
    if os.path.exists(venv_dir):
        info_id(f"Existing virtual environment found at: {venv_dir}")
        use_existing = input(f"Use the existing virtual environment at {venv_dir}? (y/n): ").strip().lower()

        if use_existing not in ["y", "yes"]:
            overwrite = input("Overwrite and create a new virtual environment? (y/n): ").strip().lower()
            if overwrite in ["y", "yes"]:
                info_id(f"Removing existing virtual environment at {venv_dir}...")
                shutil.rmtree(venv_dir)
            else:
                info_id("Skipping virtual environment creation and continuing with system Python.")
                python_executable = sys.executable
                install_reqs = input("Install dependencies using system Python? (y/n): ").strip().lower()
                if install_reqs in ["y", "yes"]:
                    install_requirements(python_executable, requirements_file)
                else:
                    info_id("Skipping dependency installation entirely.")
                return
    else:
        # Ask user if they want to create one
        create_venv = input("Would you like to create a virtual environment? (Recommended) (y/n): ").strip().lower()
        if create_venv not in ["y", "yes"]:
            info_id("Skipping virtual environment creation.")
            python_executable = sys.executable
            install_reqs = input("Install dependencies using system Python? (y/n): ").strip().lower()
            if install_reqs in ["y", "yes"]:
                install_requirements(python_executable, requirements_file)
            else:
                info_id("Skipping dependency installation entirely.")
            return

    # 🏗️ Create the virtual environment
    info_id(f"Creating virtual environment at {venv_dir}...")
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        info_id("Virtual environment created successfully.")
    except subprocess.CalledProcessError as e:
        error_id(f"Failed to create virtual environment: {e}")
        sys.exit(1)

    # Detect platform and set activation commands
    if sys.platform == "win32":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        activate_cmd = os.path.join(venv_dir, "Scripts", "activate")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")
        activate_cmd = f"source {os.path.join(venv_dir, 'bin', 'activate')}"

    # Upgrade pip
    try:
        info_id("Upgrading pip in virtual environment...")
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    except subprocess.CalledProcessError as e:
        warning_id(f"Failed to upgrade pip: {e}")

    # Print activation instructions
    print(f"\nTo activate this virtual environment in the future, run:\n   {activate_cmd}\n")
    print("Note: This script will continue using the virtual environment now, "
          "but you'll need to activate it manually for future sessions.\n")

    # Ask whether to install dependencies
    install_reqs = input("Install dependencies from requirements.txt now? (y/n): ").strip().lower()
    if install_reqs in ["y", "yes"]:
        install_requirements(venv_python, requirements_file)
    else:
        info_id("Skipping dependency installation.")


def check_internet_connectivity(host="pypi.org", port=443, timeout=3):
    """
    Checks if there is an active internet connection by attempting to reach a known host.
    Returns True if online, False if offline.
    """
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False


def install_requirements(python_executable, requirements_file):
    """Helper function to install dependencies safely, with offline detection."""
    if not os.path.isfile(requirements_file):
        warning_id(f"No requirements.txt file found at {requirements_file}. Skipping dependency installation.")
        return

    info_id(f"Preparing to install dependencies from {requirements_file}...")

    # Check for internet connection before running pip
    if not check_internet_connectivity():
        warning_id("No internet connection detected. Skipping pip installation to avoid errors.")
        print("Offline mode detected. Dependencies will not be installed.")
        return

    try:
        info_id(f"Installing dependencies from {requirements_file}...")
        subprocess.run([python_executable, "-m", "pip", "install", "-r", requirements_file], check=True)
        info_id("All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        error_id(f"Failed to install dependencies: {e}")
        sys.exit(1)


def create_directories():
    """
    Ensures all required directories exist, creating them if necessary.
    """
    global directories_to_check

    info_id("Checking and creating required directories...")

    for directory in directories_to_check:
        if not os.path.exists(directory):
            os.makedirs(directory)
            info_id(f" Created directory: {directory}")
        else:
            info_id(f"️ Directory already exists: {directory}")


def check_and_create_database():
    """
    Ensures that the main and revision control databases exist and have all required tables.
    """
    global DatabaseConfig, MainBase, RevisionControlBase

    info_id("Checking if databases and tables exist...")

    db_config = DatabaseConfig()
    main_engine = db_config.main_engine
    revision_engine = db_config.revision_control_engine

    try:
        # Ensure ORM models are registered BEFORE schema creation
        info_id("Importing ORM models from modules.emtacdb.emtacdb_fts...")
        try:
            import modules.emtacdb.emtacdb_fts  # Import all EMTAC model definitions
            info_id("All ORM models loaded successfully.")
        except Exception as e:
            warning_id(f"Failed to import EMTAC ORM models: {e}")
            raise

        # Import AI models module to ensure classes are registered
        info_id("Importing AI models module...")
        try:
            from plugins.ai_modules.ai_models.ai_models import ModelsConfig
            info_id("Successfully imported AI models module.")
        except ImportError as e:
            warning_id(f"Could not import AI models module: {e}")
            warning_id("AI functionality may be limited.")
            ModelsConfig = None

        # Check if main database tables exist
        main_inspector = inspect(main_engine)
        main_tables = main_inspector.get_table_names()

        if not main_tables:
            warning_id("No tables found in the main database. Creating all tables now...")
            MainBase.metadata.create_all(main_engine)
            info_id("Main database tables created successfully.")
        else:
            info_id(f"Main database is ready with tables: {main_tables}")

        # Handle AI models configuration table if available
        if ModelsConfig:
            info_id("Setting up AI models configuration table...")
            try:
                if 'models_config' not in main_inspector.get_table_names():
                    warning_id("AI models configuration table not found. Creating...")
                    ModelsConfig.__table__.create(main_engine)
                    info_id("AI models configuration table created successfully.")
                    info_id("Initializing AI models configuration with default values...")
                    ModelsConfig.initialize_models_config_table()
                    info_id("AI models configuration initialized.")
                else:
                    info_id("AI models configuration table already exists.")
                    session = db_config.get_main_session()
                    try:
                        config_count = session.query(ModelsConfig).count()
                        if config_count == 0:
                            info_id("AI models configuration table is empty. Initializing...")
                            ModelsConfig.initialize_models_config_table()
                            info_id("AI models configuration initialized.")
                        else:
                            info_id(f"AI models configuration has {config_count} entries.")
                    finally:
                        session.close()
            except Exception as e:
                warning_id(f"Error setting up AI models configuration: {e}")
                warning_id("AI functionality may be limited.")

        # Check if revision control database tables exist
        revision_inspector = inspect(revision_engine)
        revision_tables = revision_inspector.get_table_names()
        if not revision_tables:
            warning_id("No tables found in the revision control database. Creating tables...")
            RevisionControlBase.metadata.create_all(revision_engine)
            info_id("Revision control database tables created successfully.")
        else:
            info_id(f"Revision control database is ready with tables: {revision_tables}")

        # Verify final database state
        info_id("Verifying final database state...")
        final_main_tables = inspect(main_engine).get_table_names()
        final_revision_tables = inspect(revision_engine).get_table_names()

        info_id(f"Final main database tables: {len(final_main_tables)} tables -> {final_main_tables}")
        info_id(f"Final revision control tables: {len(final_revision_tables)} tables -> {final_revision_tables}")

    except Exception as e:
        error_id(f"Database setup failed: {e}", exc_info=True)
        sys.exit(1)

def run_setup_scripts():
    """
    Runs each of the setup scripts in sequence, prompting the user before each one.
    """
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
            "load_equipment_relationships_table_data.py": "Loads equipment relationship data into the database",
            "initial_admin.py": "Creates the initial admin user",
            "load_parts_sheet.py": "Imports parts data from spreadsheets",
            "load_active_drawing_list.py": "Loads active drawing list information",
            "load_image_folder.py": "Imports images from the image folder",
            "load_bom_loadsheet.py": "Imports bill of materials data"
        }.get(script, "No description available")

        # Prompt the user
        print(f"\n--- {script} ---")
        print(f"Description: {script_description}")

        user_input = input(f"Run {script}? (y/n): ").strip().lower()

        if user_input == 'y' or user_input == 'yes':
            print(f"\nRunning: {script}...\n")
            try:
                subprocess.run([sys.executable, script_path], check=True)
                print(f" {script} completed successfully.")
                info_id(f" {script} completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f" ERROR: Script {script} failed with: {e}")
                error_id(f" Script {script} failed with: {e}")

                # Ask if the user wants to continue despite the error
                continue_input = input("Continue with the next script despite the error? (y/n): ").strip().lower()
                if continue_input != 'y' and continue_input != 'yes':
                    info_id("Setup aborted by user after script failure.")
                    sys.exit(1)
            except FileNotFoundError:
                print(f" ERROR: Could not find the script {script_path}")
                error_id(f" Could not find the script {script_path}")

                # Ask if the user wants to continue despite the missing script
                continue_input = input(
                    "Continue with the next script despite the missing file? (y/n): ").strip().lower()
                if continue_input != 'y' and continue_input != 'yes':
                    info_id("Setup aborted by user after missing script file.")
                    sys.exit(1)
        else:
            print(f"Skipping {script}...")
            info_id(f"User chose to skip {script}")

    print("\n Setup script sequence completed!")
    info_id(" Setup script sequence completed!")

    # Ask about log compression
    print("\nWould you like to compress old initializer logs?")
    compress_input = input("Compress logs? (y/n): ").strip().lower()

    if compress_input == 'y' or compress_input == 'yes':
        logs_directory = os.path.join(this_dir, "logs")
        if os.path.exists(logs_directory):
            print("🗜️ Compressing old initializer logs...")
            info_id("🗜️ Compressing old initializer logs...")
            compress_logs_except_most_recent(logs_directory)
            print("️ Log compression completed.")
            info_id("️ Log compression completed.")
        else:
            print(f"️ No logs directory found at {logs_directory}.")
            warning_id(f"️ No logs directory found at {logs_directory}.")


def run_post_setup_associations():
    """
    Run automatic post-setup associations: Part ↔ Image and Drawing ↔ Part.
    """
    try:
        info_id("🔁 Starting post-setup association tasks...")

        # Run part-image associations
        info_id("🔗 Associating parts with images...")
        db_maintenance.associate_all_parts_with_images(export_report=True)

        # Run drawing-part associations
        info_id("🔗 Associating drawings with parts...")
        db_maintenance.associate_all_drawings_with_parts(export_report=True)

        info_id(" Post-setup associations completed.")

    except Exception as e:
        error_id(f" Failed during post-setup associations: {e}", exc_info=True)


def run_post_setup_ai_configuration():
    """
    Run AI-specific configuration tasks after the main setup is complete.
    """
    try:
        info_id(" Running post-setup AI configuration...")

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
                warning_id("️ No AI models available. This may cause issues with AI functionality.")
            else:
                info_id(f" {len(available_ai_models)} AI models available")

            available_embedding_models = ModelsConfig.get_available_models('embedding')
            if not available_embedding_models:
                warning_id("️ No embedding models available. This may cause issues with search functionality.")
            else:
                info_id(f" {len(available_embedding_models)} embedding models available")

            info_id(" AI configuration check completed.")

        except ImportError:
            warning_id("️ AI models module not available. Skipping AI configuration.")

    except Exception as e:
        error_id(f" Failed during post-setup AI configuration: {e}", exc_info=True)
        warning_id("️ AI functionality may be limited due to configuration issues.")


def main():
    """
    Main setup function that orchestrates the entire setup process.
    """
    try:
        # First, create essential base directories (Database and logs)
        info_id("Starting EMTAC database setup...")
        create_base_directories()

        # Import modules that depend on directories existing
        if not import_modules_after_directory_setup():
            error_id(" Failed to import required modules. Exiting.")
            sys.exit(1)

        # Create remaining directories defined in config
        create_directories()

        # Create virtual environment and ensure dependencies are installed
        setup_virtual_environment_and_install_requirements()

        # Ensure database and tables are ready
        check_and_create_database()

        # Run AI-specific configuration
        run_post_setup_ai_configuration()

        # Run all setup scripts
        run_setup_scripts()

        # Ask about associations
        user_input = input("\nRun automatic part-image and drawing-part associations now? (y/n): ").strip().lower()
        if user_input in ['y', 'yes']:
            run_post_setup_associations()
        else:
            logger.info("Skipped association step.")

        # Compress logs
        compress_logs_except_most_recent(LOG_DIRECTORY)

        info_id(" All setup completed successfully!")
        print("\n Setup completed successfully!")
        print("Your EMTAC database is now ready to use.")

    except KeyboardInterrupt:
        print("\n\n Setup interrupted by user.")
        info_id(" Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        error_id(f" Unexpected error during setup: {e}", exc_info=True)
        print(f"\n Setup failed with error: {e}")
        sys.exit(1)
    finally:
        # Always close the logger
        try:
            close_initializer_logger()
        except:
            pass


if __name__ == "__main__":
    main()