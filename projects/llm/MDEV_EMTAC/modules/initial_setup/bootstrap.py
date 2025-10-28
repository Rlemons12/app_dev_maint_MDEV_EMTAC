import os
import sys
import shutil


# This is a minimal bootstrap script to create essential directories
# before any database modules are imported

def create_essential_dirs():
    """Create essential directories needed before any modules can be imported"""
    print("Creating essential directories...")

    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Create essential directories
    essential_dirs = [
        os.path.join(project_root, "Database"),
        os.path.join(project_root, "logs"),
        os.path.join(project_root, "Database", "db_backup"),
        os.path.join(project_root, "Database", "DB_DOC"),
        os.path.join(project_root, "Database", "DB_IMAGES"),
        os.path.join(project_root, "Database", "DB_LOADSHEETS"),
        os.path.join(project_root, "Database", "DB_LOADSHEETS_BACKUP"),
        os.path.join(project_root, "Database", "logs"),
        os.path.join(project_root, "Database", "PDF_FILES"),
        os.path.join(project_root, "Database", "PPT_FILES")
    ]

    for directory in essential_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

    # Create .gitkeep files to maintain directory structure in git
    for directory in essential_dirs:
        if directory != os.path.join(project_root, "Database") and directory != os.path.join(project_root, "logs"):
            gitkeep_file = os.path.join(directory, ".gitkeep")
            if not os.path.exists(gitkeep_file):
                with open(gitkeep_file, "w") as f:
                    pass  # Create empty file
                print(f"Created .gitkeep file in: {directory}")


if __name__ == "__main__":
    create_essential_dirs()

    # After creating directories, run the main setup script
    setup_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "modules", "initial_setup", "emtacdb_initial_setup.py")

    print(f"\nEssential directories created. Now running the main setup script: {setup_script}\n")

    # Use sys.executable to ensure we're using the same Python interpreter
    os.execv(sys.executable, [sys.executable, setup_script])