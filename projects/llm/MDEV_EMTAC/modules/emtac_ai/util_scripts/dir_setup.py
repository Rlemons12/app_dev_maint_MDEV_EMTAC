#!/usr/bin/env python3
"""
Directory Setup Script
Creates all directories defined in config.py if they don't exist.
Uses custom logging from log_config.py for consistent logging across the application.
"""

import os
import sys
from pathlib import Path

# Import the config file
try:
    import config
except ImportError:
    print("Error: config.py not found. Make sure this script is in the same directory as config.py")
    sys.exit(1)

# Import custom logging
try:
    from modules.configuration.log_config import (
        debug_id, info_id, warning_id, error_id, critical_id,
        set_request_id, with_request_id, log_timed_operation
    )

    LOGGING_AVAILABLE = True
except ImportError:
    print("Warning: log_config.py not found. Falling back to print statements.")
    LOGGING_AVAILABLE = False


    # Fallback functions that just print
    def debug_id(msg, req_id=None):
        print(f"DEBUG: {msg}")


    def info_id(msg, req_id=None):
        print(f"INFO: {msg}")


    def warning_id(msg, req_id=None):
        print(f"WARNING: {msg}")


    def error_id(msg, req_id=None):
        print(f"ERROR: {msg}")


    def critical_id(msg, req_id=None):
        print(f"CRITICAL: {msg}")


    def set_request_id(req_id=None):
        return "fallback"


    def with_request_id(func):
        return func


    class log_timed_operation:
        def __init__(self, name, req_id=None): pass

        def __enter__(self): return self

        def __exit__(self, *args): pass


def create_directory(path, description, request_id):
    """
    Create a directory if it doesn't exist.

    Args:
        path (str): The directory path to create
        description (str): Human-readable description for logging
        request_id (str): Request ID for logging correlation

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        debug_id(f"Attempting to create directory: {path}", request_id)

        with log_timed_operation(f"Create {description}", request_id):
            os.makedirs(path, exist_ok=True)

        if os.path.exists(path):
            info_id(f"✓ Successfully verified/created {description}: {path}", request_id)
            return True
        else:
            error_id(f"✗ Failed to create {description}: {path} (directory does not exist after creation attempt)",
                     request_id)
            return False

    except PermissionError as e:
        error_id(f"✗ Permission denied creating {description} ({path}): {str(e)}", request_id)
        return False
    except OSError as e:
        error_id(f"✗ OS error creating {description} ({path}): {str(e)}", request_id)
        return False
    except Exception as e:
        error_id(f"✗ Unexpected error creating {description} ({path}): {str(e)}", request_id)
        return False


@with_request_id
def main():
    """Main function to create all directories from config."""
    request_id = set_request_id()

    info_id("Starting directory setup script", request_id)
    info_id("Setting up directory structure from config.py...", request_id)

    with log_timed_operation("Complete directory setup", request_id):
        success_count = 0
        total_count = 0

        # Base directory
        info_id("Creating base directory...", request_id)
        total_count += 1
        if create_directory(config.BASE_DIR, "Base directory", request_id):
            success_count += 1

        # Main directories
        info_id("Creating main directories...", request_id)
        directories_to_check = [
            (config.MODELS_DIR, "Models directory"),
            (config.ORC_TRAINING_DATA_DIR, "Training data directory"),
            (config.ORC_TRAINING_DATA_LOADSHEET, "Training data loadsheet directory"),
            (config.ORC_QUERY_TEMPLATES_TRAIN_DATA_DIR, "Query templates directory"),
        ]

        # Add specific model directories
        info_id("Adding model directories to creation list...", request_id)
        for model_name, model_path in config.MODEL_DIRS.items():
            directories_to_check.append((model_path, f"{model_name.replace('_', ' ').title()} model directory"))

        # Add specific training data directories
        info_id("Adding training data directories to creation list...", request_id)
        for data_name, data_path in config.TRAIN_DATA_DIRS.items():
            directories_to_check.append((data_path, f"{data_name.replace('_', ' ').title()} training data directory"))

        # Add query template specific directories
        info_id("Adding query template directories to creation list...", request_id)
        query_template_dirs = [
            (config.ORC_QUERY_TEMPLATE_PARTS, "Query template parts directory"),
            (config.ORC_QUERY_TEMPLATE_DRAWINGS, "Query template drawings directory"),
        ]
        directories_to_check.extend(query_template_dirs)

        debug_id(f"Total directories to process: {len(directories_to_check) + 1}", request_id)

        # Create all directories
        info_id("Creating/verifying all directories...", request_id)
        for dir_path, description in directories_to_check:
            total_count += 1
            if create_directory(dir_path, description, request_id):
                success_count += 1

    # Summary
    info_id("=" * 60, request_id)
    info_id("Directory setup complete!", request_id)
    info_id(f"Successfully created/verified: {success_count}/{total_count} directories", request_id)

    if success_count == total_count:
        info_id("🎉 All directories are ready!", request_id)
        return 0
    else:
        warning_id(f"⚠️  {total_count - success_count} directories had issues", request_id)
        return 1


@with_request_id
def list_directories():
    """List all directories that will be created/checked."""
    request_id = set_request_id()

    info_id("Listing all directories that will be created/checked", request_id)

    with log_timed_operation("Directory listing", request_id):
        # Collect all directories
        all_dirs = [
            (config.BASE_DIR, "Base directory"),
            (config.MODELS_DIR, "Models directory"),
            (config.ORC_TRAINING_DATA_DIR, "Training data directory"),
            (config.ORC_TRAINING_DATA_LOADSHEET, "Training data loadsheet directory"),
            (config.ORC_QUERY_TEMPLATES_TRAIN_DATA_DIR, "Query templates directory"),
        ]

        # Add model directories
        for model_name, model_path in config.MODEL_DIRS.items():
            all_dirs.append((model_path, f"{model_name.replace('_', ' ').title()} model"))

        # Add training data directories
        for data_name, data_path in config.TRAIN_DATA_DIRS.items():
            all_dirs.append((data_path, f"{data_name.replace('_', ' ').title()} training data"))

        # Add query template directories
        all_dirs.extend([
            (config.ORC_QUERY_TEMPLATE_PARTS, "Query template parts"),
            (config.ORC_QUERY_TEMPLATE_DRAWINGS, "Query template drawings"),
        ])

        # Sort by path for better readability
        all_dirs.sort(key=lambda x: x[0])

        info_id(f"Found {len(all_dirs)} directories to process", request_id)

        for i, (path, description) in enumerate(all_dirs, 1):
            status = "EXISTS" if os.path.exists(path) else "MISSING"
            status_level = debug_id if status == "EXISTS" else warning_id

            info_id(f"{i:2d}. {description}", request_id)
            debug_id(f"    Path: {path}", request_id)
            status_level(f"    Status: {status}", request_id)


if __name__ == "__main__":
    # Set up logging for the script
    if LOGGING_AVAILABLE:
        # Initialize request ID for the entire script run
        script_request_id = set_request_id()
        info_id("Directory setup script started", script_request_id)

    try:
        if len(sys.argv) > 1 and sys.argv[1] in ["-l", "--list"]:
            list_directories()
            exit_code = 0
        else:
            exit_code = main()

        if LOGGING_AVAILABLE:
            if exit_code == 0:
                info_id("Directory setup script completed successfully", script_request_id)
            else:
                warning_id(f"Directory setup script completed with exit code {exit_code}", script_request_id)

        sys.exit(exit_code)

    except KeyboardInterrupt:
        if LOGGING_AVAILABLE:
            warning_id("Directory setup script interrupted by user", script_request_id)
        else:
            print("Script interrupted by user")
        sys.exit(130)

    except Exception as e:
        if LOGGING_AVAILABLE:
            critical_id(f"Directory setup script failed with unexpected error: {str(e)}", script_request_id)
        else:
            print(f"CRITICAL ERROR: {str(e)}")
        sys.exit(1)