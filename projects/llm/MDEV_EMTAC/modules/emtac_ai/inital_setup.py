#!/usr/bin/env python3
"""
Initial Setup Script
Complete project initialization that:
1. Creates all directories defined in config.py
2. Downloads and saves all base models (intent classifier + NER models)

Uses custom logging from log_config.py for consistent logging across the application.
"""

import os
import sys
from pathlib import Path

# Import the config file
try:
    import config
except ImportError:
    # Try to find config.py in the current directory
    import sys
    import os

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Try different possible locations for config.py
    possible_config_paths = [
        script_dir,  # Same directory as script
        os.path.join(script_dir, '..'),  # Parent directory
        os.path.join(script_dir, '..', '..'),  # Grandparent directory
    ]

    config_found = False
    for path in possible_config_paths:
        config_path = os.path.join(path, 'config.py')
        if os.path.exists(config_path):
            sys.path.insert(0, path)
            try:
                import config

                config_found = True
                print(f"Found config.py at: {config_path}")
                break
            except ImportError:
                continue

    if not config_found:
        print("Error: config.py not found in any of the expected locations:")
        for path in possible_config_paths:
            print(f"  - {os.path.join(path, 'config.py')}")
        print("Make sure config.py exists and is accessible.")
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

# Import transformers (with error handling)
try:
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    critical_id("Transformers library not found. Please install it with: pip install transformers torch")


def create_init_file(directory, package_name, description, request_id):
    """
    Create an __init__.py file with documentation.

    Args:
        directory (str): Directory where to create __init__.py
        package_name (str): Name of the package
        description (str): Description for the package
        request_id (str): Request ID for logging

    Returns:
        bool: True if successful, False otherwise
    """
    init_file_path = os.path.join(directory, "__init__.py")

    try:
        # Generate appropriate documentation based on package type
        if "model" in package_name.lower():
            content = f'"""\n{package_name.replace("_", " ").title()} Package\n\n{description}\n\nThis package contains trained AI models and related utilities.\n"""\n\n__version__ = "1.0.0"\n'
        elif "training_data" in package_name.lower() or "data" in package_name.lower():
            content = f'"""\n{package_name.replace("_", " ").title()} Package\n\n{description}\n\nThis package contains training data and datasets.\n"""\n\n__version__ = "1.0.0"\n'
        elif "template" in package_name.lower():
            content = f'"""\n{package_name.replace("_", " ").title()} Package\n\n{description}\n\nThis package contains query templates and related utilities.\n"""\n\n__version__ = "1.0.0"\n'
        else:
            content = f'"""\n{package_name.replace("_", " ").title()} Package\n\n{description}\n"""\n\n__version__ = "1.0.0"\n'

        with open(init_file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        debug_id(f"✓ Created __init__.py in {directory}", request_id)
        return True

    except Exception as e:
        warning_id(f"✗ Failed to create __init__.py in {directory}: {str(e)}", request_id)
        return False


def create_directory(path, description, request_id, create_init=True):
    """
    Create a directory if it doesn't exist, optionally with __init__.py file.

    Args:
        path (str): The directory path to create
        description (str): Human-readable description for logging
        request_id (str): Request ID for logging correlation
        create_init (bool): Whether to create __init__.py file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        debug_id(f"Attempting to create directory: {path}", request_id)

        with log_timed_operation(f"Create {description}", request_id):
            os.makedirs(path, exist_ok=True)

        if os.path.exists(path):
            info_id(f"✓ Successfully verified/created {description}: {path}", request_id)

            # Create __init__.py file if requested and it's a Python package directory
            if create_init and not path.endswith(('.log', '.txt', '.json', '.md')):
                package_name = os.path.basename(path)
                create_init_file(path, package_name, description, request_id)

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


def setup_directories(request_id):
    """
    Create all directories from config.

    Returns:
        tuple: (success_count, total_count)
    """
    info_id("=" * 60, request_id)
    info_id("PHASE 1: Setting up directory structure", request_id)
    info_id("=" * 60, request_id)

    with log_timed_operation("Complete directory setup", request_id):
        success_count = 0
        total_count = 0

        # Base directory (don't create __init__.py in root)
        info_id("Creating base directory...", request_id)
        total_count += 1
        if create_directory(config.BASE_DIR, "Base directory", request_id, create_init=False):
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
        info_id("Creating/verifying all directories with __init__.py files...", request_id)
        for dir_path, description in directories_to_check:
            total_count += 1
            # Don't create __init__.py for certain directories
            skip_init = any(skip_word in description.lower() for skip_word in ['loadsheet', 'backup', 'log'])
            if create_directory(dir_path, description, request_id, create_init=not skip_init):
                success_count += 1

    return success_count, total_count


def save_base_intent_model(model_name="distilbert-base-uncased", output_dir=None, request_id=None):
    """Download and save base intent classification model."""
    info_id(f"Downloading and saving base intent model to {output_dir}...", request_id)

    try:
        with log_timed_operation("Download intent classifier model", request_id):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)  # 6 intents

        with log_timed_operation("Save intent classifier model", request_id):
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        info_id("✓ Base intent model saved successfully", request_id)
        return True

    except Exception as e:
        error_id(f"✗ Failed to download/save intent model: {str(e)}", request_id)
        return False


def save_base_ner_model(model_name="dslim/bert-base-NER", output_dir=None, intent_type=None, request_id=None):
    """Download and save base NER model."""
    info_id(f"Downloading and saving base NER model for {intent_type} to {output_dir}...", request_id)

    try:
        with log_timed_operation(f"Download NER model for {intent_type}", request_id):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)

        with log_timed_operation(f"Save NER model for {intent_type}", request_id):
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        info_id(f"✓ Base NER model for {intent_type} saved successfully", request_id)
        return True

    except Exception as e:
        error_id(f"✗ Failed to download/save NER model for {intent_type}: {str(e)}", request_id)
        return False


def setup_models(request_id):
    """
    Download and save all base models.

    Returns:
        tuple: (success_count, total_count)
    """
    if not TRANSFORMERS_AVAILABLE:
        error_id("Cannot setup models: transformers library not available", request_id)
        return 0, 7  # 0 success, 7 total models

    info_id("=" * 60, request_id)
    info_id("PHASE 2: Setting up base models", request_id)
    info_id("=" * 60, request_id)

    success_count = 0
    total_count = 0

    with log_timed_operation("Complete model setup", request_id):
        # Save intent classifier base model
        info_id("Setting up intent classifier model...", request_id)
        total_count += 1
        if save_base_intent_model(output_dir=config.MODEL_DIRS["intent_classifier"], request_id=request_id):
            success_count += 1

        # Save base NER models for all specialized intents
        info_id("Setting up NER models for all intents...", request_id)
        intent_types = ["parts", "images", "documents", "drawings", "tools", "troubleshooting"]

        for intent in intent_types:
            info_id(f"Setting up base NER model for intent: {intent}", request_id)
            total_count += 1
            if save_base_ner_model(
                    output_dir=config.MODEL_DIRS[intent],
                    intent_type=intent,
                    request_id=request_id
            ):
                success_count += 1

    return success_count, total_count


@with_request_id
def main():
    """Main function to run complete initial setup."""
    request_id = set_request_id()

    info_id("🚀 Starting EMTAC AI Initial Setup", request_id)
    info_id("This will create all directories and download base models", request_id)

    overall_start_time = None
    if LOGGING_AVAILABLE:
        import time
        overall_start_time = time.time()

    try:
        with log_timed_operation("Complete initial setup", request_id):
            # Phase 1: Setup directories
            dir_success, dir_total = setup_directories(request_id)

            # Phase 2: Setup models
            model_success, model_total = setup_models(request_id)

            # Final summary
            info_id("=" * 60, request_id)
            info_id("INITIAL SETUP COMPLETE", request_id)
            info_id("=" * 60, request_id)

            total_success = dir_success + model_success
            total_operations = dir_total + model_total

            info_id(f"Directory Setup: {dir_success}/{dir_total} successful", request_id)
            info_id(f"Model Setup: {model_success}/{model_total} successful", request_id)
            info_id(f"Overall: {total_success}/{total_operations} operations successful", request_id)

            if overall_start_time:
                duration = time.time() - overall_start_time
                info_id(f"Total setup time: {duration:.2f} seconds", request_id)

            if total_success == total_operations:
                info_id("🎉 Initial setup completed successfully!", request_id)
                info_id("Your EMTAC AI environment is ready for training and development.", request_id)
                return 0
            else:
                warning_id(f"⚠️  Setup completed with {total_operations - total_success} issues", request_id)
                warning_id("Please check the logs above for details on failed operations.", request_id)
                return 1

    except KeyboardInterrupt:
        warning_id("Initial setup interrupted by user", request_id)
        return 130
    except Exception as e:
        critical_id(f"Initial setup failed with unexpected error: {str(e)}", request_id)
        return 1


@with_request_id
def check_prerequisites():
    """Check if all prerequisites are available."""
    request_id = set_request_id()

    info_id("Checking prerequisites...", request_id)

    issues = []

    # Check if config.py exists
    if not os.path.exists('config.py'):
        issues.append("config.py not found")

    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        issues.append("transformers library not installed (pip install transformers torch)")

    # Check if we have internet connection (basic check)
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=5)
    except:
        issues.append("No internet connection (required for downloading models)")

    if issues:
        error_id("Prerequisites check failed:", request_id)
        for issue in issues:
            error_id(f"  - {issue}", request_id)
        return False
    else:
        info_id("✓ All prerequisites check passed", request_id)
        return True


if __name__ == "__main__":
    # Set up logging for the script
    if LOGGING_AVAILABLE:
        script_request_id = set_request_id()
        info_id("Initial setup script started", script_request_id)

    # Check prerequisites first
    if not check_prerequisites():
        sys.exit(1)

    # Run main setup
    exit_code = main()

    if LOGGING_AVAILABLE:
        if exit_code == 0:
            info_id("Initial setup script completed successfully", script_request_id)
        else:
            warning_id(f"Initial setup script completed with exit code {exit_code}", script_request_id)

    sys.exit(exit_code)