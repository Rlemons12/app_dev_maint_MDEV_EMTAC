#!/usr/bin/env python3
"""
Directory Explorer Script
Lists all files and directories in a project structure.
Uses custom logging and provides multiple viewing options.
"""

import os
import sys
from pathlib import Path

# Import custom logging
try:
    from modules.configuration.log_config import (
        debug_id, info_id, warning_id, error_id, critical_id,
        set_request_id, with_request_id, log_timed_operation
    )

    LOGGING_AVAILABLE = True
except ImportError:
    print("Warning: log_config.py not found. Using print statements.")
    LOGGING_AVAILABLE = False


    # Fallback functions
    def debug_id(msg, req_id=None):
        pass  # Skip debug in fallback


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


def format_file_size(size_bytes):
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def get_file_info(file_path):
    """Get file information including size and type."""
    try:
        stat = os.stat(file_path)
        size = stat.st_size
        size_str = format_file_size(size)

        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower() if ext else "no extension"

        return {
            'size': size,
            'size_str': size_str,
            'extension': ext
        }
    except OSError:
        return {
            'size': 0,
            'size_str': "Unknown",
            'extension': "unknown"
        }


def list_directory_tree(root_dir, request_id, show_files=True, show_sizes=False, max_depth=None):
    """
    List directory structure in a tree format.

    Args:
        root_dir (str): Root directory to explore
        request_id (str): Request ID for logging
        show_files (bool): Whether to show files or just directories
        show_sizes (bool): Whether to show file sizes
        max_depth (int): Maximum depth to explore (None for unlimited)
    """
    if not os.path.exists(root_dir):
        error_id(f"Directory does not exist: {root_dir}", request_id)
        return False

    if not os.path.isdir(root_dir):
        error_id(f"Path is not a directory: {root_dir}", request_id)
        return False

    info_id(f"Exploring directory tree: {root_dir}", request_id)

    try:
        with log_timed_operation("Directory tree exploration", request_id):
            total_dirs = 0
            total_files = 0
            total_size = 0

            # Print header
            print(f"\n📁 Directory Tree: {root_dir}")
            print("=" * 80)

            for root, dirs, files in os.walk(root_dir):
                # Calculate current depth
                current_depth = root.replace(root_dir, '').count(os.sep)

                # Skip if max depth exceeded
                if max_depth is not None and current_depth >= max_depth:
                    dirs.clear()  # Don't recurse deeper
                    continue

                # Create indentation
                indent = "│   " * current_depth
                folder_name = os.path.basename(root) if current_depth > 0 else os.path.basename(root_dir)

                # Print directory
                if current_depth == 0:
                    print(f"📁 {folder_name}/")
                else:
                    print(f"{indent}├── 📁 {folder_name}/")

                total_dirs += 1

                # Print files if requested
                if show_files:
                    file_indent = "│   " * (current_depth + 1)

                    for i, file in enumerate(files):
                        file_path = os.path.join(root, file)
                        file_info = get_file_info(file_path)
                        total_files += 1
                        total_size += file_info['size']

                        # Determine file icon based on extension
                        ext = file_info['extension']
                        if ext in ['.py']:
                            icon = "🐍"
                        elif ext in ['.json']:
                            icon = "📋"
                        elif ext in ['.txt', '.md']:
                            icon = "📄"
                        elif ext in ['.log']:
                            icon = "📊"
                        elif ext in ['.yml', '.yaml']:
                            icon = "⚙️"
                        elif ext == "no extension":
                            icon = "📄"
                        else:
                            icon = "📄"

                        # Show size if requested
                        size_info = f" ({file_info['size_str']})" if show_sizes else ""

                        # Use different connector for last file
                        connector = "└──" if i == len(files) - 1 else "├──"
                        print(f"{file_indent}{connector} {icon} {file}{size_info}")

            # Print summary
            print("\n" + "=" * 80)
            print(f"📊 Summary:")
            print(f"   Directories: {total_dirs}")
            print(f"   Files: {total_files}")
            if show_sizes:
                print(f"   Total size: {format_file_size(total_size)}")

            info_id(f"Explored {total_dirs} directories and {total_files} files", request_id)
            return True

    except Exception as e:
        error_id(f"Error exploring directory: {str(e)}", request_id)
        return False


def list_files_by_type(root_dir, request_id):
    """List files grouped by their extensions."""
    if not os.path.exists(root_dir):
        error_id(f"Directory does not exist: {root_dir}", request_id)
        return False

    info_id(f"Analyzing files by type in: {root_dir}", request_id)

    try:
        with log_timed_operation("File type analysis", request_id):
            file_types = {}

            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_info = get_file_info(file_path)
                    ext = file_info['extension']

                    if ext not in file_types:
                        file_types[ext] = {
                            'count': 0,
                            'total_size': 0,
                            'files': []
                        }

                    file_types[ext]['count'] += 1
                    file_types[ext]['total_size'] += file_info['size']
                    file_types[ext]['files'].append({
                        'name': file,
                        'path': os.path.relpath(file_path, root_dir),
                        'size': file_info['size'],
                        'size_str': file_info['size_str']
                    })

            # Print results
            print(f"\n📊 Files by Type in: {root_dir}")
            print("=" * 80)

            # Sort by count (descending)
            sorted_types = sorted(file_types.items(), key=lambda x: x[1]['count'], reverse=True)

            for ext, info in sorted_types:
                print(f"\n📁 {ext} files:")
                print(f"   Count: {info['count']}")
                print(f"   Total size: {format_file_size(info['total_size'])}")
                print(f"   Files:")

                # Show up to 10 files per type
                for file_info in info['files'][:10]:
                    print(f"     - {file_info['path']} ({file_info['size_str']})")

                if len(info['files']) > 10:
                    print(f"     ... and {len(info['files']) - 10} more files")

            return True

    except Exception as e:
        error_id(f"Error analyzing files by type: {str(e)}", request_id)
        return False


def find_empty_directories(root_dir, request_id):
    """Find and list empty directories."""
    if not os.path.exists(root_dir):
        error_id(f"Directory does not exist: {root_dir}", request_id)
        return False

    info_id(f"Finding empty directories in: {root_dir}", request_id)

    try:
        with log_timed_operation("Empty directory search", request_id):
            empty_dirs = []

            for root, dirs, files in os.walk(root_dir):
                # Check if directory is empty (no files and no subdirectories with content)
                if not files and not dirs:
                    empty_dirs.append(os.path.relpath(root, root_dir))

            print(f"\n🗂️  Empty Directories in: {root_dir}")
            print("=" * 80)

            if empty_dirs:
                for empty_dir in sorted(empty_dirs):
                    print(f"   📁 {empty_dir}")
                print(f"\nFound {len(empty_dirs)} empty directories")
            else:
                print("   No empty directories found")

            info_id(f"Found {len(empty_dirs)} empty directories", request_id)
            return True

    except Exception as e:
        error_id(f"Error finding empty directories: {str(e)}", request_id)
        return False


@with_request_id
def main():
    """Main function with command line argument handling."""
    request_id = set_request_id()

    # Default to current directory if no arguments
    if len(sys.argv) < 2:
        root_dir = os.getcwd()
        info_id(f"No directory specified, using current directory: {root_dir}", request_id)
    else:
        root_dir = sys.argv[1]

    # Check if directory exists
    if not os.path.exists(root_dir):
        error_id(f"Directory does not exist: {root_dir}", request_id)
        sys.exit(1)

    info_id(f"Starting directory exploration for: {root_dir}", request_id)

    # Run different analyses
    success = True

    # 1. Tree view
    info_id("Running tree view analysis...", request_id)
    if not list_directory_tree(root_dir, request_id, show_files=True, show_sizes=True, max_depth=5):
        success = False

    # 2. Files by type
    info_id("Running file type analysis...", request_id)
    if not list_files_by_type(root_dir, request_id):
        success = False

    # 3. Empty directories
    info_id("Searching for empty directories...", request_id)
    if not find_empty_directories(root_dir, request_id):
        success = False

    if success:
        info_id("Directory exploration completed successfully", request_id)
        return 0
    else:
        warning_id("Directory exploration completed with some errors", request_id)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)