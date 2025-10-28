#!/usr/bin/env python3
"""
Directory Explorer Script
Lists all files and directories in a project structure.
Uses custom logging and provides multiple viewing options.
Exports results to logs/dir_reports/<folder_name>_tree_<timestamp>.md
"""

import os
import sys
import io
from datetime import datetime

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

    def debug_id(msg, req_id=None): pass
    def info_id(msg, req_id=None): print(f"INFO: {msg}")
    def warning_id(msg, req_id=None): print(f"WARNING: {msg}")
    def error_id(msg, req_id=None): print(f"ERROR: {msg}")
    def critical_id(msg, req_id=None): print(f"CRITICAL: {msg}")
    def set_request_id(req_id=None): return "fallback"
    def with_request_id(func): return func

    class log_timed_operation:
        def __init__(self, name, req_id=None): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass


# ==========================================================
# Dual writer to capture + print
# ==========================================================
class Tee(io.TextIOBase):
    """Write to both console and a buffer at the same time."""
    def __init__(self, buffer):
        self.buffer = buffer
        self.console = sys.__stdout__

    def write(self, s):
        self.buffer.write(s)
        self.console.write(s)
        return len(s)

    def flush(self):
        self.buffer.flush()
        self.console.flush()


# ==========================================================
# Helpers
# ==========================================================
def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"


def get_file_info(file_path: str) -> str:
    """Return formatted file info (name + size)."""
    try:
        size = os.path.getsize(file_path)
        return f"{os.path.basename(file_path)} ({format_file_size(size)})"
    except Exception:
        return os.path.basename(file_path)


def calculate_summary(root_dir: str):
    """Walk the tree and calculate directory, file, and size totals."""
    dirs = 0
    files = 0
    total_size = 0
    empty_dirs = 0
    for current, dnames, fnames in os.walk(root_dir):
        dirs += len(dnames)
        files += len(fnames)
        if not dnames and not fnames:
            empty_dirs += 1
        for f in fnames:
            try:
                total_size += os.path.getsize(os.path.join(current, f))
            except Exception:
                pass
    return dirs, files, empty_dirs, total_size


# ==========================================================
# Directory functions
# ==========================================================
def list_directory_tree(root_dir, request_id, show_files=True, show_sizes=False, max_depth=None):
    """List directory tree with optional file details."""
    try:
        print(f"\n📁 Directory Tree: {root_dir}")
        print("=" * 80)

        def _walk(dir_path, prefix="", depth=0):
            if max_depth is not None and depth > max_depth:
                return
            entries = sorted(os.listdir(dir_path))
            for i, entry in enumerate(entries):
                path = os.path.join(dir_path, entry)
                connector = "└── " if i == len(entries) - 1 else "├── "
                if os.path.isdir(path):
                    print(f"{prefix}{connector}📁 {entry}/")
                    _walk(path, prefix + ("    " if i == len(entries) - 1 else "│   "), depth + 1)
                else:
                    if show_files:
                        if show_sizes:
                            print(f"{prefix}{connector}🐍 {get_file_info(path)}")
                        else:
                            print(f"{prefix}{connector}🐍 {entry}")

        print(f"📁 {os.path.basename(root_dir)}/")
        _walk(root_dir, "", 0)
        return True
    except Exception as e:
        error_id(f"Failed to list directory tree: {e}", request_id)
        return False


def list_files_by_type(root_dir, request_id):
    """List files by extension/type with counts and sizes."""
    try:
        print(f"\n📊 Files by Type in: {root_dir}")
        print("=" * 80)

        file_types = {}
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower() or "NO_EXT"
                path = os.path.join(subdir, file)
                size = os.path.getsize(path)
                file_types.setdefault(ext, {"count": 0, "size": 0, "files": []})
                file_types[ext]["count"] += 1
                file_types[ext]["size"] += size
                file_types[ext]["files"].append((file, size))

        for ext, stats in sorted(file_types.items(), key=lambda x: x[0]):
            print(f"\n📁 {ext} files:")
            print(f"   Count: {stats['count']}")
            print(f"   Total size: {format_file_size(stats['size'])}")
            print("   Files:")
            for f, s in sorted(stats["files"], key=lambda x: -x[1])[:10]:
                print(f"     - {f} ({format_file_size(s)})")
        return True
    except Exception as e:
        error_id(f"Failed to list files by type: {e}", request_id)
        return False


def find_empty_directories(root_dir, request_id):
    """Find and list empty directories under root_dir."""
    try:
        print(f"\n🗂️  Empty Directories in: {root_dir}")
        print("=" * 80)
        empty_dirs = []
        for subdir, dirs, files in os.walk(root_dir):
            if not dirs and not files:
                empty_dirs.append(subdir)

        for d in empty_dirs:
            print(f"   📁 {os.path.relpath(d, root_dir)}")

        print(f"\nFound {len(empty_dirs)} empty directories")
        return True
    except Exception as e:
        error_id(f"Failed to find empty directories: {e}", request_id)
        return False


# ==========================================================
# Export Helper
# ==========================================================
def export_output(root_dir: str, content: str, request_id: str) -> str:
    """Export output to logs/dir_reports/<folder_name>_tree_<timestamp>.md with a summary table."""
    folder_name = os.path.basename(os.path.normpath(root_dir))
    output_dir = os.path.join("logs", "dir_reports")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"{folder_name}_tree_{timestamp}.md")

    # Calculate real summary
    dirs, files, empty_dirs, size_bytes = calculate_summary(root_dir)
    total_size = format_file_size(size_bytes)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Directory Exploration Report\n\n")
            f.write(f"**Root:** `{root_dir}`  \n")
            f.write(f"**Generated:** {timestamp}\n\n")

            # Markdown summary table
            f.write("## Summary\n\n")
            f.write("| Metric       | Value |\n")
            f.write("|--------------|-------|\n")
            f.write(f"| Directories  | {dirs} |\n")
            f.write(f"| Files        | {files} |\n")
            f.write(f"| Empty Dirs   | {empty_dirs} |\n")
            f.write(f"| Total Size   | {total_size} |\n\n")

            # Full log in a code block
            f.write("## Full Output\n\n")
            f.write("```\n")
            f.write(content)
            f.write("\n```\n")

        info_id(f"Exported directory exploration to {output_file}", request_id)
        print(f"\n📄 Directory exploration exported to: {output_file}")
        return output_file
    except Exception as e:
        error_id(f"Failed to export output: {e}", request_id)
        return ""


# ==========================================================
# Main
# ==========================================================
@with_request_id
def main():
    """Main function with command line argument handling."""
    request_id = set_request_id()

    args = sys.argv[1:]
    export_enabled = True

    if "--no-export" in args:
        export_enabled = False
        args.remove("--no-export")

    # Default to current directory if no arguments
    if len(args) < 1:
        root_dir = os.getcwd()
        info_id(f"No directory specified, using current directory: {root_dir}", request_id)
    else:
        root_dir = args[0]

    if not os.path.exists(root_dir):
        error_id(f"Directory does not exist: {root_dir}", request_id)
        sys.exit(1)

    info_id(f"Starting directory exploration for: {root_dir}", request_id)

    success = True
    buffer = io.StringIO()
    tee = Tee(buffer)
    old_stdout = sys.stdout
    sys.stdout = tee  # redirect stdout to both console + buffer

    try:
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

        # 🔹 Print summary to console too
        dirs, files, empty_dirs, size_bytes = calculate_summary(root_dir)
        total_size = format_file_size(size_bytes)

        print("\n📊 Summary")
        print("=" * 80)
        print("| Metric       | Value |")
        print("|--------------|-------|")
        print(f"| Directories  | {dirs} |")
        print(f"| Files        | {files} |")
        print(f"| Empty Dirs   | {empty_dirs} |")
        print(f"| Total Size   | {total_size} |")

    finally:
        sys.stdout = old_stdout  # restore stdout

    output_text = buffer.getvalue()
    buffer.close()

    if export_enabled and output_text.strip():
        export_output(root_dir, output_text, request_id)
    elif export_enabled:
        warning_id("No output captured for export.", request_id)
        print("\n⚠️  No output captured for export.")
    else:
        print("\n⚠️  Export disabled by --no-export flag.")

    if success:
        info_id("Directory exploration completed successfully", request_id)
        return 0
    else:
        warning_id("Directory exploration completed with some errors", request_id)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
