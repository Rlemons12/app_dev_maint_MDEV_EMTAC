#!/usr/bin/env python3
"""
Directory Explorer Script
Generates:
  1. A full text .log report (folders, files, summary)
  2. An interactive HTML overview with collapsible folders
"""

import os
import sys
import io
import datetime
import webbrowser
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging imports and fallback
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
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
        _, ext = os.path.splitext(file_path)
        ext = ext.lower() if ext else "no extension"
        return {"size": size, "size_str": size_str, "extension": ext}
    except OSError:
        return {"size": 0, "size_str": "Unknown", "extension": "unknown"}


# ---------------------------------------------------------------------------
# Folder-only (for HTML and log)
# ---------------------------------------------------------------------------
def build_folder_tree(root_dir, max_depth=None):
    """Return folder structure as nested dictionary."""
    tree = {}
    for root, dirs, files in os.walk(root_dir):
        rel_path = os.path.relpath(root, root_dir)
        depth = rel_path.count(os.sep)
        if max_depth is not None and depth >= max_depth:
            dirs.clear()
            continue
        sub_tree = tree
        if rel_path != ".":
            for part in rel_path.split(os.sep):
                sub_tree = sub_tree.setdefault(part, {})
        for d in dirs:
            sub_tree[d] = {}
    return tree


def print_folder_tree(tree, indent=0):
    """Return text representation of folder tree (folders only)."""
    lines = []
    for name, sub in sorted(tree.items()):
        lines.append("    " * indent + f"{name}/")
        lines.extend(print_folder_tree(sub, indent + 1))
    return lines


def folder_tree_to_html(tree):
    """Convert nested dict folder structure to HTML <details>."""
    html = []
    for name, sub in sorted(tree.items()):
        if sub:
            html.append(f"<details open><summary>{name}/</summary>")
            html.append(folder_tree_to_html(sub))
            html.append("</details>")
        else:
            html.append(f"<details><summary>{name}/</summary></details>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Detailed directory tree
# ---------------------------------------------------------------------------
def list_detailed_tree(root_dir, show_sizes=True, max_depth=None):
    """Return string for full directory tree including files."""
    output = io.StringIO()
    total_dirs, total_files, total_size = 0, 0, 0

    output.write(f"\nDETAILED DIRECTORY TREE (With Files): {root_dir}\n")
    output.write("=" * 80 + "\n")

    for root, dirs, files in os.walk(root_dir):
        depth = root.replace(root_dir, "").count(os.sep)
        if max_depth is not None and depth >= max_depth:
            dirs.clear()
            continue

        indent = "    " * depth
        folder_name = os.path.basename(root) if depth > 0 else os.path.basename(root_dir)
        output.write(f"{indent}{folder_name}/\n")
        total_dirs += 1

        file_indent = "    " * (depth + 1)
        for file in files:
            file_path = os.path.join(root, file)
            info = get_file_info(file_path)
            total_files += 1
            total_size += info["size"]
            size_info = f" ({info['size_str']})" if show_sizes else ""
            output.write(f"{file_indent}{file}{size_info}\n")

    output.write("\nSummary:\n")
    output.write(f"   Total Directories: {total_dirs}\n")
    output.write(f"   Total Files: {total_files}\n")
    if show_sizes:
        output.write(f"   Total Size: {format_file_size(total_size)}\n")
    output.write("=" * 80 + "\n")

    return output.getvalue()


# ---------------------------------------------------------------------------
# Empty directory finder
# ---------------------------------------------------------------------------
def find_empty_directories(root_dir):
    """Return list of empty directories."""
    empty_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if not files and not dirs:
            empty_dirs.append(os.path.relpath(root, root_dir))
    return empty_dirs


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
LOG_DIR = r"E:\emtac\logs"

def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def select_directory_gui():
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Select directory to explore")
        root.destroy()
        return selected or None
    except Exception:
        return None


@with_request_id
def main():
    request_id = set_request_id()
    ensure_log_dir()

    # Select directory
    if len(sys.argv) < 2:
        info_id("No directory provided. Opening folder chooser.", request_id)
        root_dir = select_directory_gui()
        if not root_dir:
            root_dir = input("Enter directory path to explore: ").strip()
            if not root_dir:
                error_id("No directory specified. Exiting.", request_id)
                sys.exit(1)
    else:
        root_dir = sys.argv[1]

    if not os.path.isdir(root_dir):
        error_id(f"Invalid directory: {root_dir}", request_id)
        sys.exit(1)

    # Filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = Path(root_dir).name or "root"
    log_file = os.path.join(LOG_DIR, f"directory_explorer_{safe_name}_{timestamp}.log")
    html_file = os.path.join(LOG_DIR, f"directory_overview_{safe_name}_{timestamp}.html")

    info_id(f"Saving outputs to:\n  {log_file}\n  {html_file}", request_id)

    # 1️⃣ Build folder tree (folders only)
    folder_tree = build_folder_tree(root_dir, max_depth=None)
    folder_summary_lines = print_folder_tree(folder_tree)

    # 2️⃣ Build detailed report
    detailed_tree_str = list_detailed_tree(root_dir, show_sizes=True)

    # 3️⃣ Find empty directories
    empty_dirs = find_empty_directories(root_dir)

    # --- Write full log ---
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("FOLDER STRUCTURE (Folders Only)\n")
        f.write("=" * 80 + "\n")
        f.write("\n".join(folder_summary_lines))
        f.write("\n\n" + detailed_tree_str)
        if empty_dirs:
            f.write("\nEMPTY DIRECTORIES:\n" + "\n".join(empty_dirs))
        else:
            f.write("\nNo empty directories found.\n")

    # --- Write HTML overview ---
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Directory Overview - {safe_name}</title>
<style>
body {{ font-family: Consolas, monospace; background: #f5f5f5; padding: 20px; }}
details {{ margin-left: 20px; }}
summary {{ font-weight: bold; cursor: pointer; }}
h1 {{ color: #333; }}
</style>
</head>
<body>
<h1>Directory Overview: {root_dir}</h1>
{folder_tree_to_html(folder_tree)}
<p><small>Generated on {datetime.datetime.now()}</small></p>
</body>
</html>
"""
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Auto-open HTML file
    webbrowser.open(f"file://{html_file}", new=2)

    print(f"Log saved to: {log_file}")
    print(f"HTML overview saved to: {html_file}")
    info_id("Directory exploration completed successfully.", request_id)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
