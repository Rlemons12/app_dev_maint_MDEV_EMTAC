import os
import re
import sys
from pathlib import Path

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------
# Adjust these paths if needed
PROJECT_DIR = Path(r"E:\emtac\projects\llm\MDEV_EMTAC")
OLD_VENV_PATH = PROJECT_DIR / ".venv_old" / "Lib" / "site-packages"
OUTPUT_FILE = PROJECT_DIR / "requirements.txt"

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

def extract_name_version(folder_name: str):
    """
    Extract package name and version from a .dist-info or .egg-info folder name.
    Examples:
        "Flask-3.0.2.dist-info" -> ("Flask", "3.0.2")
        "SQLAlchemy-2.0.21-py3.10.egg-info" -> ("SQLAlchemy", "2.0.21")
    """
    match = re.match(r"([A-Za-z0-9_.\-]+)-(\d+(?:\.\d+){0,3})", folder_name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def scan_site_packages(site_packages_path: Path):
    """
    Scan the site-packages directory for package metadata.
    """
    packages = {}
    if not site_packages_path.exists():
        print(f"[ERROR] Path not found: {site_packages_path}")
        sys.exit(1)

    for entry in site_packages_path.iterdir():
        if entry.is_dir() and (entry.name.endswith(".dist-info") or entry.name.endswith(".egg-info")):
            name, version = extract_name_version(entry.name)
            if name and version:
                packages[name] = version
        elif entry.is_dir() and "-" not in entry.name and not entry.name.endswith(".dist-info"):
            # fallback for simple packages without version info
            if entry.name.lower() not in packages:
                packages[entry.name] = None

    return packages


def write_requirements(packages: dict, output_file: Path):
    """
    Write the extracted package list into requirements.txt
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for name, version in sorted(packages.items()):
            if version:
                f.write(f"{name}=={version}\n")
            else:
                f.write(f"{name}\n")
    print(f"[SUCCESS] requirements.txt written to: {output_file}")
    print(f"Total packages found: {len(packages)}")


# -----------------------------------------
# MAIN EXECUTION
# -----------------------------------------

if __name__ == "__main__":
    print(f"Scanning old venv site-packages: {OLD_VENV_PATH}")
    packages = scan_site_packages(OLD_VENV_PATH)
    write_requirements(packages, OUTPUT_FILE)
