#!/usr/bin/env python3
"""
Setup Dependencies Script

This script checks and installs required dependencies for the part-image matching tool.
"""

import subprocess
import sys
import importlib


def check_and_install_package(package_name, import_name=None):
    """
    Check if a package is installed, and install it if not.

    Args:
        package_name: Name of the package to install (e.g., 'pandas')
        import_name: Name to use for import (defaults to package_name)
    """
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("SETTING UP DEPENDENCIES FOR PART-IMAGE MATCHING")
    print("=" * 60)

    # List of required packages
    required_packages = [
        ('pandas', None),
        ('xlrd', None),  # For reading .xls files
        ('openpyxl', None),  # For reading .xlsx files (backup)
        ('sqlalchemy', None),  # Database ORM
    ]

    print("Checking and installing required packages...\n")

    all_success = True

    for package_name, import_name in required_packages:
        success = check_and_install_package(package_name, import_name)
        all_success = all_success and success
        print()

    print("=" * 60)
    if all_success:
        print("🎉 ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("\nYou're ready to run:")
        print("1. excel_reader_test.py (to test reading your Excel file)")
        print("2. custom_part_image_matcher.py (for the full matching process)")
    else:
        print("❌ SOME DEPENDENCIES FAILED TO INSTALL")
        print("\nTry installing manually:")
        print("pip install pandas xlrd openpyxl sqlalchemy")

    print("=" * 60)


if __name__ == "__main__":
    main()