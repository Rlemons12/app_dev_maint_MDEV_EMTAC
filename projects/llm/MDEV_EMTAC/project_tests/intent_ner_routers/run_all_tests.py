"""
Simple script to run all pytest tests inside a given folder.

Usage:
    python run_all_tests.py
"""

import os
import sys
import pytest


def main():
    # Change this to the folder you want to test
    TEST_FOLDER = os.path.join(os.path.dirname(__file__), "")

    print(f"Running all tests in: {TEST_FOLDER}")

    # Run pytest programmatically
    exit_code = pytest.main([TEST_FOLDER, "-q"])

    # Return pytest exit code to shell
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
