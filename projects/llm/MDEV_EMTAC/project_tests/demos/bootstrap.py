"""
Bootstrap for demo scripts that need to load project modules
from directories *higher* in the tree.

This avoids modifying PyCharm source roots.
"""

import sys
import os


def bootstrap_paths():
    """
    Add the REAL EMTAC project root to sys.path,
    no matter where this demo script is located.
    """

    # Path to this file: .../project_tests/demos/bootstrap.py
    this_file = os.path.abspath(__file__)

    # Step UP to /demos
    demos_dir = os.path.dirname(this_file)

    # Step up to /project_tests
    tests_dir = os.path.dirname(demos_dir)

    # Step up to /MDEV_EMTAC  <-- THIS IS YOUR PROJECT ROOT
    project_root = os.path.dirname(tests_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    return project_root
