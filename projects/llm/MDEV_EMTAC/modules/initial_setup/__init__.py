"""
The 'initial_setup' package contains scripts and utilities for initializing
the AuMaintDB system. This includes database setup, creating an admin user,
loading equipment relationships, parts sheets, and more.

Typical usage:
    from AuMaintdb.modules.initial_setup import (
        create_initial_admin,
        upload_data_from_excel,
        load_parts_sheet,
        load_drawing_list
    )
"""

import logging

# A package-level logger. The NullHandler prevents "No handler found" errors
# in case the parent application does not configure logging for this package.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# If you want to expose functions/classes from each script at the package level,
# import them here and include them in `__all__`.

try:
    from .initial_admin import create_initial_admin
except ImportError:
    create_initial_admin = None

try:
    from .emtacdb_initial_setup import upload_data_from_excel
except ImportError:
    upload_data_from_excel = None

try:
    from .load_equipment_relationships_table_data import load_equipment_relationships
except ImportError:
    load_equipment_relationships = None

try:
    from modules.initial_setup.load_active_drawing_list import load_drawing_list
except ImportError:
    load_drawing_list = None

try:
    from .load_parts_sheet import load_parts_sheet
except ImportError:
    load_parts_sheet = None


# __all__ defines what is imported when someone does "from initial_setup import *".
# Adjust accordingly to match the names you actually want to expose publicly.
__all__ = [
    "create_initial_admin",
    "upload_data_from_excel",
    "load_equipment_relationships",
    "load_drawing_list",
    "load_parts_sheet"
]

# (Optional) you can log a quick debug to confirm that the package was initialized.
# logger.debug("initial_setup package initialized.")
