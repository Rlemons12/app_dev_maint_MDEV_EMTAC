
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Maintenance Troubleshooting Application
Main entry point for the application
"""

import os
import sys
import logging

# Make sure we import from the correct paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import the logger
from modules.configuration.log_config import logger, initial_log_cleanup

# Import the application
from main_app import MaintenanceTroubleshootingApp

if __name__ == '__main__':
    try:
        # Run initial log cleanup
        initial_log_cleanup()

        logger.info("Starting Maintenance Troubleshooting Application")

        # Run the application
        app = MaintenanceTroubleshootingApp()
        app.run()
    except Exception as e:
        logger.exception(f"Error running application: {e}")