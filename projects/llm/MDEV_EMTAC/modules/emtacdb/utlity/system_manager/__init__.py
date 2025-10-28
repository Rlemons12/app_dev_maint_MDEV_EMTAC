# modules/system_resources/__init__.py
"""
System Resource Management Module
=================================

This module provides utilities for monitoring and optimizing system resources
in a Flask application with SQLite database backend. It integrates with the
existing logging and database connection management infrastructure.

The primary class is SystemResourceManager, which contains methods for:
- Calculating optimal worker/thread counts
- Optimizing SQLite database performance
- Monitoring Flask request performance
- Auditing and managing database connections
- Monitoring system resources (CPU, memory, disk)
- Analyzing slow database queries
- Optimizing static file serving
- Configuring rate limiting
- Performing health checks

Example usage:

    from modules.system_resources import SystemResourceManager

    # During application initialization
    app = Flask(__name__)

    # Optimize SQLite database
    SystemResourceManager.optimize_sqlite_performance('instance/app.db')

    # Set up performance monitoring
    stats_func = SystemResourceManager.monitor_flask_performance(app)

    # Start system resource monitoring
    resource_monitor = SystemResourceManager.monitor_system_resources()

    # In a health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify(SystemResourceManager.health_check())
"""

# Version information
__version__ = '0.1.0'

# Import the main class to make it available at the package level
from .system_resource_manager import SystemResourceManager

# Make the class directly available when importing the package
__all__ = ['SystemResourceManager']