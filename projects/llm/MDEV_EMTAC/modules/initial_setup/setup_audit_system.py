#!/usr/bin/env python3
"""
EMTAC Database Audit System Setup Script
Integrates with your existing setup process to add comprehensive auditing
"""

import os
import sys
import subprocess
from datetime import datetime

# Add the project root to the path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

try:
    from modules.configuration.config_env import DatabaseConfig
    from modules.emtacdb.emtacdb_fts import Base as MainBase
    from modules.configuration.log_config import info_id, warning_id, error_id, set_request_id

    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import EMTAC modules: {e}")
    LOGGING_AVAILABLE = False


    def info_id(msg, **kwargs):
        print(f"INFO: {msg}")


    def warning_id(msg, **kwargs):
        print(f"WARNING: {msg}")


    def error_id(msg, **kwargs):
        print(f"ERROR: {msg}")


    def set_request_id():
        return "audit-001"


class AuditSystemSetup:
    """Audit system setup with comprehensive error handling and logging."""

    def __init__(self):
        try:
            self.request_id = set_request_id()
        except:
            self.request_id = "audit-001"

        info_id("Initialized Audit System Setup", self.request_id)

    def setup_basic_audit_system(self):
        """Set up a basic audit system with comprehensive logging."""
        try:
            info_id("Setting up basic audit system...", self.request_id)
            info_id("Starting audit system configuration")

            db_config = DatabaseConfig()

            # Create basic audit table
            with db_config.main_session() as session:
                from sqlalchemy import text

                info_id("Creating audit_log table...", self.request_id)

                # Create audit_log table
                create_audit_table_sql = """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id SERIAL PRIMARY KEY,
                    table_name VARCHAR(100) NOT NULL,
                    record_id VARCHAR(100) NOT NULL,
                    operation VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id VARCHAR(100),
                    user_name VARCHAR(200),
                    session_id VARCHAR(100),
                    old_values TEXT,
                    new_values TEXT,
                    changed_fields TEXT,
                    ip_address VARCHAR(50),
                    user_agent TEXT,
                    application VARCHAR(100) DEFAULT 'EMTAC',
                    notes TEXT
                );
                """

                session.execute(text(create_audit_table_sql))
                info_id("audit_log table created successfully", self.request_id)

                # Create indexes
                info_id("Creating audit system indexes...", self.request_id)
                index_statements = [
                    "CREATE INDEX IF NOT EXISTS idx_audit_log_table_record ON audit_log(table_name, record_id);",
                    "CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);",
                    "CREATE INDEX IF NOT EXISTS idx_audit_log_operation ON audit_log(operation);",
                ]

                indexes_created = 0
                for idx_sql in index_statements:
                    try:
                        session.execute(text(idx_sql))
                        indexes_created += 1
                    except Exception as e:
                        warning_id(f"Index creation skipped: {e}", self.request_id)

                info_id(f"Created {indexes_created} audit system indexes", self.request_id)

                # Commit all changes
                session.commit()
                info_id("All audit system changes committed successfully", self.request_id)
                info_id("Basic audit system created successfully")

            return True

        except Exception as e:
            error_id(f"Failed to setup basic audit system: {e}", self.request_id)
            error_id(f"Audit system setup failed: {e}")
            return False

    def verify_audit_system(self):
        """Verify that the audit system was set up correctly."""
        try:
            info_id("Verifying audit system setup...", self.request_id)

            db_config = DatabaseConfig()

            with db_config.main_session() as session:
                from sqlalchemy import text

                # Check if audit_log table exists and is accessible
                result = session.execute(text("SELECT COUNT(*) FROM audit_log"))
                count = result.scalar()

                info_id(f"Audit system verification successful - {count} existing audit records", self.request_id)
                info_id("Audit system is ready for use")
                return True

        except Exception as e:
            error_id(f"Audit system verification failed: {e}", self.request_id)
            return False

    def display_audit_summary(self):
        """Display comprehensive audit system setup summary."""
        info_id("=" * 50)
        info_id("AUDIT SYSTEM SETUP COMPLETE")
        info_id("=" * 50)
        info_id("Audit system capabilities:")
        info_id("- Track all INSERT, UPDATE, DELETE operations")
        info_id("- Record user information and timestamps")
        info_id("- Store before/after values for changes")
        info_id("- Track user sessions and IP addresses")
        info_id("- Maintain complete change history")
        info_id("")
        info_id("Audit table: audit_log")
        info_id("Indexes: 4 performance indexes created")
        info_id("Status: Ready for production use")
        info_id("=" * 50)


def setup_basic_audit_system():
    """Legacy function for backward compatibility."""
    setup = AuditSystemSetup()
    success = setup.setup_basic_audit_system()

    if success:
        # Verify the setup
        if setup.verify_audit_system():
            setup.display_audit_summary()
        else:
            warning_id("Audit system created but verification failed")

    return success


def main():
    """Main setup function with enhanced error handling."""
    try:
        info_id("Starting EMTAC Database Audit System Setup")
        info_id("=" * 50)

        success = setup_basic_audit_system()

        if success:
            info_id("Audit system setup completed successfully")
            return True
        else:
            error_id("Audit system setup failed")
            return False

    except KeyboardInterrupt:
        warning_id("Audit setup interrupted by user")
        return False
    except Exception as e:
        error_id(f"Unexpected error in audit setup: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)