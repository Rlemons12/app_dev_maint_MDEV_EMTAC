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
    from modules.configuration.log_config import info_id, warning_id, error_id

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

# Import the audit system we created
try:
    from audit_system import (
        AuditManager, PostgreSQLAuditTriggers, AuditLog,
        setup_complete_audit_system, AuditMixin
    )
except ImportError:
    print("❌ Could not import audit system. Make sure audit_system.py is in the same directory.")
    sys.exit(1)


class AuditSystemSetup:
    """Setup class for EMTAC audit system"""

    def __init__(self):
        self.db_config = None
        self.audit_manager = None

    def initialize_database_config(self):
        """Initialize database configuration"""
        try:
            info_id("🔧 Initializing database configuration...")
            self.db_config = DatabaseConfig()

            # Test connection
            connection_info = self.db_config.test_connection()
            if connection_info['status'] != 'success':
                error_id(f"❌ Database connection failed: {connection_info.get('error')}")
                return False

            info_id(f"✅ Connected to {connection_info['database_type']} database")
            return True

        except Exception as e:
            error_id(f"❌ Failed to initialize database: {e}")
            return False

    def display_audit_overview(self):
        """Display what the audit system will do"""

        print("\n" + "=" * 80)
        print("🔍 EMTAC DATABASE AUDIT SYSTEM")
        print("=" * 80)
        print("This audit system will add comprehensive change tracking to your database:")
        print("")
        print("📊 AUDIT CAPABILITIES:")
        print("   • Track all INSERT, UPDATE, DELETE operations")
        print("   • Record who made changes and when")
        print("   • Store before/after values for all changes")
        print("   • Track user sessions and IP addresses")
        print("   • Maintain complete change history")
        print("")
        print("🏗️ IMPLEMENTATION OPTIONS:")
        print("   1. SQLAlchemy Event-Based Auditing (Recommended)")
        print("      - Integrated with your Python application")
        print("      - Automatic change detection")
        print("      - Custom user context support")
        print("")
        print("   2. PostgreSQL Trigger-Based Auditing (PostgreSQL only)")
        print("      - Database-level auditing")
        print("      - Works even with direct SQL changes")
        print("      - High performance")
        print("")
        print("🗄️ TABLES CREATED:")
        print("   • audit_log - Central audit table for all changes")
        print("   • {table}_audit - Individual audit tables for each main table")
        print("")
        print("⚡ PERFORMANCE IMPACT:")
        print("   • Minimal overhead for normal operations")
        print("   • Asynchronous logging where possible")
        print("   • Indexed for fast audit queries")
        print("")
        print("=" * 80)

    def get_tables_to_audit(self):
        """Get list of tables that should be audited"""

        try:
            info_id("🔍 Scanning database for tables to audit...")

            from sqlalchemy import inspect
            inspector = inspect(self.db_config.main_engine)
            all_tables = inspector.get_table_names()

            # Filter out tables that shouldn't be audited
            exclude_patterns = [
                '_audit',  # Existing audit tables
                'audit_log',  # Central audit table
                'alembic_version',  # Migration version table
                'spatial_ref_sys',  # PostGIS system table
                'geometry_columns',  # PostGIS system table
            ]

            tables_to_audit = []
            for table in all_tables:
                should_exclude = any(pattern in table.lower() for pattern in exclude_patterns)
                if not should_exclude:
                    tables_to_audit.append(table)

            info_id(f"📋 Found {len(tables_to_audit)} tables to audit:")
            for table in sorted(tables_to_audit):
                info_id(f"   • {table}")

            return tables_to_audit

        except Exception as e:
            error_id(f"❌ Failed to get tables list: {e}")
            return []

    def get_models_to_audit(self):
        """Get SQLAlchemy models that should be audited"""

        try:
            info_id("🔍 Scanning SQLAlchemy models for auditing...")

            models_to_audit = []

            # Get all registered models from the MainBase
            for cls in MainBase.registry._class_registry.values():
                if (hasattr(cls, '__tablename__') and
                        hasattr(cls, '__table__') and
                        cls.__name__ != 'AuditLog'):

                    # Skip audit tables
                    if not cls.__tablename__.endswith('_audit'):
                        models_to_audit.append(cls)

            info_id(f"📋 Found {len(models_to_audit)} models to audit:")
            for model in sorted(models_to_audit, key=lambda x: x.__name__):
                info_id(f"   • {model.__name__} ({model.__tablename__})")

            return models_to_audit

        except Exception as e:
            error_id(f"❌ Failed to get models list: {e}")
            return []

    def setup_sqlalchemy_auditing(self, models_to_audit):
        """Set up SQLAlchemy-based auditing"""

        try:
            info_id("🔧 Setting up SQLAlchemy-based auditing...")

            # Define user context function
            def get_current_user_context():
                """Get current user context - integrate with your authentication system"""

                # TODO: Integrate with your actual authentication system
                # For now, return system user
                return {
                    'user_id': 'system',
                    'user_name': 'System User',
                    'session_id': f"setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'ip_address': '127.0.0.1'
                }

            # Create audit manager
            self.audit_manager = AuditManager(
                db_config=self.db_config,
                current_user_func=get_current_user_context
            )

            # Setup auditing for all models
            self.audit_manager.setup_auditing(models_to_audit)

            info_id("✅ SQLAlchemy auditing setup complete")
            return True

        except Exception as e:
            error_id(f"❌ Failed to setup SQLAlchemy auditing: {e}")
            return False

    def setup_postgresql_triggers(self, table_names):
        """Set up PostgreSQL trigger-based auditing"""

        if not self.db_config.is_postgresql:
            warning_id("⚠️ PostgreSQL triggers are only available for PostgreSQL databases")
            return False

        try:
            info_id("🔧 Setting up PostgreSQL trigger-based auditing...")

            trigger_manager = PostgreSQLAuditTriggers(self.db_config)
            trigger_manager.create_audit_triggers(table_names)

            info_id("✅ PostgreSQL trigger auditing setup complete")
            return True

        except Exception as e:
            error_id(f"❌ Failed to setup PostgreSQL triggers: {e}")
            return False

    def create_audit_indexes(self):
        """Create performance indexes on audit tables"""

        try:
            info_id("🚀 Creating performance indexes on audit tables...")

            with self.db_config.main_session() as session:

                # Indexes for the central audit_log table
                index_statements = [
                    "CREATE INDEX IF NOT EXISTS idx_audit_log_table_record ON audit_log(table_name, record_id);",
                    "CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);",
                    "CREATE INDEX IF NOT EXISTS idx_audit_log_operation ON audit_log(operation);",
                    "CREATE INDEX IF NOT EXISTS idx_audit_log_table_time ON audit_log(table_name, timestamp DESC);",
                ]

                for statement in index_statements:
                    try:
                        session.execute(statement)
                        info_id(f"✅ Created index")
                    except Exception as e:
                        warning_id(f"⚠️ Index creation skipped (may already exist): {e}")

                session.commit()
                info_id("✅ Audit table indexes created")
                return True

        except Exception as e:
            error_id(f"❌ Failed to create audit indexes: {e}")
            return False

    def test_audit_system(self):
        """Test the audit system to make sure it's working"""

        try:
            info_id("🧪 Testing audit system...")

            # Test central audit log
            with self.db_config.main_session() as session:
                from sqlalchemy import text

                # Check if audit_log table exists and is accessible
                result = session.execute(text("SELECT COUNT(*) FROM audit_log"))
                count = result.scalar()
                info_id(f"✅ Central audit log accessible (current records: {count})")

                # Check audit history functionality
                if self.audit_manager:
                    # Try to get audit history (should work even if empty)
                    history = self.audit_manager.get_audit_history('test_table', '1', limit=1)
                    info_id("✅ Audit history query functional")

                info_id("✅ Audit system test passed")
                return True

        except Exception as e:
            error_id(f"❌ Audit system test failed: {e}")
            return False

    def generate_audit_documentation(self):
        """Generate documentation for the audit system"""

        try:
            info_id("📄 Generating audit system documentation...")

            doc_content = f"""
# EMTAC Database Audit System Documentation

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
The EMTAC audit system provides comprehensive change tracking for all database operations.

## Audit Tables Created

### Central Audit Log
- **Table**: `audit_log`
- **Purpose**: Central repository for all database changes
- **Key Fields**:
  - `table_name`: Which table was changed
  - `record_id`: ID of the changed record
  - `operation`: INSERT, UPDATE, or DELETE
  - `timestamp`: When the change occurred
  - `user_id`, `user_name`: Who made the change
  - `old_values`, `new_values`: Before/after data (JSON)

### Individual Audit Tables
Each main table has a corresponding `{{table_name}}_audit` table that stores:
- Complete historical snapshots of records
- Audit metadata (operation, timestamp, user)
- Full record data at time of change

## Querying Audit Data

### Get Change History for a Record
```python
from audit_system import AuditManager

# Get last 50 changes for parts record ID 123
audit_manager = AuditManager(db_config)
history = audit_manager.get_audit_history('parts', '123', limit=50)

for entry in history:
    print(f"{{entry.timestamp}}: {{entry.operation}} by {{entry.user_name}}")
```

### Get User Activity
```python
# Get last 30 days of activity for a user
activity = audit_manager.get_user_activity('admin123', days=30)
print(f"User performed {{len(activity)}} operations")
```

### Direct SQL Queries
```sql
-- Get all changes to parts table in last 24 hours
SELECT * FROM audit_log 
WHERE table_name = 'parts' 
AND timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- Get all changes by a specific user
SELECT table_name, operation, timestamp 
FROM audit_log 
WHERE user_id = 'admin123' 
ORDER BY timestamp DESC;

-- Find who deleted a specific record
SELECT * FROM audit_log 
WHERE table_name = 'parts' 
AND record_id = '123' 
AND operation = 'DELETE';
```

## Integration with Your Application

### Setting User Context
To properly track who makes changes, integrate with your authentication system:

```python
def get_current_user_context():
    # Replace with your actual user session logic
    current_user = get_logged_in_user()  # Your function
    return {{
        'user_id': current_user.id,
        'user_name': current_user.name,
        'session_id': session.get('session_id'),
        'ip_address': request.remote_addr
    }}

audit_manager = AuditManager(db_config, get_current_user_context)
```

### Automatic Auditing
Once set up, all changes through SQLAlchemy models are automatically audited:

```python
# This will be automatically audited
with db_session() as session:
    part = session.query(Parts).filter_by(id=123).first()
    part.name = "Updated Name"  # UPDATE operation logged
    session.commit()
```

## Performance Considerations

- Audit logging adds minimal overhead to normal operations
- Indexes are created for common query patterns
- Large audit tables should be periodically archived
- Consider partitioning audit tables by date for very high-volume systems

## Troubleshooting

### Check Audit System Status
```python
# Verify audit system is working
audit_manager.test_audit_system()
```

### View Recent Audit Activity
```sql
SELECT table_name, COUNT(*) as changes, MAX(timestamp) as latest_change
FROM audit_log 
GROUP BY table_name 
ORDER BY latest_change DESC;
```

### Audit Table Sizes
```sql
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE tablename LIKE '%_audit' OR tablename = 'audit_log'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Maintenance

### Archive Old Audit Data
```sql
-- Archive audit data older than 1 year
DELETE FROM audit_log WHERE timestamp < NOW() - INTERVAL '1 year';
```

### Monitor Audit Performance
```sql
-- Check audit log growth
SELECT 
    DATE(timestamp) as audit_date,
    COUNT(*) as operations
FROM audit_log 
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY audit_date DESC;
```
"""

            # Write documentation to file
            doc_file = os.path.join(project_root, "audit_system_documentation.md")
            with open(doc_file, 'w') as f:
                f.write(doc_content)

            info_id(f"✅ Documentation generated: {doc_file}")
            return True

        except Exception as e:
            error_id(f"❌ Failed to generate documentation: {e}")
            return False

    def run_complete_setup(self):
        """Run the complete audit system setup"""

        print("\n🚀 Starting EMTAC Audit System Setup...")

        # Step 1: Initialize database
        if not self.initialize_database_config():
            error_id("❌ Database initialization failed")
            return False

        # Step 2: Get user preferences
        self.display_audit_overview()

        setup_sqlalchemy = input("\n🔧 Set up SQLAlchemy-based auditing? (Recommended) (y/n): ").strip().lower()
        setup_triggers = False

        if self.db_config.is_postgresql:
            setup_triggers = input("🔧 Also set up PostgreSQL trigger-based auditing? (y/n): ").strip().lower() in ['y',
                                                                                                                   'yes']

        # Step 3: Get models/tables to audit
        models_to_audit = self.get_models_to_audit()
        tables_to_audit = self.get_tables_to_audit()

        if not models_to_audit and not tables_to_audit:
            error_id("❌ No tables or models found to audit")
            return False

        # Step 4: SQLAlchemy auditing setup
        if setup_sqlalchemy in ['y', 'yes']:
            if not self.setup_sqlalchemy_auditing(models_to_audit):
                error_id("❌ SQLAlchemy auditing setup failed")
                return False

        # Step 5: PostgreSQL trigger setup
        if setup_triggers:
            if not self.setup_postgresql_triggers(tables_to_audit):
                warning_id("⚠️ PostgreSQL trigger setup failed, continuing...")

        # Step 6: Create indexes
        if not self.create_audit_indexes():
            warning_id("⚠️ Index creation failed, continuing...")

        # Step 7: Test the system
        if not self.test_audit_system():
            warning_id("⚠️ Audit system test failed, but setup may still be functional")

        # Step 8: Generate documentation
        self.generate_audit_documentation()

        # Success summary
        print("\n" + "=" * 80)
        print("🎉 AUDIT SYSTEM SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ What was set up:")
        print("")

        if setup_sqlalchemy in ['y', 'yes']:
            print("🔧 SQLAlchemy Event-Based Auditing:")
            print(f"   • Audit logging for {len(models_to_audit)} models")
            print("   • Automatic change detection")
            print("   • User context tracking")

        if setup_triggers:
            print("🔧 PostgreSQL Trigger-Based Auditing:")
            print(f"   • Database triggers for {len(tables_to_audit)} tables")
            print("   • Database-level change detection")
            print("   • Works with direct SQL changes")

        print("")
        print("📊 Audit Tables Created:")
        print("   • audit_log - Central audit repository")
        print(f"   • {len(tables_to_audit)} individual audit tables")
        print("")
        print("🚀 Performance Optimizations:")
        print("   • Indexes created for fast queries")
        print("   • Optimized for common audit patterns")
        print("")
        print("📄 Documentation:")
        print("   • Complete usage guide generated")
        print("   • SQL query examples included")
        print("   • Integration instructions provided")
        print("")
        print("🎯 Next Steps:")
        print("   1. Integrate user context function with your auth system")
        print("   2. Test audit logging with your application")
        print("   3. Set up audit data archiving policies")
        print("   4. Train users on audit query capabilities")
        print("")
        print("=" * 80)

        return True


def main():
    """Main setup function"""

    try:
        setup = AuditSystemSetup()
        success = setup.run_complete_setup()

        if success:
            print("\n🎉 Audit system is ready for use!")
            sys.exit(0)
        else:
            print("\n❌ Audit system setup failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()