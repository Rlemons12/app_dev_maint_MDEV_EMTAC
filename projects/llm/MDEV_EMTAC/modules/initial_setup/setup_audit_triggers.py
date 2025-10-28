#!/usr/bin/env python3
"""
PostgreSQL Audit Triggers Setup Script
Creates comprehensive database triggers for automatic audit logging
"""

import os
import sys
import json
from datetime import datetime

# Add the project root to the path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

try:
    from modules.configuration.config_env import DatabaseConfig
    from modules.configuration.log_config import info_id, warning_id, error_id
    from sqlalchemy import text, inspect

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


class AuditTriggersManager:
    """Manages PostgreSQL audit triggers for comprehensive change tracking."""

    def __init__(self):
        self.db_config = DatabaseConfig()
        self.excluded_tables = {
            'audit_log',  # Don't audit the audit table itself
            'alembic_version',  # Migration tracking
            'pg_stat_statements',  # PostgreSQL statistics
        }

    def create_audit_trigger_function(self):
        """Create the main audit trigger function that logs all changes."""

        info_id("Creating audit trigger function...")

        trigger_function_sql = """
        CREATE OR REPLACE FUNCTION audit_trigger_function()
        RETURNS TRIGGER AS $$
        DECLARE
            old_data JSON;
            new_data JSON;
            changed_fields TEXT[];
            field_name TEXT;
            audit_user TEXT;
            audit_session TEXT;
        BEGIN
            -- Get current user and session info
            audit_user := COALESCE(current_setting('audit.user_id', true), 'system');
            audit_session := COALESCE(current_setting('audit.session_id', true), 'unknown');

            -- Handle different operations
            IF TG_OP = 'DELETE' THEN
                old_data := row_to_json(OLD);
                new_data := NULL;

                INSERT INTO audit_log (
                    table_name, record_id, operation, timestamp,
                    user_id, session_id, old_values, new_values,
                    changed_fields, application
                ) VALUES (
                    TG_TABLE_NAME,
                    COALESCE(OLD.id::text, 'unknown'),
                    TG_OP,
                    NOW(),
                    audit_user,
                    audit_session,
                    old_data::text,
                    NULL,
                    NULL,
                    'EMTAC-TRIGGER'
                );

                RETURN OLD;

            ELSIF TG_OP = 'INSERT' THEN
                old_data := NULL;
                new_data := row_to_json(NEW);

                INSERT INTO audit_log (
                    table_name, record_id, operation, timestamp,
                    user_id, session_id, old_values, new_values,
                    changed_fields, application
                ) VALUES (
                    TG_TABLE_NAME,
                    COALESCE(NEW.id::text, 'unknown'),
                    TG_OP,
                    NOW(),
                    audit_user,
                    audit_session,
                    NULL,
                    new_data::text,
                    NULL,
                    'EMTAC-TRIGGER'
                );

                RETURN NEW;

            ELSIF TG_OP = 'UPDATE' THEN
                old_data := row_to_json(OLD);
                new_data := row_to_json(NEW);

                -- Find changed fields
                changed_fields := ARRAY[]::TEXT[];

                -- Compare old and new data to find changes
                FOR field_name IN 
                    SELECT key FROM json_each_text(new_data)
                LOOP
                    IF old_data->>field_name IS DISTINCT FROM new_data->>field_name THEN
                        changed_fields := array_append(changed_fields, field_name);
                    END IF;
                END LOOP;

                -- Only log if there are actual changes
                IF array_length(changed_fields, 1) > 0 THEN
                    INSERT INTO audit_log (
                        table_name, record_id, operation, timestamp,
                        user_id, session_id, old_values, new_values,
                        changed_fields, application
                    ) VALUES (
                        TG_TABLE_NAME,
                        COALESCE(NEW.id::text, 'unknown'),
                        TG_OP,
                        NOW(),
                        audit_user,
                        audit_session,
                        old_data::text,
                        new_data::text,
                        array_to_string(changed_fields, ','),
                        'EMTAC-TRIGGER'
                    );
                END IF;

                RETURN NEW;
            END IF;

            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """

        try:
            with self.db_config.main_session() as session:
                session.execute(text(trigger_function_sql))
                session.commit()
                info_id("✅ Audit trigger function created successfully")
                return True
        except Exception as e:
            error_id(f"❌ Failed to create audit trigger function: {e}")
            return False

    def get_all_tables(self):
        """Get all tables in the database excluding system tables."""

        try:
            inspector = inspect(self.db_config.main_engine)
            all_tables = inspector.get_table_names()

            # Filter out excluded tables
            tables_to_audit = [
                table for table in all_tables
                if table not in self.excluded_tables
            ]

            info_id(f"Found {len(tables_to_audit)} tables to audit: {', '.join(tables_to_audit)}")
            return tables_to_audit

        except Exception as e:
            error_id(f"❌ Failed to get table list: {e}")
            return []

    def create_trigger_for_table(self, table_name):
        """Create audit trigger for a specific table."""

        trigger_sql = f"""
        CREATE TRIGGER audit_trigger_{table_name}
        AFTER INSERT OR UPDATE OR DELETE ON {table_name}
        FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
        """

        try:
            with self.db_config.main_session() as session:
                session.execute(text(trigger_sql))
                session.commit()
                info_id(f"✅ Created audit trigger for table: {table_name}")
                return True
        except Exception as e:
            if "already exists" in str(e).lower():
                info_id(f"⚠️ Trigger for {table_name} already exists")
                return True
            else:
                error_id(f"❌ Failed to create trigger for {table_name}: {e}")
                return False

    def drop_trigger_for_table(self, table_name):
        """Drop audit trigger for a specific table."""

        trigger_sql = f"DROP TRIGGER IF EXISTS audit_trigger_{table_name} ON {table_name};"

        try:
            with self.db_config.main_session() as session:
                session.execute(text(trigger_sql))
                session.commit()
                info_id(f"✅ Dropped audit trigger for table: {table_name}")
                return True
        except Exception as e:
            error_id(f"❌ Failed to drop trigger for {table_name}: {e}")
            return False

    def setup_all_triggers(self):
        """Set up audit triggers for all tables."""

        info_id("🚀 Setting up audit triggers for all tables...")

        # Step 1: Create the trigger function
        if not self.create_audit_trigger_function():
            return False

        # Step 2: Get all tables
        tables = self.get_all_tables()
        if not tables:
            warning_id("⚠️ No tables found to audit")
            return False

        # Step 3: Create triggers for each table
        success_count = 0
        for table in tables:
            if self.create_trigger_for_table(table):
                success_count += 1

        info_id(f"✅ Successfully created triggers for {success_count}/{len(tables)} tables")
        return success_count > 0

    def remove_all_triggers(self):
        """Remove all audit triggers."""

        info_id("🗑️ Removing all audit triggers...")

        tables = self.get_all_tables()
        success_count = 0

        for table in tables:
            if self.drop_trigger_for_table(table):
                success_count += 1

        info_id(f"✅ Successfully removed triggers from {success_count}/{len(tables)} tables")
        return success_count > 0

    def list_existing_triggers(self):
        """List all existing audit triggers."""

        info_id("📋 Listing existing audit triggers...")

        query = """
        SELECT 
            schemaname,
            tablename,
            triggername
        FROM pg_triggers 
        WHERE triggername LIKE 'audit_trigger_%'
        ORDER BY tablename;
        """

        try:
            with self.db_config.main_session() as session:
                result = session.execute(text(query))
                triggers = result.fetchall()

                if triggers:
                    info_id(f"Found {len(triggers)} audit triggers:")
                    for trigger in triggers:
                        info_id(f"  • {trigger.tablename} → {trigger.triggername}")
                else:
                    info_id("No audit triggers found")

                return triggers

        except Exception as e:
            error_id(f"❌ Failed to list triggers: {e}")
            return []

    def test_audit_system(self):
        """Test the audit system with sample data."""

        info_id("🧪 Testing audit system...")

        test_sql = """
        -- Create test table
        CREATE TABLE IF NOT EXISTS audit_test (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            value INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Clear any existing test data
        DELETE FROM audit_test;
        DELETE FROM audit_log WHERE table_name = 'audit_test';

        -- Set audit context
        SELECT set_config('audit.user_id', 'test_user', false);
        SELECT set_config('audit.session_id', 'test_session_123', false);

        -- Test INSERT
        INSERT INTO audit_test (name, value) VALUES ('Test Record 1', 100);
        INSERT INTO audit_test (name, value) VALUES ('Test Record 2', 200);

        -- Test UPDATE
        UPDATE audit_test SET name = 'Updated Record', value = 150 WHERE id = 1;

        -- Test DELETE
        DELETE FROM audit_test WHERE id = 2;
        """

        try:
            with self.db_config.main_session() as session:
                session.execute(text(test_sql))
                session.commit()

                # Check audit results
                result = session.execute(text("""
                    SELECT 
                        table_name, 
                        operation, 
                        user_id, 
                        session_id,
                        changed_fields,
                        timestamp
                    FROM audit_log 
                    WHERE table_name = 'audit_test' 
                    ORDER BY timestamp DESC
                """))

                audit_entries = result.fetchall()

                if audit_entries:
                    info_id(f"✅ Audit system test successful! Found {len(audit_entries)} audit entries:")
                    for entry in audit_entries:
                        info_id(f"  • {entry.operation} on {entry.table_name} by {entry.user_id} at {entry.timestamp}")
                        if entry.changed_fields:
                            info_id(f"    Changed fields: {entry.changed_fields}")
                else:
                    warning_id("⚠️ No audit entries found - triggers may not be working")

                # Clean up test data
                session.execute(text("DROP TABLE IF EXISTS audit_test"))
                session.execute(text("DELETE FROM audit_log WHERE table_name = 'audit_test'"))
                session.commit()

                return len(audit_entries) > 0

        except Exception as e:
            error_id(f"❌ Audit system test failed: {e}")
            return False

    def show_audit_statistics(self):
        """Show audit log statistics."""

        info_id("📊 Audit Log Statistics:")

        stats_sql = """
        SELECT 
            table_name,
            operation,
            COUNT(*) as count
        FROM audit_log 
        GROUP BY table_name, operation
        ORDER BY table_name, operation;
        """

        try:
            with self.db_config.main_session() as session:
                result = session.execute(text(stats_sql))
                stats = result.fetchall()

                if stats:
                    current_table = None
                    for stat in stats:
                        if stat.table_name != current_table:
                            info_id(f"\n📋 Table: {stat.table_name}")
                            current_table = stat.table_name
                        info_id(f"  • {stat.operation}: {stat.count} entries")
                else:
                    info_id("No audit entries found")

        except Exception as e:
            error_id(f"❌ Failed to get audit statistics: {e}")


def main():
    """Main function with interactive menu."""

    print("🔍 PostgreSQL Audit Triggers Manager")
    print("=" * 50)

    try:
        manager = AuditTriggersManager()

        while True:
            print(f"\n📋 Audit Triggers Management Menu")
            print("1. 🚀 Setup all audit triggers")
            print("2. 📋 List existing triggers")
            print("3. 🧪 Test audit system")
            print("4. 📊 Show audit statistics")
            print("5. 🗑️ Remove all triggers")
            print("6. ❓ Help")
            print("7. 🚪 Exit")

            choice = input("\nChoose an option [1-7]: ").strip()

            if choice == '1':
                print("\n🚀 Setting up audit triggers...")
                if manager.setup_all_triggers():
                    print("✅ Audit triggers setup completed!")
                    print("💡 All database changes will now be automatically logged.")
                else:
                    print("❌ Audit triggers setup failed!")

            elif choice == '2':
                manager.list_existing_triggers()

            elif choice == '3':
                print("\n🧪 Testing audit system...")
                if manager.test_audit_system():
                    print("✅ Audit system is working correctly!")
                else:
                    print("❌ Audit system test failed!")

            elif choice == '4':
                manager.show_audit_statistics()

            elif choice == '5':
                confirm = input("\n⚠️ Are you sure you want to remove all audit triggers? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    if manager.remove_all_triggers():
                        print("✅ All audit triggers removed!")
                    else:
                        print("❌ Failed to remove some triggers!")
                else:
                    print("❌ Operation cancelled")

            elif choice == '6':
                print("\n❓ Audit Triggers Help")
                print("=" * 30)
                print("🔍 What this script does:")
                print("  • Creates a universal audit trigger function")
                print("  • Applies triggers to all your database tables")
                print("  • Automatically logs INSERT, UPDATE, DELETE operations")
                print("  • Captures before/after values and changed fields")
                print("  • Records user ID, session ID, and timestamps")
                print("")
                print("💡 Usage Tips:")
                print("  • Run option 1 to setup triggers on all tables")
                print("  • Use option 3 to test that auditing works")
                print("  • Check option 4 to see audit activity")
                print("  • Triggers are persistent - they survive database restarts")
                print("")
                print("🎯 Audit Context:")
                print("  • Set user context: SELECT set_config('audit.user_id', 'username', false);")
                print("  • Set session context: SELECT set_config('audit.session_id', 'session123', false);")

            elif choice == '7':
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid option. Please choose 1-7.")

            input("\n📋 Press Enter to continue...")

    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user.")
    except Exception as e:
        error_id(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()