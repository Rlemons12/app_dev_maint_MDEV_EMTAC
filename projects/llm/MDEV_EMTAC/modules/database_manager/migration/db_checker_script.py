#!/usr/bin/env python3
# check_database_tables.py
"""
Standalone script to check the current state of your database tables after migration
"""

import sys
import os

# Add the project root to Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger, with_request_id, info_id, error_id
from sqlalchemy import text, inspect
import json


@with_request_id
def check_database_tables(request_id=None):
    """
    Check current database table structure
    """
    info_id("Checking database table structure...", request_id)

    try:
        db_config = DatabaseConfig()
        with db_config.main_session() as session:
            # Get all tables in the public schema
            inspector = inspect(session.bind)
            tables = inspector.get_table_names(schema='public')

            info_id(f"Found {len(tables)} tables in database", request_id)

            # Check for embedding-related tables
            embedding_tables = [t for t in tables if 'embedding' in t.lower()]

            print("\n" + "=" * 60)
            print("DATABASE TABLE ANALYSIS")
            print("=" * 60)

            print(f"\nAll tables ({len(tables)}):")
            for table in sorted(tables):
                print(f"  - {table}")

            print(f"\nEmbedding-related tables ({len(embedding_tables)}):")
            for table in embedding_tables:
                print(f"  - {table}")

                # Get column details for embedding tables
                columns = inspector.get_columns(table, schema='public')
                print(f"    Columns:")
                for col in columns:
                    col_type = str(col['type'])
                    nullable = "NULL" if col['nullable'] else "NOT NULL"
                    print(f"      - {col['name']}: {col_type} {nullable}")

                # Get row count
                try:
                    count_result = session.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                    row_count = count_result[0]
                    print(f"    Rows: {row_count}")
                except Exception as e:
                    print(f"    Rows: Error getting count - {e}")

                print()

            return {
                'all_tables': tables,
                'embedding_tables': embedding_tables,
                'success': True
            }

    except Exception as e:
        error_id(f"Error checking database tables: {e}", request_id)
        print(f"ERROR: {e}")
        return {
            'all_tables': [],
            'embedding_tables': [],
            'success': False,
            'error': str(e)
        }


@with_request_id
def check_document_embedding_structure(request_id=None):
    """
    Specifically check the document_embedding table structure
    """
    info_id("Checking document_embedding table structure...", request_id)

    try:
        db_config = DatabaseConfig()
        with db_config.main_session() as session:

            # Check if table exists
            result = session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'document_embedding'
                )
            """)).fetchone()

            table_exists = result[0]

            if not table_exists:
                error_id("document_embedding table does not exist!", request_id)
                print(" document_embedding table does not exist!")
                return {'exists': False}

            # Get table structure
            columns = session.execute(text("""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'document_embedding'
                ORDER BY ordinal_position
            """)).fetchall()

            # Get row count
            count_result = session.execute(text("SELECT COUNT(*) FROM document_embedding")).fetchone()
            row_count = count_result[0]

            # Get sample data
            sample_data = session.execute(text("""
                SELECT id, document_id, model_name, 
                       CASE WHEN model_embedding IS NOT NULL THEN 'HAS_DATA' ELSE 'NULL' END as model_embedding_status,
                       CASE WHEN embedding_vector IS NOT NULL THEN 'HAS_DATA' ELSE 'NULL' END as embedding_vector_status,
                       created_at, updated_at
                FROM document_embedding 
                ORDER BY id 
                LIMIT 5
            """)).fetchall()

            # Check for new columns from migration
            has_actual_dimensions = any(col.column_name == 'actual_dimensions' for col in columns)
            has_embedding_metadata = any(col.column_name == 'embedding_metadata' for col in columns)

            print("\n" + "=" * 60)
            print("DOCUMENT_EMBEDDING TABLE ANALYSIS")
            print("=" * 60)

            print(f"\nTable exists: {table_exists}")
            print(f"Total rows: {row_count}")
            print(f"Has actual_dimensions column: {has_actual_dimensions}")
            print(f"Has embedding_metadata column: {has_embedding_metadata}")

            print(f"\nColumn structure:")
            for col in columns:
                nullable = "NULL" if col.is_nullable == 'YES' else "NOT NULL"
                default = f" DEFAULT {col.column_default}" if col.column_default else ""
                print(f"  - {col.column_name}: {col.data_type} {nullable}{default}")

            print(f"\nSample data (first 5 rows):")
            for row in sample_data:
                print(f"  ID {row.id}: doc_id={row.document_id}, model={row.model_name}")
                print(f"    model_embedding: {row.model_embedding_status}")
                print(f"    embedding_vector: {row.embedding_vector_status}")
                print(f"    created: {row.created_at}")
                print()

            return {
                'exists': True,
                'columns': [dict(col._mapping) for col in columns],
                'row_count': row_count,
                'sample_data': [dict(row._mapping) for row in sample_data],
                'has_migration_columns': has_actual_dimensions and has_embedding_metadata,
                'success': True
            }

    except Exception as e:
        error_id(f"Error checking document_embedding table: {e}", request_id)
        print(f"ERROR checking document_embedding: {e}")
        return {
            'exists': False,
            'success': False,
            'error': str(e)
        }


@with_request_id
def check_backup_tables(request_id=None):
    """
    Check if backup tables exist from migration
    """
    info_id("Checking for backup tables...", request_id)

    try:
        db_config = DatabaseConfig()
        with db_config.main_session() as session:

            # Check for backup table
            backup_exists = session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'document_embedding_backup'
                )
            """)).fetchone()[0]

            # Check for any other embedding tables
            embedding_table_query = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%embedding%'
                ORDER BY table_name
            """)).fetchall()

            embedding_tables = [row.table_name for row in embedding_table_query]

            print("\n" + "=" * 60)
            print("BACKUP TABLES ANALYSIS")
            print("=" * 60)

            print(f"\ndocument_embedding_backup exists: {backup_exists}")

            if backup_exists:
                # Get backup table row count
                backup_count = session.execute(text("SELECT COUNT(*) FROM document_embedding_backup")).fetchone()[0]
                print(f"Backup table rows: {backup_count}")

            print(f"\nAll embedding-related tables:")
            for table in embedding_tables:
                try:
                    count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
                    print(f"  - {table}: {count} rows")
                except Exception as e:
                    print(f"  - {table}: Error getting count - {e}")

            return {
                'backup_exists': backup_exists,
                'all_embedding_tables': embedding_tables,
                'success': True
            }

    except Exception as e:
        error_id(f"Error checking backup tables: {e}", request_id)
        print(f"ERROR checking backup tables: {e}")
        return {
            'backup_exists': False,
            'success': False,
            'error': str(e)
        }


@with_request_id
def check_migration_status(request_id=None):
    """
    Check if migration completed successfully
    """
    try:
        db_config = DatabaseConfig()
        with db_config.main_session() as session:

            # Check dimension distribution
            dimension_query = """
            SELECT 
                CASE 
                    WHEN actual_dimensions IS NOT NULL THEN actual_dimensions
                    WHEN embedding_vector IS NOT NULL THEN 
                        array_length(string_to_array(trim(both '[]' from embedding_vector::text), ','), 1)
                    ELSE NULL
                END as dimensions,
                COUNT(*) as count,
                array_agg(DISTINCT model_name) as models
            FROM document_embedding 
            GROUP BY dimensions
            ORDER BY dimensions
            """

            results = session.execute(text(dimension_query)).fetchall()

            print("\n" + "=" * 60)
            print("MIGRATION STATUS ANALYSIS")
            print("=" * 60)

            print(f"\nEmbedding dimension distribution:")
            total_embeddings = 0
            for row in results:
                if row.dimensions:
                    print(f"  {row.dimensions}d: {row.count} embeddings from {row.models}")
                    total_embeddings += row.count
                else:
                    print(f"  Unknown dimensions: {row.count} embeddings")

            print(f"\nTotal embeddings: {total_embeddings}")

            # Check if migration columns exist
            has_migration_cols = session.execute(text("""
                SELECT 
                    COUNT(*) FILTER (WHERE column_name = 'actual_dimensions') as has_actual_dims,
                    COUNT(*) FILTER (WHERE column_name = 'embedding_metadata') as has_metadata
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'document_embedding'
            """)).fetchone()

            migration_complete = has_migration_cols.has_actual_dims > 0 and has_migration_cols.has_metadata > 0

            print(f"\nMigration status:")
            print(f"  Has actual_dimensions column: {has_migration_cols.has_actual_dims > 0}")
            print(f"  Has embedding_metadata column: {has_migration_cols.has_metadata > 0}")
            print(f"  Migration appears complete: {migration_complete}")

            return {
                'migration_complete': migration_complete,
                'total_embeddings': total_embeddings,
                'dimension_distribution': [dict(row._mapping) for row in results]
            }

    except Exception as e:
        print(f"ERROR checking migration status: {e}")
        return {'migration_complete': False, 'error': str(e)}


@with_request_id
def fix_navigator_issue(request_id=None):
    """
    Try to fix the navigator issue by testing the connection
    """
    info_id("Testing database connection...", request_id)

    try:
        db_config = DatabaseConfig()

        # Test basic connection
        connection_test = db_config.test_connection()

        print("\n" + "=" * 60)
        print("CONNECTION DIAGNOSTIC")
        print("=" * 60)

        print(f"\nConnection test result:")
        print(f"  Status: {connection_test.get('status', 'unknown')}")
        print(f"  Database type: {connection_test.get('database_type', 'unknown')}")
        print(f"  Version: {connection_test.get('version', 'unknown')}")
        print(f"  Current user: {connection_test.get('current_user', 'unknown')}")
        print(f"  URL: {connection_test.get('url', 'unknown')}")

        if connection_test.get('status') == 'error':
            print(f"  Error: {connection_test.get('error', 'unknown')}")

        # Get connection stats
        stats = db_config.get_connection_stats()
        print(f"\nConnection statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return connection_test

    except Exception as e:
        error_id(f"Error in connection diagnostic: {e}", request_id)
        print(f"ERROR in connection test: {e}")
        return {'status': 'error', 'error': str(e)}


def main():
    print("DATABASE DIAGNOSTIC TOOL")
    print("=" * 60)

    # Run all checks
    print("\n1. Checking all database tables...")
    table_check = check_database_tables()

    print("\n2. Checking document_embedding table specifically...")
    embedding_check = check_document_embedding_structure()

    print("\n3. Checking for backup tables...")
    backup_check = check_backup_tables()

    print("\n4. Checking migration status...")
    migration_check = check_migration_status()

    print("\n5. Testing database connection...")
    connection_check = fix_navigator_issue()

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    if embedding_check.get('exists'):
        print(" document_embedding table exists and is accessible")
        print(f" Table has {embedding_check.get('row_count', 0)} rows")
    else:
        print(" document_embedding table not found or not accessible")

    if backup_check.get('backup_exists'):
        print(" Backup table exists (migration completed)")
    else:
        print("  No backup table found")

    if migration_check.get('migration_complete'):
        print(" Migration appears to have completed successfully")
    else:
        print("  Migration may not have completed properly")

    if connection_check.get('status') == 'success':
        print(" Database connection is working")
    else:
        print(" Database connection issues detected")

    print("\n" + "=" * 60)
    print("DBEAVER RECOMMENDATIONS")
    print("=" * 60)

    if not embedding_check.get('exists'):
        print("🔧 Try refreshing your database navigator/connection")
        print("🔧 Check if the migration completed successfully")
        print("🔧 Verify your database connection settings")
    else:
        print(" Database structure looks good!")
        print("🔧 In DBeaver:")
        print("   - Right-click your PostgreSQL connection → Refresh (F5)")
        print("   - Or right-click connection → Edit Connection → Test Connection")
        print("   - Or right-click 'public' schema → Refresh")
        print("   - Or disconnect and reconnect your connection")

    if migration_check.get('migration_complete'):
        print(" Ready to proceed with TinyLlama embedding creation!")
    else:
        print("  Consider re-running migration or checking for issues")


if __name__ == "__main__":
    main()