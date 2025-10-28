#!/usr/bin/env python3
"""
Migration script to add timestamp columns to document_embedding table.
Run this script to safely update your database schema.
"""


def migrate_document_embedding_table():
    """Add timestamp columns and indexes to document_embedding table."""

    print("🔧 Starting document_embedding table migration...")

    try:
        from modules.configuration.config_env import DatabaseConfig
        from sqlalchemy import text
        from modules.configuration.log_config import info_id, error_id, warning_id

        db_config = DatabaseConfig()

        with db_config.get_main_session() as session:
            print("📊 Checking current table structure...")

            # Check if columns already exist
            check_columns_sql = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'document_embedding' 
                AND column_name IN ('created_at', 'updated_at')
            """)

            existing_columns = session.execute(check_columns_sql).fetchall()
            existing_column_names = [col[0] for col in existing_columns]

            print(f"   Found existing timestamp columns: {existing_column_names}")

            # Add created_at column if it doesn't exist
            if 'created_at' not in existing_column_names:
                print("➕ Adding created_at column...")
                session.execute(text("""
                    ALTER TABLE document_embedding 
                    ADD COLUMN created_at TIMESTAMP DEFAULT NOW()
                """))
                print("   ✅ created_at column added")
            else:
                print("   ✅ created_at column already exists")

            # Add updated_at column if it doesn't exist
            if 'updated_at' not in existing_column_names:
                print("➕ Adding updated_at column...")
                session.execute(text("""
                    ALTER TABLE document_embedding 
                    ADD COLUMN updated_at TIMESTAMP DEFAULT NOW()
                """))
                print("   ✅ updated_at column added")
            else:
                print("   ✅ updated_at column already exists")

            # Update existing rows that might have NULL timestamps
            print("🔄 Updating existing rows with timestamps...")
            result = session.execute(text("""
                UPDATE document_embedding 
                SET created_at = COALESCE(created_at, NOW()), 
                    updated_at = COALESCE(updated_at, NOW())
                WHERE created_at IS NULL OR updated_at IS NULL
            """))
            updated_rows = result.rowcount
            print(f"   📝 Updated {updated_rows} existing rows")

            # Add unique constraint (with error handling in case it already exists)
            print("🔒 Adding unique constraint...")
            try:
                session.execute(text("""
                    ALTER TABLE document_embedding 
                    ADD CONSTRAINT unique_document_model_embedding 
                    UNIQUE (document_id, model_name)
                """))
                print("   ✅ Unique constraint added")
            except Exception as e:
                if "already exists" in str(e) or "duplicate key" in str(e):
                    print("   ✅ Unique constraint already exists")
                else:
                    print(f"   ⚠️ Could not add unique constraint: {e}")

            # Create indexes
            print("📋 Creating indexes...")

            indexes = [
                ("idx_document_embedding_document_id", "document_id"),
                ("idx_document_embedding_model_name", "model_name"),
                ("idx_document_embedding_created_at", "created_at")
            ]

            for index_name, column_name in indexes:
                try:
                    session.execute(text(f"""
                        CREATE INDEX IF NOT EXISTS {index_name} 
                        ON document_embedding({column_name})
                    """))
                    print(f"   ✅ Index {index_name} created")
                except Exception as e:
                    print(f"   ⚠️ Could not create index {index_name}: {e}")

            # Create trigger function for automatic updated_at updates
            print("⚡ Creating update trigger...")
            try:
                session.execute(text("""
                    CREATE OR REPLACE FUNCTION update_document_embedding_updated_at()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = NOW();
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql
                """))

                # Drop existing trigger if it exists
                session.execute(text("""
                    DROP TRIGGER IF EXISTS document_embedding_updated_at_trigger ON document_embedding
                """))

                # Create the trigger
                session.execute(text("""
                    CREATE TRIGGER document_embedding_updated_at_trigger
                        BEFORE UPDATE ON document_embedding
                        FOR EACH ROW
                        EXECUTE FUNCTION update_document_embedding_updated_at()
                """))
                print("   ✅ Update trigger created")
            except Exception as e:
                print(f"   ⚠️ Could not create trigger: {e}")

            # Commit all changes
            session.commit()
            print("💾 All changes committed successfully")

            # Get final statistics
            print("📊 Final table statistics...")
            stats_result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_embeddings,
                    COUNT(DISTINCT model_name) as unique_models,
                    MIN(created_at) as oldest_embedding,
                    MAX(created_at) as newest_embedding
                FROM document_embedding
            """)).fetchone()

            if stats_result:
                print(f"   📈 Total embeddings: {stats_result[0]}")
                print(f"   🤖 Unique models: {stats_result[1]}")
                print(f"   📅 Date range: {stats_result[2]} to {stats_result[3]}")

            print("\n🎉 Migration completed successfully!")
            return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_migration():
    """Verify that the migration was successful."""

    print("\n🔍 Verifying migration...")

    try:
        from modules.configuration.config_env import DatabaseConfig
        from sqlalchemy import text

        db_config = DatabaseConfig()

        with db_config.get_main_session() as session:
            # Check table structure
            structure_sql = text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'document_embedding'
                ORDER BY ordinal_position
            """)

            columns = session.execute(structure_sql).fetchall()

            print("📋 Current table structure:")
            for col in columns:
                print(f"   • {col[0]} ({col[1]}) - {'NULL' if col[2] == 'YES' else 'NOT NULL'}")

            # Check indexes
            index_sql = text("""
                SELECT indexname, indexdef
                FROM pg_indexes 
                WHERE tablename = 'document_embedding'
                ORDER BY indexname
            """)

            indexes = session.execute(index_sql).fetchall()

            print("📋 Current indexes:")
            for idx in indexes:
                print(f"   • {idx[0]}")

            # Check constraints
            constraint_sql = text("""
                SELECT constraint_name, constraint_type
                FROM information_schema.table_constraints
                WHERE table_name = 'document_embedding'
                ORDER BY constraint_name
            """)

            constraints = session.execute(constraint_sql).fetchall()

            print("📋 Current constraints:")
            for const in constraints:
                print(f"   • {const[0]} ({const[1]})")

            print("\n✅ Migration verification completed!")
            return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Document Embedding Table Migration")
    print("=" * 50)

    # Run migration
    success = migrate_document_embedding_table()

    if success:
        # Verify migration
        verify_migration()

        print("\n" + "=" * 50)
        print("✅ Migration completed successfully!")
        print("You can now use the enhanced DocumentEmbedding class.")
    else:
        print("\n" + "=" * 50)
        print("❌ Migration failed!")
        print("Please check the error messages above and try again.")