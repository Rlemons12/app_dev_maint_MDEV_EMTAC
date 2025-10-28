# safe_embedding_migration.py
"""
Safe migration strategy for DocumentEmbedding table:
1. Create new_document_embedding table with enhanced structure
2. Migrate all data from document_embedding to new_document_embedding
3. Rename tables: document_embedding → document_embedding_backup, new_document_embedding → document_embedding
4. Update foreign key references
5. Recreate indexes and constraints

Updated for your PostgreSQL configuration and DatabaseConfig setup.
"""

from sqlalchemy import text, inspect
from datetime import datetime
import logging
import json
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger, with_request_id, info_id, error_id, warning_id

# Use your existing logger
migration_logger = logging.getLogger(__name__)


class EmbeddingTableMigration:
    """
    Handles the complete migration process safely using your DatabaseConfig
    """

    def __init__(self):
        self.migration_log = []
        self.db_config = DatabaseConfig()

    def log_step(self, message: str, success: bool = True, request_id: str = None):
        """Log migration steps using your logging system"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'success': success
        }
        self.migration_log.append(log_entry)

        if success:
            info_id(f" {message}", request_id)
        else:
            error_id(f"✗ {message}", request_id)

    @with_request_id
    def step_1_create_new_table(self, request_id=None):
        """
        Step 1: Create new_document_embedding table with enhanced structure
        """
        try:
            with self.db_config.main_session() as session:
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS new_document_embedding (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    model_name VARCHAR NOT NULL,

                    -- Legacy storage (for backward compatibility)
                    model_embedding BYTEA,

                    -- Modern flexible pgvector storage (no dimension constraint)
                    embedding_vector vector,

                    -- New enhanced columns
                    actual_dimensions INTEGER,
                    embedding_metadata JSONB,

                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),

                    -- Add foreign key constraint (will be recreated later)
                    CONSTRAINT fk_new_doc_embedding_document 
                        FOREIGN KEY (document_id) REFERENCES document(id) ON DELETE CASCADE
                );
                """

                session.execute(text(create_table_sql))
                session.commit()
                self.log_step("Created new_document_embedding table with enhanced structure", True, request_id)
                return True

        except Exception as e:
            self.log_step(f"Failed to create new table: {e}", False, request_id)
            return False

    @with_request_id
    def step_2_migrate_data(self, batch_size: int = 1000, request_id=None):
        """
        Step 2: Migrate all data from document_embedding to new_document_embedding
        """
        try:
            with self.db_config.main_session() as session:
                # First, check if old table exists
                inspector = inspect(session.bind)
                tables = inspector.get_table_names()

                if 'document_embedding' not in tables:
                    self.log_step("Original document_embedding table not found - creating empty new table", True,
                                  request_id)
                    return True

                # Get total count for progress tracking
                count_result = session.execute(text("SELECT COUNT(*) FROM document_embedding")).fetchone()
                total_records = count_result[0]

                if total_records == 0:
                    self.log_step("No records to migrate", True, request_id)
                    return True

                self.log_step(f"Starting migration of {total_records} records", True, request_id)

                migrated_count = 0
                offset = 0

                while offset < total_records:
                    # Get batch of records
                    batch_sql = f"""
                    SELECT id, document_id, model_name, model_embedding, embedding_vector, 
                           created_at, updated_at
                    FROM document_embedding 
                    ORDER BY id 
                    LIMIT {batch_size} OFFSET {offset}
                    """

                    batch_records = session.execute(text(batch_sql)).fetchall()

                    if not batch_records:
                        break

                    # Process each record in the batch
                    for record in batch_records:
                        try:
                            # Extract embedding data and calculate dimensions
                            embedding_list = self._extract_embedding_from_record(record)
                            actual_dimensions = len(embedding_list) if embedding_list else None

                            # Create metadata
                            metadata = self._create_metadata_for_record(record, embedding_list)

                            # Insert into new table
                            insert_sql = """
                            INSERT INTO new_document_embedding 
                            (id, document_id, model_name, model_embedding, embedding_vector, 
                             actual_dimensions, embedding_metadata, created_at, updated_at)
                            VALUES (:id, :document_id, :model_name, :model_embedding, :embedding_vector,
                                    :actual_dimensions, :embedding_metadata, :created_at, :updated_at)
                            """

                            session.execute(text(insert_sql), {
                                'id': record.id,
                                'document_id': record.document_id,
                                'model_name': record.model_name,
                                'model_embedding': record.model_embedding,
                                'embedding_vector': embedding_list if embedding_list else None,
                                'actual_dimensions': actual_dimensions,
                                'embedding_metadata': json.dumps(metadata) if metadata else None,
                                'created_at': record.created_at,
                                'updated_at': record.updated_at or datetime.utcnow()
                            })

                            migrated_count += 1

                        except Exception as e:
                            self.log_step(f"Error migrating record {record.id}: {e}", False, request_id)
                            continue

                    # Commit batch
                    session.commit()
                    offset += batch_size

                    # Progress update
                    progress = min(100, (offset / total_records) * 100)
                    self.log_step(f"Migration progress: {progress:.1f}% ({migrated_count}/{total_records})", True,
                                  request_id)

                # Reset sequence to match the highest ID
                session.execute(text("""
                    SELECT setval('new_document_embedding_id_seq', 
                                  (SELECT COALESCE(MAX(id), 1) FROM new_document_embedding))
                """))
                session.commit()

                self.log_step(f"Successfully migrated {migrated_count} records", True, request_id)
                return True

        except Exception as e:
            self.log_step(f"Data migration failed: {e}", False, request_id)
            return False

    def _extract_embedding_from_record(self, record) -> list:
        """Extract embedding as list from a record"""
        try:
            # Try pgvector first
            if record.embedding_vector is not None:
                vector_str = str(record.embedding_vector)
                if vector_str.startswith('[') and vector_str.endswith(']'):
                    vector_str = vector_str[1:-1]
                return [float(x.strip()) for x in vector_str.split(',') if x.strip()]

            # Try legacy format
            if record.model_embedding is not None:
                try:
                    # Try numpy array
                    import numpy as np
                    return np.frombuffer(record.model_embedding, dtype=np.float32).tolist()
                except:
                    try:
                        # Try JSON
                        return json.loads(record.model_embedding.decode('utf-8'))
                    except:
                        pass

            return []

        except Exception as e:
            migration_logger.warning(f"Could not extract embedding from record {record.id}: {e}")
            return []

    def _create_metadata_for_record(self, record, embedding_list: list) -> dict:
        """Create metadata for a migrated record"""
        if not embedding_list:
            return {}

        dimensions = len(embedding_list)

        # Infer model type from name and dimensions
        model_type = self._infer_model_type(record.model_name, dimensions)

        return {
            'dimensions': dimensions,
            'model_type': model_type,
            'migration_timestamp': datetime.utcnow().isoformat(),
            'migrated_from': 'document_embedding',
            'original_model_name': record.model_name,
            'storage_method': 'migrated_enhanced'
        }

    def _infer_model_type(self, model_name: str, dimensions: int) -> str:
        """Infer model type from name and dimensions"""
        model_name_lower = model_name.lower() if model_name else ''

        # Check name patterns first
        if 'openai' in model_name_lower or 'text-embedding' in model_name_lower:
            return f'openai_{dimensions}d'
        elif 'tinyllama' in model_name_lower:
            return f'tinyllama_{dimensions}d'
        elif any(x in model_name_lower for x in ['sentence', 'transformer', 'mini', 'mpnet']):
            return f'sentence_transformer_{dimensions}d'

        # Fall back to dimension-based inference
        dimension_mapping = {
            384: 'sentence_transformers_mini',
            768: 'sentence_transformers_base',
            1024: 'sentence_transformers_large',
            1536: 'openai_text_embedding_3_small',
            3072: 'openai_text_embedding_3_large',
            512: 'clip_base',
        }

        return dimension_mapping.get(dimensions, f'unknown_{dimensions}d')

    @with_request_id
    def step_3_rename_tables(self, request_id=None):
        """
        Step 3: Rename tables safely
        """
        try:
            with self.db_config.main_session() as session:
                # Check if old table exists before trying to rename it
                inspector = inspect(session.bind)
                tables = inspector.get_table_names()

                rename_statements = []

                if 'document_embedding' in tables:
                    # Rename old table to backup
                    rename_statements.append(
                        "ALTER TABLE document_embedding RENAME TO document_embedding_backup"
                    )

                # Rename new table to main name
                rename_statements.append(
                    "ALTER TABLE new_document_embedding RENAME TO document_embedding"
                )

                for stmt in rename_statements:
                    session.execute(text(stmt))
                    session.commit()

                self.log_step("Successfully renamed tables", True, request_id)
                return True

        except Exception as e:
            self.log_step(f"Table renaming failed: {e}", False, request_id)
            return False

    @with_request_id
    def step_4_recreate_indexes(self, request_id=None):
        """
        Step 4: Create optimized indexes for the new table structure
        """
        try:
            with self.db_config.main_session() as session:
                index_statements = [
                    # Basic indexes
                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_document_id 
                    ON document_embedding (document_id);
                    """,

                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_model_name 
                    ON document_embedding (model_name);
                    """,

                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_dimensions 
                    ON document_embedding (actual_dimensions);
                    """,

                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_created 
                    ON document_embedding (created_at);
                    """,

                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_metadata 
                    ON document_embedding USING gin (embedding_metadata);
                    """,

                    # Dimension-specific HNSW indexes for vector similarity
                    # Common TinyLlama/SentenceTransformer dimensions
                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_vector_cosine_384d 
                    ON document_embedding 
                    USING hnsw (embedding_vector vector_cosine_ops)
                    WHERE actual_dimensions = 384
                    WITH (m = 16, ef_construction = 64);
                    """,

                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_vector_cosine_768d 
                    ON document_embedding 
                    USING hnsw (embedding_vector vector_cosine_ops)
                    WHERE actual_dimensions = 768
                    WITH (m = 16, ef_construction = 64);
                    """,

                    # OpenAI dimensions
                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_vector_cosine_1536d 
                    ON document_embedding 
                    USING hnsw (embedding_vector vector_cosine_ops)
                    WHERE actual_dimensions = 1536
                    WITH (m = 16, ef_construction = 64);
                    """,

                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_vector_cosine_3072d 
                    ON document_embedding 
                    USING hnsw (embedding_vector vector_cosine_ops)
                    WHERE actual_dimensions = 3072
                    WITH (m = 16, ef_construction = 64);
                    """,

                    # L2 distance indexes for common dimensions
                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_vector_l2_384d 
                    ON document_embedding 
                    USING hnsw (embedding_vector vector_l2_ops)
                    WHERE actual_dimensions = 384
                    WITH (m = 16, ef_construction = 64);
                    """,

                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_vector_l2_768d 
                    ON document_embedding 
                    USING hnsw (embedding_vector vector_l2_ops)
                    WHERE actual_dimensions = 768
                    WITH (m = 16, ef_construction = 64);
                    """,

                    """
                    CREATE INDEX IF NOT EXISTS idx_document_embedding_vector_l2_1536d 
                    ON document_embedding 
                    USING hnsw (embedding_vector vector_l2_ops)
                    WHERE actual_dimensions = 1536
                    WITH (m = 16, ef_construction = 64);
                    """,
                ]

                successful_indexes = 0
                for stmt in index_statements:
                    try:
                        session.execute(text(stmt))
                        session.commit()
                        successful_indexes += 1
                    except Exception as e:
                        # Index creation might fail if pgvector not available or already exists
                        warning_id(f"Index creation warning: {e}", request_id)
                        session.rollback()
                        continue

                self.log_step(f"Created {successful_indexes} indexes successfully", True, request_id)
                return True

        except Exception as e:
            self.log_step(f"Index creation failed: {e}", False, request_id)
            return False

    @with_request_id
    def step_5_update_foreign_keys(self, request_id=None):
        """
        Step 5: Ensure foreign key constraints are properly set up
        """
        try:
            with self.db_config.main_session() as session:
                # The foreign key should already be created, but let's ensure it exists
                fk_statements = [
                    """
                    ALTER TABLE document_embedding 
                    DROP CONSTRAINT IF EXISTS fk_doc_embedding_document;
                    """,

                    """
                    ALTER TABLE document_embedding 
                    ADD CONSTRAINT fk_doc_embedding_document 
                        FOREIGN KEY (document_id) REFERENCES document(id) ON DELETE CASCADE;
                    """
                ]

                for stmt in fk_statements:
                    session.execute(text(stmt))
                    session.commit()

                self.log_step("Foreign key constraints updated", True, request_id)
                return True

        except Exception as e:
            self.log_step(f"Foreign key update failed: {e}", False, request_id)
            return False

    @with_request_id
    def step_6_verify_migration(self, request_id=None):
        """
        Step 6: Verify the migration was successful
        """
        try:
            with self.db_config.main_session() as session:
                # Count records in new table
                new_count = session.execute(text("SELECT COUNT(*) FROM document_embedding")).fetchone()[0]

                # Check if backup table exists and count it
                inspector = inspect(session.bind)
                tables = inspector.get_table_names()

                if 'document_embedding_backup' in tables:
                    old_count = session.execute(text("SELECT COUNT(*) FROM document_embedding_backup")).fetchone()[0]

                    if new_count == old_count:
                        self.log_step(f"Migration verified: {new_count} records in both tables", True, request_id)
                    else:
                        self.log_step(f"Migration count mismatch: new={new_count}, old={old_count}", False, request_id)
                        return False
                else:
                    self.log_step(f"Migration verified: {new_count} records in new table (no backup table)", True,
                                  request_id)

                # Test that we can query the new structure
                sample_query = """
                SELECT id, document_id, model_name, actual_dimensions, embedding_metadata
                FROM document_embedding 
                LIMIT 1
                """

                result = session.execute(text(sample_query)).fetchone()
                if result:
                    self.log_step("New table structure verified with sample query", True, request_id)
                else:
                    self.log_step("No records found in new table structure", True, request_id)

                return True

        except Exception as e:
            self.log_step(f"Migration verification failed: {e}", False, request_id)
            return False

    @with_request_id
    def run_complete_migration(self, batch_size: int = 1000, request_id=None):
        """
        Run the complete migration process
        """
        self.log_step("Starting complete embedding table migration", True, request_id)

        steps = [
            ("Creating new table", self.step_1_create_new_table),
            ("Migrating data", lambda: self.step_2_migrate_data(batch_size)),
            ("Renaming tables", self.step_3_rename_tables),
            ("Creating indexes", self.step_4_recreate_indexes),
            ("Updating foreign keys", self.step_5_update_foreign_keys),
            ("Verifying migration", self.step_6_verify_migration),
        ]

        for step_name, step_func in steps:
            self.log_step(f"Starting: {step_name}", True, request_id)

            if not step_func():
                self.log_step(f"Migration failed at step: {step_name}", False, request_id)
                return False

            self.log_step(f"Completed: {step_name}", True, request_id)

        self.log_step("Complete migration finished successfully!", True, request_id)
        return True

    @with_request_id
    def rollback_migration(self, request_id=None):
        """
        Rollback migration if something goes wrong
        """
        try:
            with self.db_config.main_session() as session:
                inspector = inspect(session.bind)
                tables = inspector.get_table_names()

                if 'document_embedding_backup' in tables and 'document_embedding' in tables:
                    # Drop the new table and restore the backup
                    session.execute(text("DROP TABLE IF EXISTS document_embedding CASCADE"))
                    session.execute(text("ALTER TABLE document_embedding_backup RENAME TO document_embedding"))
                    session.commit()
                    self.log_step("Migration rolled back successfully", True, request_id)
                    return True
                else:
                    self.log_step("Cannot rollback - backup table not found", False, request_id)
                    return False

        except Exception as e:
            self.log_step(f"Rollback failed: {e}", False, request_id)
            return False

    @with_request_id
    def cleanup_backup_table(self, request_id=None):
        """
        Clean up backup table after successful migration (optional)
        """
        try:
            with self.db_config.main_session() as session:
                inspector = inspect(session.bind)
                tables = inspector.get_table_names()

                if 'document_embedding_backup' in tables:
                    session.execute(text("DROP TABLE document_embedding_backup CASCADE"))
                    session.commit()
                    self.log_step("Backup table cleaned up", True, request_id)
                    return True
                else:
                    self.log_step("No backup table to clean up", True, request_id)
                    return True

        except Exception as e:
            self.log_step(f"Backup cleanup failed: {e}", False, request_id)
            return False


# Convenience functions for your existing codebase
@with_request_id
def run_migration(batch_size: int = 1000, cleanup_backup: bool = False, request_id=None):
    """
    Run the complete migration process using your DatabaseConfig

    Args:
        batch_size: Number of records to migrate per batch
        cleanup_backup: Whether to drop the backup table after successful migration
        request_id: Request ID for logging

    Returns:
        tuple: (success: bool, migration_log: list)
    """
    migration = EmbeddingTableMigration()

    success = migration.run_complete_migration(batch_size)

    if success and cleanup_backup:
        migration.cleanup_backup_table()

    return success, migration.migration_log


@with_request_id
def rollback_migration(request_id=None):
    """
    Rollback a migration

    Args:
        request_id: Request ID for logging

    Returns:
        tuple: (success: bool, migration_log: list)
    """
    migration = EmbeddingTableMigration()
    success = migration.rollback_migration()
    return success, migration.migration_log


# Helper functions for working with TinyLlama embeddings after migration
@with_request_id
def store_tinyllama_embedding(document_id: int, text: str, model_name: str = "tinyllama_st", request_id=None):
    """
    Generate and store a TinyLlama embedding using sentence transformers

    Args:
        document_id: Document ID
        text: Text to embed
        model_name: Model identifier
        request_id: Request ID for logging

    Returns:
        dict with embedding info or None
    """
    try:
        from sentence_transformers import SentenceTransformer

        # Load a sentence transformer model (adjust as needed)
        st_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embedding
        embedding = st_model.encode([text])[0].tolist()

        db_config = DatabaseConfig()
        with db_config.main_session() as session:
            # Create DocumentEmbedding instance using the new structure
            insert_sql = """
            INSERT INTO document_embedding 
            (document_id, model_name, embedding_vector, actual_dimensions, embedding_metadata, created_at, updated_at)
            VALUES (:document_id, :model_name, :embedding_vector, :actual_dimensions, :embedding_metadata, NOW(), NOW())
            RETURNING id
            """

            metadata = {
                'dimensions': len(embedding),
                'model_type': f'sentence_transformer_{len(embedding)}d',
                'model_family': 'sentence_transformers',
                'creation_timestamp': datetime.utcnow().isoformat(),
                'storage_method': 'direct_tinyllama'
            }

            result = session.execute(text(insert_sql), {
                'document_id': document_id,
                'model_name': model_name,
                'embedding_vector': embedding,
                'actual_dimensions': len(embedding),
                'embedding_metadata': json.dumps(metadata)
            })

            embedding_id = result.fetchone()[0]
            session.commit()

            info_id(f"Stored TinyLlama embedding: doc_id={document_id}, dims={len(embedding)}", request_id)

            return {
                'id': embedding_id,
                'document_id': document_id,
                'dimensions': len(embedding),
                'model_name': model_name,
                'success': True
            }

    except Exception as e:
        error_id(f"Error storing TinyLlama embedding: {e}", request_id)
        return None


@with_request_id
def get_embeddings_by_dimension(dimensions: int, model_pattern: str = None, request_id=None):
    """
    Find all embeddings with specific dimensions

    Args:
        dimensions: Target dimension count
        model_pattern: Optional pattern to match model names
        request_id: Request ID for logging

    Returns:
        List of embedding records
    """
    try:
        db_config = DatabaseConfig()
        with db_config.main_session() as session:
            query = """
            SELECT id, document_id, model_name, actual_dimensions, embedding_metadata, created_at
            FROM document_embedding 
            WHERE actual_dimensions = :dimensions
            """

            params = {'dimensions': dimensions}

            if model_pattern:
                query += " AND model_name ILIKE :pattern"
                params['pattern'] = f'%{model_pattern}%'

            results = session.execute(text(query), params).fetchall()

            return [dict(row._mapping) for row in results]

    except Exception as e:
        error_id(f"Error querying embeddings by dimension: {e}", request_id)
        return []


@with_request_id
def get_migration_statistics(request_id=None):
    """
    Get statistics about the migrated embedding table

    Args:
        request_id: Request ID for logging

    Returns:
        dict: Statistics about dimensions and models
    """
    try:
        db_config = DatabaseConfig()
        with db_config.main_session() as session:
            # Get dimension distribution
            query = """
            SELECT 
                actual_dimensions,
                COUNT(*) as count,
                array_agg(DISTINCT model_name) as models
            FROM document_embedding 
            WHERE actual_dimensions IS NOT NULL
            GROUP BY actual_dimensions
            ORDER BY actual_dimensions
            """

            results = session.execute(text(query)).fetchall()

            stats = {
                'total_embeddings': session.execute(text("SELECT COUNT(*) FROM document_embedding")).fetchone()[0],
                'embeddings_with_dimensions': session.execute(text(
                    "SELECT COUNT(*) FROM document_embedding WHERE actual_dimensions IS NOT NULL"
                )).fetchone()[0],
                'dimension_distribution': {}
            }

            for result in results:
                stats['dimension_distribution'][result.actual_dimensions] = {
                    'count': result.count,
                    'models': result.models
                }

            return stats

    except Exception as e:
        error_id(f"Error getting migration statistics: {e}", request_id)
        return {'error': str(e)}


# Usage example
if __name__ == "__main__":
    # Example of how to use this with your existing setup

    print("Starting safe table migration...")
    success, log = run_migration(batch_size=500, cleanup_backup=False)

    if success:
        print(" Migration completed successfully!")
        print("\nMigration log:")
        for entry in log:
            status = "" if entry['success'] else "✗"
            print(f"  {status} {entry['timestamp']}: {entry['message']}")

        # Show statistics
        print("\nPost-migration statistics:")
        stats = get_migration_statistics()
        print(f"Total embeddings: {stats.get('total_embeddings', 0)}")
        for dim, info in stats.get('dimension_distribution', {}).items():
            print(f"  {dim}d: {info['count']} embeddings from models {info['models']}")

    else:
        print("✗ Migration failed!")
        print("\nError log:")
        for entry in log:
            if not entry['success']:
                print(f"  ✗ {entry['timestamp']}: {entry['message']}")

        # Ask if user wants to rollback
        user_input = input("\nDo you want to rollback the migration? (y/n): ")
        if user_input.lower() == 'y':
            print("Rolling back migration...")
            rollback_success, rollback_log = rollback_migration()
            if rollback_success:
                print(" Rollback successful")
            else:
                print("✗ Rollback failed - manual intervention may be required")