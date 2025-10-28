# modules/database_manager/migration_scripts/setup_document_fts_vectors.py

from sqlalchemy import text
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, set_request_id, with_request_id


@with_request_id
def setup_document_fts(db: DatabaseConfig, request_id=None):
    """
    Drop and recreate the documents_fts table, add indexes, triggers,
    and backfill from the document table.
    """
    rid = request_id or set_request_id()

    with db.main_session() as session:
        try:
            info_id("Dropping and recreating documents_fts table", rid)

            # Drop table if exists
            session.execute(text("DROP TABLE IF EXISTS documents_fts CASCADE"))
            session.commit()

            # Create documents_fts table
            session.execute(text("""
                CREATE TABLE documents_fts (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    file_path TEXT,
                    chunk_id INTEGER,  -- Links to document.id
                    complete_document_id INTEGER,  -- Links to complete_document.id
                    has_images BOOLEAN DEFAULT FALSE,
                    search_vector TSVECTOR,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """))
            session.commit()

            # Indexes
            index_statements = [
                "CREATE INDEX IF NOT EXISTS idx_documents_fts_search_vector ON documents_fts USING gin(search_vector)",
                "CREATE INDEX IF NOT EXISTS idx_documents_fts_title ON documents_fts(title)",
                "CREATE INDEX IF NOT EXISTS idx_documents_fts_chunk_id ON documents_fts(chunk_id)",
                "CREATE INDEX IF NOT EXISTS idx_documents_fts_complete_doc_id ON documents_fts(complete_document_id)",
                "CREATE INDEX IF NOT EXISTS idx_documents_fts_has_images ON documents_fts(has_images)",
                "CREATE INDEX IF NOT EXISTS idx_documents_fts_title_gin ON documents_fts USING gin(title gin_trgm_ops)",
                "CREATE INDEX IF NOT EXISTS idx_documents_fts_content_gin ON documents_fts USING gin(content gin_trgm_ops)"
            ]

            for stmt in index_statements:
                session.execute(text(stmt))
                session.commit()

            # Trigger function
            session.execute(text("""
                CREATE OR REPLACE FUNCTION update_documents_fts_vector() RETURNS trigger AS $$
                BEGIN
                    NEW.search_vector :=
                        to_tsvector('english',
                            NEW.title || ' ' ||
                            COALESCE(NEW.content, '') || ' ' ||
                            COALESCE(NEW.file_path, '')
                        );
                    NEW.updated_at := NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """))
            session.commit()

            # Trigger
            session.execute(text("DROP TRIGGER IF EXISTS documents_fts_vector_update ON documents_fts"))
            session.commit()
            session.execute(text("""
                CREATE TRIGGER documents_fts_vector_update
                BEFORE INSERT OR UPDATE ON documents_fts
                FOR EACH ROW EXECUTE FUNCTION update_documents_fts_vector()
            """))
            session.commit()

            info_id("Backfilling documents_fts from document table", rid)

            # Backfill from existing document table
            session.execute(text("""
                INSERT INTO documents_fts (title, content, file_path, chunk_id, complete_document_id, has_images)
                SELECT
                    name,
                    content,
                    file_path,
                    id AS chunk_id,
                    complete_document_id,
                    FALSE
                FROM document
            """))
            session.commit()

            info_id("Backfill complete for documents_fts", rid)

        except Exception as e:
            session.rollback()
            error_id(f"Failed during setup_document_fts: {e}", rid)
            raise


def main():
    rid = set_request_id()
    db = DatabaseConfig()
    setup_document_fts(db, request_id=rid)


if __name__ == "__main__":
    main()
