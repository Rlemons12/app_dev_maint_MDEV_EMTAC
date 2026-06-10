from modules.configuration.log_config import info_id, error_id
from modules.configuration.config_env import DatabaseConfig
from sqlalchemy import text

def update_complete_document_table():
    request_id = "update-complete-document-rag-columns"

    sql = """
    ALTER TABLE public.complete_document
    ADD COLUMN IF NOT EXISTS summary TEXT;

    ALTER TABLE public.complete_document
    ADD COLUMN IF NOT EXISTS rag_metadata JSONB;

    ALTER TABLE public.complete_document
    ADD COLUMN IF NOT EXISTS topics JSONB;

    ALTER TABLE public.complete_document
    ADD COLUMN IF NOT EXISTS keywords JSONB;

    ALTER TABLE public.complete_document
    ADD COLUMN IF NOT EXISTS questions_answered JSONB;

    ALTER TABLE public.complete_document
    ADD COLUMN IF NOT EXISTS equipment JSONB;

    CREATE INDEX IF NOT EXISTS idx_complete_document_rag_metadata_gin
    ON public.complete_document
    USING GIN (rag_metadata);

    CREATE INDEX IF NOT EXISTS idx_complete_document_topics_gin
    ON public.complete_document
    USING GIN (topics);

    CREATE INDEX IF NOT EXISTS idx_complete_document_keywords_gin
    ON public.complete_document
    USING GIN (keywords);

    CREATE INDEX IF NOT EXISTS idx_complete_document_questions_answered_gin
    ON public.complete_document
    USING GIN (questions_answered);

    CREATE INDEX IF NOT EXISTS idx_complete_document_equipment_gin
    ON public.complete_document
    USING GIN (equipment);
    """

    verify_sql = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'complete_document'
      AND column_name IN (
          'summary',
          'rag_metadata',
          'topics',
          'keywords',
          'questions_answered',
          'equipment'
      )
    ORDER BY column_name;
    """

    db_config = DatabaseConfig()

    try:
        with db_config.main_session() as session:
            session.execute(text(sql))
            session.commit()

            rows = session.execute(text(verify_sql)).fetchall()

            info_id(
                f"Updated complete_document table. Verified columns: {rows}",
                request_id,
            )

            print("complete_document updated successfully.")
            print("Verified columns:")
            for row in rows:
                print(f" - {row[0]}: {row[1]}")

    except Exception as e:
        error_id(
            f"Failed updating complete_document table: {e}",
            request_id,
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    update_complete_document_table()