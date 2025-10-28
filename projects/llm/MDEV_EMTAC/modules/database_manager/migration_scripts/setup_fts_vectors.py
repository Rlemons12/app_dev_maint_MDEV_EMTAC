from sqlalchemy import text
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, warning_id, error_id, get_request_id


def setup_full_text_search():
    rid = get_request_id()
    print(f"[{rid}] Starting FTS setup...")

    db_config = DatabaseConfig()
    if not db_config.is_postgresql:
        error_id("FTS setup requires PostgreSQL. Current DB is not PostgreSQL.")
        print("❌ FTS setup requires PostgreSQL. Exiting.")
        return

    # Tables and text columns mapped correctly to your models
    tables = {
        "site_location": ["title", "room_number", "site_area"],
        "area": ["name", "description"],
        "equipment_group": ["name", "description"],
        "model": ["name", "description"],
        "asset_number": ["number", "description"],
        "location": ["name", "description"],
        "subassembly": ["name", "description"],
        "component_assembly": ["name", "description"],
        "assembly_view": ["name", "description"],
        "part": ["part_number", "name", "oem_mfg", "model", "class_flag", "notes", "documentation"],
        "document": ["name","content"],   # fixed from title → name
        "image": ["title", "description", "img_metadata::text"],
        "drawing": ["drw_equipment_name", "drw_number", "drw_name", "drw_revision", "drw_spare_part_number"],
        "problem": ["name","description"],
        "solution": ["name", "description"],
        "task": ["name", "description"],
        "tool": ["name", "description"],
        "tool_manufacturer": ["name", "description"],

    }

    with db_config.main_session() as session:
        for table_name, text_columns in tables.items():
            try:
                msg = f"Setting up FTS for table: {table_name}"
                info_id(msg, rid)
                print(f"[{rid}] {msg}")

                # 1. Add column
                session.execute(text(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN IF NOT EXISTS search_vector tsvector
                """))

                # 2. GIN index
                session.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_fts
                    ON {table_name} USING gin(search_vector)
                """))

                # 3. Trigger function
                columns_concat = " || ' ' || ".join([f"COALESCE(NEW.{col}, '')" for col in text_columns])
                session.execute(text(f"""
                    CREATE OR REPLACE FUNCTION update_{table_name}_search_vector()
                    RETURNS trigger AS $$
                    BEGIN
                        NEW.search_vector := to_tsvector('english', {columns_concat});
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """))

                # 4. Trigger
                session.execute(text(f"""
                    DROP TRIGGER IF EXISTS {table_name}_search_vector_trigger ON {table_name};
                    CREATE TRIGGER {table_name}_search_vector_trigger
                    BEFORE INSERT OR UPDATE ON {table_name}
                    FOR EACH ROW EXECUTE FUNCTION update_{table_name}_search_vector();
                """))

                # 5. Backfill
                cols_expr = " || ' ' || ".join([f"COALESCE({col}, '')" for col in text_columns])
                session.execute(text(f"""
                    UPDATE {table_name}
                    SET search_vector = to_tsvector('english', {cols_expr})
                    WHERE search_vector IS NULL;
                """))

                session.commit()
                msg = f"FTS setup + backfill completed for {table_name}"
                info_id(msg, rid)
                print(f"[{rid}] {msg}")

            except Exception as e:
                warning_id(f"Skipping FTS setup for {table_name}: {e}", rid)
                print(f"[{rid}] ⚠️ Skipping FTS setup for {table_name}: {e}")
                session.rollback()

        # --- Validation
        validation_targets = {
            "part": "part_number",
            "document": "name",
            "image": "title",
            "drawing": "drw_name",
            "problem": "description",
            "solution": "description",
            "task": "name",
        }

        for table, sample_col in validation_targets.items():
            try:
                rows = session.execute(text(f"""
                    SELECT {sample_col}, search_vector
                    FROM {table}
                    WHERE search_vector IS NOT NULL
                    LIMIT 5;
                """)).fetchall()
                if rows:
                    print(f"\n[{rid}] Validation results for {table}:")
                    for r in rows:
                        print(f"[{rid}]   {sample_col}={r[0]} | search_vector={r[1]}")
                else:
                    print(f"[{rid}] ⚠️ No populated search_vector found in {table}")
            except Exception as e:
                print(f"[{rid}] ⚠️ Validation query failed for {table}: {e}")

        print(f"[{rid}] ✅ Full-text search setup + validation completed")


if __name__ == "__main__":
    setup_full_text_search()
