# modules/database_manager/search_scripts/fts_search.py

import argparse
from sqlalchemy import text
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, log_with_id
import re

TABLES = {
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
    "document": ["name", "content"],
    "image": ["title", "description"],
    "drawing": ["drw_name", "drw_equipment_name", "drw_number", "drw_revision", "drw_spare_part_number"],
    "problem": ["description"],
    "solution": ["description"],
    "task": ["name", "description"],
}


def build_tsquery(user_input: str) -> str:
    """Convert user input into safe tsquery string."""
    terms = re.findall(r'\w+', user_input)
    return " & ".join(terms)


def run_search(db, search_text: str, limit: int = 5):
    tsquery = build_tsquery(search_text)
    results = []

    # Use the context manager provided by DatabaseConfig
    with db.main_session() as session:
        for table, columns in TABLES.items():
            try:
                preview_expr = " || ' | ' || ".join([f"COALESCE({col}, '')" for col in columns])
                sql = text(f"""
                    SELECT id, {preview_expr} AS preview
                    FROM {table}
                    WHERE search_vector @@ to_tsquery('english', :tsq)
                    ORDER BY ts_rank(search_vector, to_tsquery('english', :tsq)) DESC
                    LIMIT :limit;
                """)
                rows = session.execute(sql, {"tsq": tsquery, "limit": limit}).fetchall()
                if rows:
                    results.append((table, rows))
            except Exception as e:
                error_id(f"FTS query failed for {table}: {e}")

    return results



def main():
    parser = argparse.ArgumentParser(description="Full-text search across multiple tables.")
    parser.add_argument("--limit", type=int, default=5, help="Max results per table")
    args = parser.parse_args()

    db = DatabaseConfig()

    # Multi-question loop
    while True:
        search_text = input("Enter search text (or 'quit' to exit): ").strip()
        if not search_text or search_text.lower() in {"quit", "exit"}:
            print("👋 Exiting search.")
            break

        info_id(f"Running search for: {search_text}")
        results = run_search(db, search_text, args.limit)

        if not results:
            print("⚠️ No results found.\n")
        else:
            for table, rows in results:
                print(f"\n🔎 {table.upper()}:")
                for r in rows:
                    print(f"  [id={r.id}] {r.preview}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
