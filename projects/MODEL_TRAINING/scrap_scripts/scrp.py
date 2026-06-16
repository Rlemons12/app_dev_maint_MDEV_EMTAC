import os
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------
# Configuration (read from env)
# ---------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set")

print(f"[INFO] Connecting to MLflow DB via:")
print(f"       {MLFLOW_TRACKING_URI}")

# ---------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------
engine = create_engine(MLFLOW_TRACKING_URI)

# ---------------------------------------------------------------------
# Query all tables
# ---------------------------------------------------------------------
QUERY = """
SELECT
    schemaname,
    tablename
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY schemaname, tablename;
"""

with engine.connect() as conn:
    result = conn.execute(text(QUERY)).fetchall()

# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------
if not result:
    print("[WARN] No tables found")
else:
    print("\n[MLFLOW TABLES]")
    current_schema = None
    for schema, table in result:
        if schema != current_schema:
            print(f"\nSchema: {schema}")
            current_schema = schema
        print(f"  - {table}")

print("\n[SUCCESS] Table listing complete")
