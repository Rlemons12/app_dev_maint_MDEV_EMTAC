from sqlalchemy import text
from configuration.config_env import (
    DatabaseConfig,
    TrainingDatabaseConfig,
    MLflowDatabaseConfig,
)
from configuration.log_config import logger


def print_tables(engine, label: str):
    print("\n" + "=" * 80)
    print(f"{label} — TABLE LIST")
    print("=" * 80)

    with engine.connect() as conn:
        # PostgreSQL-compatible table listing
        rows = conn.execute(
            text("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name
            """)
        ).fetchall()

        if not rows:
            print("⚠️  No tables found")
            return

        for schema, table in rows:
            print(f"{schema}.{table}")


def validate_database_config():
    print("\n\nVALIDATING DatabaseConfig (general / fallback-safe)")
    print("-" * 80)

    db = DatabaseConfig()
    engine = db.get_engine()

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        print("✔ Connection OK")

    print_tables(engine, "DatabaseConfig")


def validate_training_database_config():
    print("\n\nVALIDATING TrainingDatabaseConfig (STRICT training DB)")
    print("-" * 80)

    db = TrainingDatabaseConfig()
    engine = db.get_engine()

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        print("✔ Connection OK")

    print_tables(engine, "TrainingDatabaseConfig")


def validate_mlflow_database_config():
    print("\n\nVALIDATING MLflowDatabaseConfig (STRICT MLflow DB)")
    print("-" * 80)

    db = MLflowDatabaseConfig()
    engine = db.get_engine()

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        print("✔ Connection OK")

    print_tables(engine, "MLflowDatabaseConfig")


if __name__ == "__main__":
    validate_database_config()
    validate_training_database_config()
    validate_mlflow_database_config()

    print("\n\nALL DATABASE CONFIGS VALIDATED SUCCESSFULLY")
