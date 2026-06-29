import os
from dotenv import load_dotenv
load_dotenv()


print("Loaded POSTGRES_TRAIN_PASSWORD:", os.getenv("POSTGRES_TRAIN_PASSWORD"))

from sqlalchemy import create_engine, text

def test_training_db():
    # Read from .env or use defaults
    train_db = os.getenv("POSTGRES_TRAIN_DB", "emtac_training")
    train_user = os.getenv("POSTGRES_TRAIN_USER", "postgres")
    train_password = os.getenv("POSTGRES_TRAIN_PASSWORD", "postgres")
    train_host = os.getenv("POSTGRES_TRAIN_HOST", "localhost")
    train_port = os.getenv("POSTGRES_TRAIN_PORT", "5432")

    url = f"postgresql+psycopg2://{train_user}:{train_password}@{train_host}:{train_port}/{train_db}"
    print(f"🔍 Testing connection to: {url}")

    try:
        engine = create_engine(url, echo=False)
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).fetchone()[0]
            print("✅ Connection successful!")
            print(f"PostgreSQL version: {version}")
    except Exception as e:
        print("❌ Connection failed:")
        print(e)

if __name__ == "__main__":
    test_training_db()
