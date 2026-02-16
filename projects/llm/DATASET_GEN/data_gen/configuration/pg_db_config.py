# pg_db_config.py
# Centralized PostgreSQL configuration for EMTAC + Training datasets

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ---------------------------------------------------------
# LOAD ENVIRONMENT (.env)
# ---------------------------------------------------------
# Does not assume working directory; auto-resolves correct path
ROOT = Path(__file__).resolve().parents[3]   # points to E:\emtac\projects\llm
ENV_PATH = ROOT / "dev_env" / ".env"

if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found at expected path: {ENV_PATH}")

load_dotenv(ENV_PATH)



# ---------------------------------------------------------
# 🟦 1. Production EMTAC Database
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env — required for EMTAC main DB")

emtac_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
EMTACSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=emtac_engine)



# ---------------------------------------------------------
# 🟩 2. Training / Fine-Tuning Database (emtac_training)
# ---------------------------------------------------------
POSTGRES_TRAIN_DB = os.getenv("POSTGRES_TRAIN_DB")
POSTGRES_TRAIN_USER = os.getenv("POSTGRES_TRAIN_USER")
POSTGRES_TRAIN_PASSWORD = os.getenv("POSTGRES_TRAIN_PASSWORD")
POSTGRES_TRAIN_HOST = os.getenv("POSTGRES_TRAIN_HOST", "127.0.0.1")
POSTGRES_TRAIN_PORT = os.getenv("POSTGRES_TRAIN_PORT", "5432")

if not all([POSTGRES_TRAIN_DB, POSTGRES_TRAIN_USER, POSTGRES_TRAIN_PASSWORD]):
    raise ValueError(
        "Training DB credentials missing in .env "
        "(POSTGRES_TRAIN_DB, POSTGRES_TRAIN_USER, POSTGRES_TRAIN_PASSWORD)"
    )

TRAIN_DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_TRAIN_USER}:{POSTGRES_TRAIN_PASSWORD}"
    f"@{POSTGRES_TRAIN_HOST}:{POSTGRES_TRAIN_PORT}/{POSTGRES_TRAIN_DB}"
)

train_engine = create_engine(TRAIN_DATABASE_URL, pool_pre_ping=True)
TrainSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=train_engine)



# ---------------------------------------------------------
# Helper function to get DB session
# ---------------------------------------------------------
def get_training_session():
    """Return a new SQLAlchemy session for the training DB."""
    return TrainSessionLocal()


def get_emtac_session():
    """Return a new SQLAlchemy session for the main EMTAC DB."""
    return EMTACSessionLocal()
