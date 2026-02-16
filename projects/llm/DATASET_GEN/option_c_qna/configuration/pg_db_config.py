# DATASET_GEN/option_c_qna/configuration/pg_db_config.py
# Centralized PostgreSQL configuration for EMTAC + Training datasets

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------
# FIXED .env PATH FOR YOUR SYSTEM
# ---------------------------------------------------------
ENV_PATH = Path(r"E:\emtac\dev_env\.env")

if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found at expected path: {ENV_PATH}")

load_dotenv(ENV_PATH)


# =========================================================
# 🟦 1. Production EMTAC Database
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env — required for EMTAC main DB")

emtac_engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

EMTACSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=emtac_engine
)


# =========================================================
# 🟩 2. Training / Fine-Tuning Database (emtac_training)
# =========================================================
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

train_engine = create_engine(
    TRAIN_DATABASE_URL,
    pool_pre_ping=True,
)

TrainSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=train_engine
)


# =========================================================
# 🟨 Unified Access for Option-C Q&A Pipeline
# =========================================================
# Option C *always* uses the training database only.
QNA_ENGINE = train_engine
QNA_SessionLocal = TrainSessionLocal


# =========================================================
# Helper functions
# =========================================================
def get_training_session():
    """Return a new SQLAlchemy session for the training DB."""
    return TrainSessionLocal()


def get_emtac_session():
    """Return a new SQLAlchemy session for the main EMTAC DB."""
    return EMTACSessionLocal()


def get_qna_session():
    """Unified access for Option C Q&A models."""
    return QNA_SessionLocal()
