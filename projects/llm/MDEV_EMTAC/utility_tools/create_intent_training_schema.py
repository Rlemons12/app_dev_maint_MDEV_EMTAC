from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# import YOUR models
from modules.emtac_ai.models.emtac_ai_db_models import Base

# -------------------------------------------------------
# Database connection (training DB)
# -------------------------------------------------------
DB_URL = (
    "postgresql+psycopg2://"
    "emtac_trainer:emtac_trainer123"
    "@localhost:5432/emtac_training"
)

engine = create_engine(
    DB_URL,
    echo=True,          # IMPORTANT: shows SQL being executed
    future=True,
)

# -------------------------------------------------------
# Create tables
# -------------------------------------------------------
if __name__ == "__main__":
    print("Creating intent_training tables...")
    Base.metadata.create_all(engine)
    print("Done.")
