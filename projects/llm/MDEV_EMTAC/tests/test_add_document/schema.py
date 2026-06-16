import os
from sqlalchemy import create_engine
from modules.configuration.base import Base

def create_test_schema():
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        raise RuntimeError("TEST_DATABASE_URL not set")

    engine = create_engine(test_db_url)

    # Drop all tables first (clean state)
    Base.metadata.drop_all(engine)

    # Create all tables
    Base.metadata.create_all(engine)

    return engine
