import os
from pathlib import Path
import textwrap


# ==========================================================
# TARGET ROOT (AUTO-DETECT BASED ON THIS SCRIPT LOCATION)
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_CONFIG_DIR = PROJECT_ROOT / "test_config"

print(f"Project root detected: {PROJECT_ROOT}")
print(f"Creating test_config in: {TEST_CONFIG_DIR}")


# ==========================================================
# FILE CONTENTS
# ==========================================================

CONFIG_CONTENT = """
import os
import tempfile
from pathlib import Path


class TestConfig:
    \"""
    Isolated configuration layer for EMTAC testing.
    Does NOT interfere with production configuration.
    \"""

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # ---------------------------------------------------
    # DATABASE CONFIGURATION
    # ---------------------------------------------------

    # Default: SQLite in-memory
    DATABASE_URL = "sqlite:///:memory:"

    # Uncomment for Postgres realism
    # DATABASE_URL = "postgresql+psycopg2://postgres:emtac123@127.0.0.1:5432/emtac_test"

    TESTING = True
    DEBUG = False

    # ---------------------------------------------------
    # FILE STORAGE (ISOLATED)
    # ---------------------------------------------------

    DATABASE_DIR = tempfile.mkdtemp(prefix="emtac_test_")

    DB_DOC = os.path.join(DATABASE_DIR, "DB_DOC")
    DB_IMAGES = os.path.join(DATABASE_DIR, "DB_IMAGES")

    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------

    @classmethod
    def init_directories(cls):
        os.makedirs(cls.DB_DOC, exist_ok=True)
        os.makedirs(cls.DB_IMAGES, exist_ok=True)

    @classmethod
    def apply_environment(cls):
        os.environ["DATABASE_URL"] = cls.DATABASE_URL
        os.environ["DATABASE_DIR"] = cls.DATABASE_DIR
        os.environ["EMTAC_ENV"] = "test"
"""

DB_BOOTSTRAP_CONTENT = """
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from test_config.config import TestConfig


def create_test_engine():
    return create_engine(TestConfig.DATABASE_URL)


def create_test_session():
    engine = create_test_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def bootstrap_schema(Base):
    engine = create_test_engine()
    Base.metadata.create_all(engine)
    return engine
"""

TRACE_UTILS_CONTENT = """
import sys
from contextlib import contextmanager


@contextmanager
def trace_calls():
    \"""
    Capture full Python call trace during execution.
    \"""
    call_stack = []

    def tracer(frame, event, arg):
        if event == "call":
            func_name = frame.f_code.co_name
            module = frame.f_globals.get("__name__", "")
            call_stack.append(f"{module}.{func_name}")
        return tracer

    sys.settrace(tracer)
    try:
        yield call_stack
    finally:
        sys.settrace(None)
"""

INIT_CONTENT = """
from .config import TestConfig
from .db_bootstrap import create_test_engine, create_test_session, bootstrap_schema
from .trace_utils import trace_calls
"""


# ==========================================================
# CREATE STRUCTURE
# ==========================================================

def create_file(path: Path, content: str):
    if path.exists():
        print(f"Skipping existing file: {path.name}")
        return
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    print(f"Created: {path.name}")


def main():
    TEST_CONFIG_DIR.mkdir(exist_ok=True)

    create_file(TEST_CONFIG_DIR / "__init__.py", INIT_CONTENT)
    create_file(TEST_CONFIG_DIR / "config.py", CONFIG_CONTENT)
    create_file(TEST_CONFIG_DIR / "db_bootstrap.py", DB_BOOTSTRAP_CONTENT)
    create_file(TEST_CONFIG_DIR / "trace_utils.py", TRACE_UTILS_CONTENT)

    print("\\nTestConfig module successfully created.")
    print("You can now import it anywhere with:")
    print("    from test_config import TestConfig")


if __name__ == "__main__":
    main()
