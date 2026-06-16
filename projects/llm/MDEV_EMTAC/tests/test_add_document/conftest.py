import os

# ==================================================
# FORCE TEST DATABASE BEFORE ANY PROJECT IMPORTS
# ==================================================

test_db_url = os.getenv("TEST_DATABASE_URL")
if not test_db_url:
    raise RuntimeError(
        "TEST_DATABASE_URL is required. "
        "Example: postgresql+psycopg2://postgres:pass@127.0.0.1:5432/emtac_test"
    )

# Force both main + revision DB to use test DB
os.environ["DATABASE_URL"] = test_db_url
os.environ["DATABASE_REVISION_URL"] = test_db_url

# ==================================================
# SAFE TO IMPORT PROJECT MODULES
# ==================================================

import logging
import pytest
import requests
from flask import Flask
from sqlalchemy import create_engine, text

# 🔥 Use your real ORM Base
from modules.configuration.base import Base

# 🔥 IMPORTANT: Import models so metadata is populated
import modules.emtacdb.emtacdb_fts  # noqa

from .helpers.db import make_session_factory
from .helpers.fs import ensure_dirs
from .helpers.stubs import (
    stub_generate_embedding,
    stub_requests_post_success,
)
import logging
import sqlalchemy as sa
from sqlalchemy import text
import pytest


log = logging.getLogger(__name__)


# ==================================================
# ENGINE FIXTURE (SESSION SCOPE)
# ==================================================

@pytest.fixture(scope="session")
def engine():
    """
    Creates engine and ensures full schema exists once per test session.
    """
    engine = create_engine(test_db_url)

    # Create all ORM tables
    Base.metadata.create_all(engine)

    return engine


@pytest.fixture(scope="session")
def Session(engine):
    return make_session_factory(engine)


# ==================================================
# DB RESET BETWEEN TESTS
# ==================================================



def _truncate_all_tables(conn):
    """
    Truncate all tables in the current schema without relying on SQLAlchemy
    dependency sorting (which can fail if metadata has unresolved FKs).
    """
    # Disable FK checks (Postgres-safe)
    conn.execute(text("SET session_replication_role = replica;"))

    # Grab real tables from the database (NOT from SQLAlchemy metadata)
    rows = conn.execute(text("""
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public';
    """)).fetchall()

    table_names = [r[0] for r in rows]

    # Optional: skip alembic_version if you use migrations
    table_names = [t for t in table_names if t != "alembic_version"]

    if not table_names:
        logger.warning("No tables found to truncate.")
        conn.execute(text("SET session_replication_role = DEFAULT;"))
        return

    # Quote names properly
    quoted = ", ".join([f'"{t}"' for t in table_names])

    # CASCADE clears dependent tables too; RESTART IDENTITY resets sequences
    conn.execute(text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE;"))

    # Re-enable FK checks
    conn.execute(text("SET session_replication_role = DEFAULT;"))


@pytest.fixture(autouse=True)
def reset_db(engine):
    """
    Enterprise-safe DB reset between tests:
      - Keeps schema
      - Truncates all tables
      - Resets identity sequences
      - Works even if SQLAlchemy metadata has unresolved FK targets
    """
    with engine.begin() as conn:
        _truncate_all_tables(conn)

    yield

    # Clean after test too (double safety)
    with engine.begin() as conn:
        _truncate_all_tables(conn)



# ==================================================
# FLASK APP FIXTURE
# ==================================================

@pytest.fixture
def app(tmp_path, monkeypatch):
    """
    Flask app configured for enterprise Add Document route tests:
      - Real local Postgres (TEST_DATABASE_URL)
      - Isolated filesystem under tmp_path
      - Stub external AI + HTTP calls
    """

    app = Flask(__name__)
    app.config.update({
        "TESTING": True,
        "SECRET_KEY": "test-secret",

        # Isolated filesystem roots
        "DATABASE_DIR": str(tmp_path),
        "DATABASE_DOC": str(tmp_path / "DB_DOC"),
        "DATABASE_PATH_IMAGES_FOLDER": str(tmp_path / "DB_IMAGES"),
        "TEMPORARY_UPLOAD_FILES": str(tmp_path / "TMP_UPLOADS"),

        "CURRENT_EMBEDDING_MODEL": "NoEmbeddingModel",
    })

    ensure_dirs(
        app.config["DATABASE_DOC"],
        app.config["DATABASE_PATH_IMAGES_FOLDER"],
        app.config["TEMPORARY_UPLOAD_FILES"],
    )

    # Force env again inside test context
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    monkeypatch.setenv("DATABASE_REVISION_URL", test_db_url)

    # --------------------------------------------------
    # Stub AI embedding
    # --------------------------------------------------
    try:
        import plugins.ai_modules as ai_mod
        monkeypatch.setattr(ai_mod, "generate_embedding", stub_generate_embedding)
    except Exception as e:
        log.warning("Could not patch generate_embedding: %s", e)

    # --------------------------------------------------
    # Minimal index route (exists in real app)
    # --------------------------------------------------
    @app.route("/")
    def index():
        return "OK"

    # --------------------------------------------------
    # Stub outbound HTTP calls
    # --------------------------------------------------
    monkeypatch.setattr(requests, "post", stub_requests_post_success)

    # --------------------------------------------------
    # Register blueprint
    # --------------------------------------------------
    from blueprints.add_document_bp import add_document_bp
    app.register_blueprint(add_document_bp, url_prefix="/documents")

    return app


@pytest.fixture
def client(app):
    return app.test_client()
