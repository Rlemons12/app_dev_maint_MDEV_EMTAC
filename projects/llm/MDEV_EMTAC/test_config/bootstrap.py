"""
test_config.bootstrap

Forces TEST_DATABASE_URL before any EMTAC imports.
Prevents accidental production DB usage.
Auto-loads .env.test from project root.
"""

import os
import logging
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

log = logging.getLogger(__name__)


def _validate_test_db_url(db_url: str) -> None:
    """
    Strict validation to prevent accidental production usage.
    """

    parsed = urlparse(db_url)

    if parsed.scheme not in ("postgresql", "postgresql+psycopg2"):
        raise RuntimeError(
            f"Test DB must be PostgreSQL. Got scheme: {parsed.scheme}"
        )

    if not parsed.hostname:
        raise RuntimeError("Invalid TEST_DATABASE_URL: missing hostname")

    if not parsed.path or parsed.path == "/":
        raise RuntimeError("Invalid TEST_DATABASE_URL: missing database name")

    db_name = parsed.path.lstrip("/")

    # Require explicit test naming convention
    if "test" not in db_name.lower():
        raise RuntimeError(
            f"Refusing to use non-test database: {db_name}\n"
            "Database name must contain 'test'."
        )


def bootstrap_test_env() -> str:
    """
    Load .env.test and force DATABASE_URL to TEST_DATABASE_URL.

    MUST be called before importing any EMTAC modules that
    initialize DatabaseConfig or read DATABASE_URL.
    """

    # ------------------------------------------------------
    # Locate project root (parent of test_config)
    # ------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    env_test_path = project_root / ".env.test"

    # ------------------------------------------------------
    # Load .env.test (override ensures it wins)
    # ------------------------------------------------------
    if env_test_path.exists():
        load_dotenv(env_test_path, override=True)
        log.info(f"[TEST CONFIG] Loaded .env.test from {env_test_path}")
    else:
        raise RuntimeError(
            f".env.test not found at expected location:\n{env_test_path}"
        )

    # ------------------------------------------------------
    # Read TEST_DATABASE_URL
    # ------------------------------------------------------
    test_db_url = os.getenv("TEST_DATABASE_URL")

    if not test_db_url:
        raise RuntimeError(
            "TEST_DATABASE_URL must be set in .env.test.\n"
            "Example:\n"
            "postgresql+psycopg2://postgres:pass@127.0.0.1:5432/emtac_test"
        )

    # ------------------------------------------------------
    # Strict validation
    # ------------------------------------------------------
    _validate_test_db_url(test_db_url)

    # ------------------------------------------------------
    # Force runtime environment
    # ------------------------------------------------------
    os.environ["DATABASE_URL"] = test_db_url
    os.environ["DATABASE_REVISION_URL"] = test_db_url

    log.info(f"[TEST CONFIG] Using test DB: {test_db_url}")

    return test_db_url
