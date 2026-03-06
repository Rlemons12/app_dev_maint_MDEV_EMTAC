import os
from pathlib import Path
from dotenv import load_dotenv


class PostgresTestConfig:
    """
    Enterprise Postgres test harness.

    Forces project to use TEST_DATABASE_URL before
    any project modules are imported.
    """

    @staticmethod
    def force_environment():

        # -------------------------------------------------
        # Load .env.test automatically if present
        # -------------------------------------------------
        root = Path(__file__).resolve().parents[1]
        env_path = root / ".env.test"

        if env_path.exists():
            load_dotenv(env_path)

        # -------------------------------------------------
        # Read TEST_DATABASE_URL
        # -------------------------------------------------
        test_db_url = os.getenv("TEST_DATABASE_URL")

        if not test_db_url:
            raise RuntimeError(
                "TEST_DATABASE_URL is required.\n"
                "Example:\n"
                "postgresql+psycopg2://postgres:pass@127.0.0.1:5432/emtac_test"
            )

        # -------------------------------------------------
        # Safety check — refuse non-test DB
        # -------------------------------------------------
        if "test" not in test_db_url.lower():
            raise RuntimeError(
                f"Refusing to use non-test database:\n{test_db_url}"
            )

        # -------------------------------------------------
        # Force runtime environment
        # -------------------------------------------------
        os.environ["DATABASE_URL"] = test_db_url
        os.environ["DATABASE_REVISION_URL"] = test_db_url

        print(f"[TEST CONFIG] Using TEST database: {test_db_url}")

        return test_db_url
