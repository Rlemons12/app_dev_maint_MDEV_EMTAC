import os
import threading
import time
from functools import wraps
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from configuration.log_config import logger
from pathlib import Path

# Explicit EMTAC environment (required for PyCharm Console + scripts)
ENV_PATH = Path(r"E:\emtac\dev_env\.env")
# ---------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------
load_dotenv()

# =====================================================
# Connection limiting (shared, optional)
# =====================================================
CONNECTION_LIMITING_ENABLED = False
MAX_CONCURRENT_CONNECTIONS = int(os.getenv("MAX_DB_CONNECTIONS", "10"))
CONNECTION_TIMEOUT = int(os.getenv("DB_CONNECTION_TIMEOUT", "60"))

_db_semaphore = threading.Semaphore(MAX_CONCURRENT_CONNECTIONS)
_active_connections = 0
_connection_lock = threading.Lock()


def _is_postgresql_url(database_url: str) -> bool:
    if not database_url:
        return False
    url = database_url.lower().strip()
    return (
        url.startswith("postgresql://")
        or url.startswith("postgresql+psycopg2://")
        or url.startswith("postgresql+asyncpg://")
        or url.startswith("postgres://")
    )


def with_connection_limiting(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _active_connections

        if not CONNECTION_LIMITING_ENABLED:
            return func(*args, **kwargs)

        start_time = time.time()
        acquired = False

        while not acquired:
            acquired = _db_semaphore.acquire(blocking=False)
            if acquired:
                break
            if time.time() - start_time > CONNECTION_TIMEOUT:
                logger.warning("[DB] Connection timeout — forcing acquire")
                _db_semaphore.acquire()
                break
            time.sleep(0.1)

        try:
            with _connection_lock:
                _active_connections += 1

            session = func(*args, **kwargs)

        except Exception:
            _db_semaphore.release()
            raise

        original_close = session.close

        def patched_close():
            global _active_connections
            original_close()
            _db_semaphore.release()
            with _connection_lock:
                _active_connections = max(0, _active_connections - 1)

        session.close = patched_close
        return session

    return wrapper


# =====================================================
# DatabaseConfig (GENERAL – PostgreSQL preferred)
# =====================================================
class DatabaseConfig:
    """
    General-purpose database configuration.

    - DATABASE_URL required
    - PostgreSQL preferred
    - SQLite fallback ONLY if PostgreSQL connection fails
    """

    def __init__(self):
        env_url = os.getenv("DATABASE_URL")
        if not env_url:
            raise RuntimeError(
                "DATABASE_URL is not set. "
                "Set DATABASE_URL or use TrainingDatabaseConfig explicitly."
            )

        self.database_url = env_url.strip()
        self.is_postgresql = _is_postgresql_url(self.database_url)

        try:
            if self.is_postgresql:
                logger.info("[DB] Using PostgreSQL: %s", self.database_url)
                self.engine = create_engine(
                    self.database_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False,
                    connect_args={
                        "application_name": "emtac_model_training",
                        "options": "-c timezone=utc",
                    },
                )
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
            else:
                logger.warning("[DB] Using SQLite: %s", self.database_url)
                self.engine = create_engine(self.database_url)

        except Exception as e:
            logger.error("[DB] PostgreSQL failed, falling back to SQLite: %s", e)
            self.database_url = "sqlite:///emtac_training.db"
            self.engine = create_engine(self.database_url)
            self.is_postgresql = False

        self.Base = declarative_base()
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self.SessionMaker = sessionmaker(bind=self.engine)

        if self.is_postgresql:
            self._apply_postgresql_settings(self.engine)
        else:
            self._apply_sqlite_pragmas(self.engine)

        logger.info(
            "[DB] Initialized (%s), connection limiting=%s",
            "PostgreSQL" if self.is_postgresql else "SQLite",
            CONNECTION_LIMITING_ENABLED,
        )

    @with_connection_limiting
    def get_session(self):
        return self.Session()

    @contextmanager
    def session(self):
        sess = self.get_session() if CONNECTION_LIMITING_ENABLED else self.SessionMaker()
        try:
            yield sess
            sess.commit()
        except Exception as e:
            sess.rollback()
            logger.error("[DB] Session error: %s", e)
            raise
        finally:
            sess.close()

    def get_engine(self):
        return self.engine

    def _apply_sqlite_pragmas(self, engine):
        def set_sqlite_pragmas(dbapi_connection, _):
            if dbapi_connection.__class__.__module__.startswith("sqlite3"):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA foreign_keys=ON;")
                cursor.close()

        event.listen(engine, "connect", set_sqlite_pragmas)

    def _apply_postgresql_settings(self, engine):
        def set_postgres_settings(dbapi_connection, _):
            with dbapi_connection.cursor() as cursor:
                cursor.execute("SET timezone = 'UTC'")
                cursor.execute("SET statement_timeout = '30s'")
                cursor.execute("SET idle_in_transaction_session_timeout = '60s'")

        event.listen(engine, "connect", set_postgres_settings)

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def list_tables(self, schema: str = "public"):
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = :schema
                    ORDER BY tablename
                """),
                {"schema": schema},
            ).fetchall()
            return [r[0] for r in rows]

    def test_connection(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "ok", "db": "PostgreSQL" if self.is_postgresql else "SQLite"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def info(self):
        return {
            "type": "DatabaseConfig",
            "database_url": self.database_url,
            "is_postgresql": self.is_postgresql,
            "connection_limiting": CONNECTION_LIMITING_ENABLED,
        }

# =====================================================
# TrainingDatabaseConfig (STRICT)
# =====================================================
class TrainingDatabaseConfig:
    """
    STRICT training-only database configuration.

    - PostgreSQL ONLY
    - Explicit schema
    - Fail-fast
    """

    def __init__(self):
        self.db_name = os.getenv("POSTGRES_TRAIN_DB")
        self.db_user = os.getenv("POSTGRES_TRAIN_USER")
        self.db_pass = os.getenv("POSTGRES_TRAIN_PASSWORD")
        self.db_host = os.getenv("POSTGRES_TRAIN_HOST", "localhost")
        self.db_port = os.getenv("POSTGRES_TRAIN_PORT", "5432")
        self.schema = os.getenv("POSTGRES_TRAIN_SCHEMA", "intent_training")

        missing = [
            k for k, v in {
                "POSTGRES_TRAIN_DB": self.db_name,
                "POSTGRES_TRAIN_USER": self.db_user,
                "POSTGRES_TRAIN_PASSWORD": self.db_pass,
            }.items()
            if not v
        ]

        if missing:
            raise RuntimeError(f"[TRAINING-DB] Missing env vars: {missing}")

        self.database_url = (
            f"postgresql+psycopg2://{self.db_user}:{self.db_pass}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

        logger.info("[TRAINING-DB] Using %s", self.database_url)

        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=False,
            connect_args={
                "application_name": "emtac_training",
                "options": f"-c search_path={self.schema},public",
            },
        )

        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            exists = conn.execute(
                text(
                    """
                    SELECT schema_name
                    FROM information_schema.schemata
                    WHERE schema_name = :schema
                    """
                ),
                {"schema": self.schema},
            ).scalar()

            if not exists:
                raise RuntimeError(f"[TRAINING-DB] Schema '{self.schema}' does not exist")

        self.Base = declarative_base()
        self.SessionMaker = sessionmaker(bind=self.engine)

        event.listen(self.engine, "connect", self._set_postgres_settings)
        logger.info("[TRAINING-DB] Schema '%s' validated", self.schema)

    def get_session(self):
        return self.SessionMaker()

    def get_engine(self):
        return self.engine

    def get_base(self):
        return self.Base

    @staticmethod
    def _set_postgres_settings(dbapi_connection, _):
        with dbapi_connection.cursor() as cur:
            cur.execute("SET timezone = 'UTC'")
            cur.execute("SET statement_timeout = '0'")
            cur.execute("SET idle_in_transaction_session_timeout = '60s'")

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def list_tables(self):
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = :schema
                    ORDER BY tablename
                """),
                {"schema": self.schema},
            ).fetchall()
            return [r[0] for r in rows]

    def test_connection(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {
                "status": "ok",
                "database": self.db_name,
                "schema": self.schema,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def info(self):
        return {
            "type": "TrainingDatabaseConfig",
            "database": self.db_name,
            "schema": self.schema,
            "database_url": self.database_url,
        }



# =====================================================
# MLflowDatabaseConfig (STRICT)
# =====================================================
class MLflowDatabaseConfig:
    """
    STRICT MLflow tracking DB configuration.

    - PostgreSQL ONLY
    - public schema
    - Fail-fast
    """

    def __init__(self):
        self.db_name = os.getenv("POSTGRES_MLFLOW_DB", "mlflow_training")
        self.db_user = os.getenv("POSTGRES_MLFLOW_USER")
        self.db_pass = os.getenv("POSTGRES_MLFLOW_PASSWORD")
        self.db_host = os.getenv("POSTGRES_MLFLOW_HOST", "localhost")
        self.db_port = os.getenv("POSTGRES_MLFLOW_PORT", "5432")

        missing = [
            k for k, v in {
                "POSTGRES_MLFLOW_USER": self.db_user,
                "POSTGRES_MLFLOW_PASSWORD": self.db_pass,
            }.items()
            if not v
        ]

        if missing:
            raise RuntimeError(f"[MLFLOW-DB] Missing env vars: {missing}")

        self.database_url = (
            f"postgresql+psycopg2://{self.db_user}:{self.db_pass}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

        logger.info("[MLFLOW-DB] Using %s", self.database_url)

        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=False,
            connect_args={
                "application_name": "emtac_mlflow",
                "options": "-c search_path=public",
            },
        )

        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        self.SessionMaker = sessionmaker(bind=self.engine)
        event.listen(self.engine, "connect", self._set_postgres_settings)

    def get_engine(self):
        return self.engine

    def get_session(self):
        return self.SessionMaker()

    def get_database_url(self):
        return self.database_url

    @staticmethod
    def _set_postgres_settings(dbapi_connection, _):
        with dbapi_connection.cursor() as cur:
            cur.execute("SET timezone = 'UTC'")
            cur.execute("SET statement_timeout = '30s'")
            cur.execute("SET idle_in_transaction_session_timeout = '60s'")

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def list_tables(self, schema: str = "public"):
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = :schema
                    ORDER BY tablename
                """),
                {"schema": schema},
            ).fetchall()
            return [r[0] for r in rows]

    def test_connection(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {
                "status": "ok",
                "database": self.db_name,
                "schema": "public",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def info(self):
        return {
            "type": "MLflowDatabaseConfig",
            "database": self.db_name,
            "database_url": self.database_url,
        }

