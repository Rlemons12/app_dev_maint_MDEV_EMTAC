# modules/configuration/config_env.py
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Optional, Dict, Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from modules.configuration.config import DATABASE_URL, REVISION_CONTROL_DB_PATH
from modules.configuration.log_config import logger

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
CONNECTION_LIMITING_ENABLED = False
MAX_CONCURRENT_CONNECTIONS = int(os.environ.get("MAX_DB_CONNECTIONS", "10"))
CONNECTION_TIMEOUT = int(os.environ.get("DB_CONNECTION_TIMEOUT", "60"))

_main_db_semaphore = threading.Semaphore(MAX_CONCURRENT_CONNECTIONS)
_revision_db_semaphore = threading.Semaphore(MAX_CONCURRENT_CONNECTIONS)

_active_main_connections = 0
_active_revision_connections = 0
_connection_lock = threading.Lock()

# -----------------------------------------------------------------------------
# PostgreSQL request tracing context
# -----------------------------------------------------------------------------
_pg_request_id: ContextVar[Optional[str]] = ContextVar("pg_request_id", default=None)
_pg_endpoint: ContextVar[Optional[str]] = ContextVar("pg_endpoint", default=None)
_pg_path: ContextVar[Optional[str]] = ContextVar("pg_path", default=None)


def set_pg_request_context(
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    """
    Store the current request context so SQLAlchemy connection checkout hooks can
    tag PostgreSQL sessions with route/request metadata.
    """
    _pg_request_id.set(request_id)
    _pg_endpoint.set(endpoint)
    _pg_path.set(path)


def clear_pg_request_context() -> None:
    """Clear request-scoped PostgreSQL tagging context."""
    _pg_request_id.set(None)
    _pg_endpoint.set(None)
    _pg_path.set(None)


def build_pg_application_name(base_name: str = "emtac_app") -> str:
    """
    Build a PostgreSQL application_name string that includes request context.
    Keep this reasonably short because PostgreSQL truncates long values.
    """
    rid = _pg_request_id.get() or "no-rid"
    endpoint = _pg_endpoint.get() or "no-endpoint"
    path = (_pg_path.get() or "no-path").replace(" ", "_")

    app_name = f"{base_name}|rid={rid}|ep={endpoint}|path={path}"
    return app_name[:120]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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
    """
    Decorator to apply semaphore-based connection limiting to session factories.
    Preserves backwards compatibility with existing code.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        global _active_main_connections, _active_revision_connections

        if not CONNECTION_LIMITING_ENABLED:
            return func(*args, **kwargs)

        if "main" in func.__name__.lower():
            semaphore = _main_db_semaphore
            connection_type = "main"
        else:
            semaphore = _revision_db_semaphore
            connection_type = "revision"

        start_time = time.time()
        acquired = False

        while not acquired and (time.time() - start_time < CONNECTION_TIMEOUT):
            acquired = semaphore.acquire(blocking=False)
            if acquired:
                break

            elapsed = int(time.time() - start_time)
            if elapsed > 0 and elapsed % 5 == 0:
                logger.debug(
                    "Waiting for %s database connection (%ss)...",
                    connection_type,
                    elapsed,
                )
            time.sleep(0.1)

        if not acquired:
            logger.warning(
                "Timeout waiting for %s database connection after %ss",
                connection_type,
                CONNECTION_TIMEOUT,
            )
            semaphore.acquire(blocking=True)
            logger.info(
                "Forced acquisition of %s database connection after timeout",
                connection_type,
            )

        with _connection_lock:
            if connection_type == "main":
                _active_main_connections += 1
            else:
                _active_revision_connections += 1

        logger.debug(
            "Acquired %s database connection. Active: %s main, %s revision",
            connection_type,
            _active_main_connections,
            _active_revision_connections,
        )

        session = func(*args, **kwargs)
        original_close = session.close

        def patched_close():
            global _active_main_connections, _active_revision_connections

            try:
                return original_close()
            finally:
                semaphore.release()

                with _connection_lock:
                    if connection_type == "main":
                        _active_main_connections = max(0, _active_main_connections - 1)
                    else:
                        _active_revision_connections = max(
                            0, _active_revision_connections - 1
                        )

                logger.debug(
                    "Released %s database connection. Active: %s main, %s revision",
                    connection_type,
                    _active_main_connections,
                    _active_revision_connections,
                )

        session.close = patched_close
        return session

    return wrapper


# -----------------------------------------------------------------------------
# Main DatabaseConfig
# -----------------------------------------------------------------------------
class DatabaseConfig:
    """
    Primary database configuration.

    Design goals:
    - PostgreSQL-first
    - single engine / pool per process when used via get_db_config()
    - request-aware PostgreSQL connection tracing
    - SQLite fallback retained for compatibility
    """

    def __init__(self):
        env_url = os.getenv("DATABASE_URL")
        if env_url and env_url.strip():
            self.main_database_url = env_url.strip()
            logger.info(
                "[CONFIG] Using DATABASE_URL from environment: %s",
                self.main_database_url,
            )
        else:
            self.main_database_url = DATABASE_URL
            logger.info(
                "[CONFIG] Using DATABASE_URL from config.py: %s",
                self.main_database_url,
            )

        self.is_postgresql = _is_postgresql_url(self.main_database_url)

        try:
            if self.is_postgresql:
                logger.info(
                    "[CONFIG] Creating PostgreSQL engine: %s",
                    self.main_database_url,
                )

                self.main_engine = create_engine(
                    self.main_database_url,
                    pool_size=5,
                    max_overflow=5,
                    pool_pre_ping=True,
                    pool_recycle=1800,
                    echo=False,
                    future=True,
                    connect_args={
                        "application_name": "emtac_app",
                        "options": "-c timezone=utc",
                    },
                )

                self._apply_postgresql_settings(self.main_engine)
                self._apply_postgresql_request_tracing(self.main_engine)

            else:
                logger.warning(
                    "[CONFIG] Non-PostgreSQL URL detected. Using SQLite: %s",
                    self.main_database_url,
                )
                self.main_engine = create_engine(
                    self.main_database_url,
                    future=True,
                )
                self._apply_sqlite_pragmas(self.main_engine)

        except Exception as exc:
            logger.error(
                "[CONFIG] Main database engine creation failed, switching to SQLite. Error: %s",
                exc,
            )
            self.main_database_url = "sqlite:///emtac.db"
            self.main_engine = create_engine(self.main_database_url, future=True)
            self.is_postgresql = False
            self._apply_sqlite_pragmas(self.main_engine)

        # ORM
        self.MainBase = declarative_base()
        self.MainSession = scoped_session(
            sessionmaker(
                bind=self.main_engine,
                autoflush=False,
                autocommit=False,
                future=True,
            )
        )
        self.MainSessionMaker = sessionmaker(
            bind=self.main_engine,
            autoflush=False,
            autocommit=False,
            future=True,
        )

        # Revision control DB (always SQLite)
        self.revision_control_db_path = REVISION_CONTROL_DB_PATH
        self.revision_control_engine = create_engine(
            f"sqlite:///{self.revision_control_db_path}",
            future=True,
        )
        self.RevisionControlBase = declarative_base()
        self.RevisionControlSession = scoped_session(
            sessionmaker(
                bind=self.revision_control_engine,
                autoflush=False,
                autocommit=False,
                future=True,
            )
        )
        self.RevisionControlSessionMaker = sessionmaker(
            bind=self.revision_control_engine,
            autoflush=False,
            autocommit=False,
            future=True,
        )
        self._apply_sqlite_pragmas(self.revision_control_engine)

        db_type = "PostgreSQL" if self.is_postgresql else "SQLite"
        logger.info(
            "DatabaseConfig initialized with %s, connection limiting: %s, max connections: %s",
            db_type,
            CONNECTION_LIMITING_ENABLED,
            MAX_CONCURRENT_CONNECTIONS,
        )

    # -------------------------------------------------------------------------
    # Session factories
    # -------------------------------------------------------------------------
    @with_connection_limiting
    def get_main_session(self):
        return self.MainSession()

    @with_connection_limiting
    def get_revision_control_session(self):
        return self.RevisionControlSession()

    # Backwards compatibility alias
    def get_session(self):
        return self.get_main_session()

    @contextmanager
    def main_session(self):
        session = self.get_main_session() if CONNECTION_LIMITING_ENABLED else self.MainSessionMaker()
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Database session error: %s", exc)
            raise
        finally:
            session.close()

    @contextmanager
    def revision_control_session(self):
        session = (
            self.get_revision_control_session()
            if CONNECTION_LIMITING_ENABLED
            else self.RevisionControlSessionMaker()
        )
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Revision control database session error: %s", exc)
            raise
        finally:
            session.close()

    # Aliases for clarity / compatibility
    @contextmanager
    def get_main_session_context(self):
        with self.main_session() as session:
            yield session

    @contextmanager
    def get_revision_control_session_context(self):
        with self.revision_control_session() as session:
            yield session

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------
    def get_main_base(self):
        return self.MainBase

    def get_revision_control_base(self):
        return self.RevisionControlBase

    def get_main_session_registry(self):
        return self.MainSession

    def get_revision_control_session_registry(self):
        return self.RevisionControlSession

    def get_engine(self):
        return self.main_engine

    def get_database_url(self):
        return self.main_database_url

    def get_connection_stats(self) -> Dict[str, Any]:
        return {
            "database_type": "PostgreSQL" if self.is_postgresql else "SQLite",
            "database_url": self.main_database_url,
            "connection_limiting_enabled": CONNECTION_LIMITING_ENABLED,
            "max_concurrent_connections": MAX_CONCURRENT_CONNECTIONS,
            "active_main_connections": _active_main_connections,
            "active_revision_connections": _active_revision_connections,
            "connection_timeout": CONNECTION_TIMEOUT,
        }

    # -------------------------------------------------------------------------
    # Engine event hooks
    # -------------------------------------------------------------------------
    def _apply_sqlite_pragmas(self, engine):
        def set_sqlite_pragmas(dbapi_connection, connection_record):
            try:
                if dbapi_connection.__class__.__module__.startswith("sqlite3"):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL;")
                    cursor.execute("PRAGMA synchronous=NORMAL;")
                    cursor.execute("PRAGMA temp_store=MEMORY;")
                    cursor.execute("PRAGMA foreign_keys=ON;")
                    cursor.close()
            except Exception as exc:
                logger.warning("Skipped SQLite PRAGMAs: %s", exc)

        event.listen(engine, "connect", set_sqlite_pragmas)

    def _apply_postgresql_settings(self, engine):
        def set_postgresql_settings(dbapi_connection, connection_record):
            with dbapi_connection.cursor() as cursor:
                cursor.execute("SET timezone = 'UTC'")
                cursor.execute("SET statement_timeout = '30s'")
                cursor.execute("SET idle_in_transaction_session_timeout = '60s'")

        event.listen(engine, "connect", set_postgresql_settings)

    def _apply_postgresql_request_tracing(self, engine):
        """
        Tag PostgreSQL sessions with request-aware application_name values.

        connect:
            sets a stable base application_name for brand-new DB connections

        checkout:
            refreshes application_name for the current request whenever
            a pooled connection is checked out
        """

        @event.listens_for(engine, "connect")
        def set_base_application_name(dbapi_connection, connection_record):
            try:
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SET application_name = %s", ("emtac_app",))
            except Exception as exc:
                logger.warning(
                    "[DB TRACE] Failed to set base application_name on connect: %s",
                    exc,
                )

        @event.listens_for(engine, "checkout")
        def set_request_application_name(
            dbapi_connection, connection_record, connection_proxy
        ):
            try:
                application_name = build_pg_application_name("emtac_app")
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SET application_name = %s", (application_name,))
            except Exception as exc:
                logger.warning(
                    "[DB TRACE] Failed to set request application_name on checkout: %s",
                    exc,
                )

    # -------------------------------------------------------------------------
    # Full-text search helpers
    # -------------------------------------------------------------------------
    def create_documents_fts(self):
        if self.is_postgresql:
            self._create_postgresql_fts()
        else:
            self._create_sqlite_fts()

    def _create_postgresql_fts(self):
        try:
            with self.main_session() as session:
                logger.info("Setting up PostgreSQL full-text search...")

                fts_tables = [
                    ("part", ["part_number", "description"]),
                    ("image", ["title", "filename"]),
                    ("drawing", ["drw_number", "drw_spare_part_number"]),
                ]

                for table_name, text_columns in fts_tables:
                    try:
                        session.execute(
                            text(
                                f"""
                                ALTER TABLE {table_name}
                                ADD COLUMN IF NOT EXISTS search_vector tsvector
                                """
                            )
                        )

                        session.execute(
                            text(
                                f"""
                                CREATE INDEX IF NOT EXISTS idx_{table_name}_fts
                                ON {table_name} USING gin(search_vector)
                                """
                            )
                        )

                        columns_concat = " || ' ' || ".join(
                            [f"COALESCE({col}, '')" for col in text_columns]
                        )

                        session.execute(
                            text(
                                f"""
                                CREATE OR REPLACE FUNCTION update_{table_name}_search_vector()
                                RETURNS trigger AS $$
                                BEGIN
                                    NEW.search_vector := to_tsvector('english', {columns_concat});
                                    RETURN NEW;
                                END;
                                $$ LANGUAGE plpgsql;
                                """
                            )
                        )

                        logger.info(
                            "PostgreSQL FTS setup completed for table: %s",
                            table_name,
                        )

                    except Exception as exc:
                        logger.warning(
                            "Could not setup FTS for table %s: %s",
                            table_name,
                            exc,
                        )

                logger.info("PostgreSQL full-text search setup completed")

        except Exception as exc:
            logger.error("Error setting up PostgreSQL full-text search: %s", exc)

    def _create_sqlite_fts(self):
        try:
            with self.main_session() as session:
                session.execute(
                    text(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts "
                        "USING FTS5(title, content)"
                    )
                )
                logger.info("SQLite FTS5 table created")
        except Exception as exc:
            logger.error("Error creating SQLite FTS table: %s", exc)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def test_connection(self) -> Dict[str, Any]:
        try:
            with self.main_session() as session:
                if self.is_postgresql:
                    version_info = session.execute(text("SELECT version()")).scalar()
                    current_user = session.execute(text("SELECT current_user")).scalar()
                    app_name = session.execute(text("SHOW application_name")).scalar()

                    return {
                        "status": "success",
                        "database_type": "PostgreSQL",
                        "version": version_info or "Unknown",
                        "current_user": current_user or "Unknown",
                        "application_name": app_name or "Unknown",
                        "url": self.main_database_url,
                    }

                version_info = session.execute(text("SELECT sqlite_version()")).scalar()
                return {
                    "status": "success",
                    "database_type": "SQLite",
                    "version": version_info or "Unknown",
                    "url": self.main_database_url,
                }

        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "database_type": "PostgreSQL" if self.is_postgresql else "SQLite",
                "url": self.main_database_url,
            }

    def get_unicode_database_url(self):
        base_url = self.get_database_url()

        if "?" in base_url:
            return base_url + "&client_encoding=utf8&connect_timeout=30"
        return base_url + "?client_encoding=utf8&connect_timeout=30"

    def create_unicode_engine(self):
        """
        Compatibility helper. Prefer using the singleton main engine instead of
        creating extra engines in application code.
        """
        logger.warning(
            "create_unicode_engine() called. Prefer get_db_config().get_engine() to avoid extra pools."
        )

        return create_engine(
            self.get_unicode_database_url(),
            connect_args={"client_encoding": "utf8"},
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
            future=True,
        )

# -----------------------------------------------------------------------------
# TrainingDatabaseConfig
# -----------------------------------------------------------------------------
class TrainingDatabaseConfig:
    """
    STRICT training-only database configuration.

    - PostgreSQL ONLY
    - Explicit database/schema
    - Fail-fast behavior
    """

    def __init__(self):
        self.db_name = os.getenv("POSTGRES_TRAIN_DB")
        self.db_user = os.getenv("POSTGRES_TRAIN_USER")
        self.db_pass = os.getenv("POSTGRES_TRAIN_PASSWORD")
        self.db_host = os.getenv("POSTGRES_TRAIN_HOST", "localhost")
        self.db_port = os.getenv("POSTGRES_TRAIN_PORT", "5432")
        self.schema = os.getenv("POSTGRES_TRAIN_SCHEMA", "intent_training")

        missing = [
            key
            for key, value in {
                "POSTGRES_TRAIN_DB": self.db_name,
                "POSTGRES_TRAIN_USER": self.db_user,
                "POSTGRES_TRAIN_PASSWORD": self.db_pass,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing required training DB env vars: {missing}")

        self.database_url = (
            f"postgresql+psycopg2://{self.db_user}:{self.db_pass}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

        logger.info("[TRAINING-DB] Using database URL: %s", self.database_url)

        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            echo=False,
            future=True,
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
                raise RuntimeError(
                    f"Training schema '{self.schema}' does not exist in database '{self.db_name}'"
                )

        logger.info(
            "[TRAINING-DB] Connection verified, schema '%s' active",
            self.schema,
        )

        self.Base = declarative_base(metadata=None)
        self.SessionMaker = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            future=True,
        )

        event.listen(self.engine, "connect", self._set_postgres_session_settings)

    def get_session(self):
        return self.SessionMaker()

    def get_engine(self):
        return self.engine

    def get_base(self):
        return self.Base

    @staticmethod
    def _set_postgres_session_settings(dbapi_connection, connection_record):
        with dbapi_connection.cursor() as cur:
            cur.execute("SET timezone = 'UTC'")
            cur.execute("SET statement_timeout = '0'")
            cur.execute("SET idle_in_transaction_session_timeout = '60s'")


# -----------------------------------------------------------------------------
# Singleton database config
# -----------------------------------------------------------------------------
_DB_CONFIG_SINGLETON: Optional[DatabaseConfig] = None
_DB_CONFIG_LOCK = threading.Lock()


def get_db_config() -> DatabaseConfig:
    global _DB_CONFIG_SINGLETON

    if _DB_CONFIG_SINGLETON is None:
        with _DB_CONFIG_LOCK:
            if _DB_CONFIG_SINGLETON is None:
                _DB_CONFIG_SINGLETON = DatabaseConfig()
                logger.info("[CONFIG] DatabaseConfig singleton created")

    return _DB_CONFIG_SINGLETON