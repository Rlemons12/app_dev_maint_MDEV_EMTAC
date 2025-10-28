# modules/configuration/config_env.py
import os
import threading
import time
from functools import wraps
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL, REVISION_CONTROL_DB_PATH
from modules.configuration.log_config import logger

# Global environment variable that can be set to enable/disable connection limiting
CONNECTION_LIMITING_ENABLED = False

# Global maximum concurrent connections - can also be set via environment variable
MAX_CONCURRENT_CONNECTIONS = int(os.environ.get('MAX_DB_CONNECTIONS', '10'))  # Increased for PostgreSQL

# Global timeout for acquiring a database connection (in seconds)
CONNECTION_TIMEOUT = int(os.environ.get('DB_CONNECTION_TIMEOUT', '60'))

# Global semaphores for database connection limiting
_main_db_semaphore = threading.Semaphore(MAX_CONCURRENT_CONNECTIONS)
_revision_db_semaphore = threading.Semaphore(MAX_CONCURRENT_CONNECTIONS)

# Global counter for active connections (for monitoring purposes)
_active_main_connections = 0
_active_revision_connections = 0
_connection_lock = threading.Lock()


def _is_postgresql_url(database_url: str) -> bool:
    """
    Check if the given database URL is a PostgreSQL connection string.
    Handles both standard and driver-specific prefixes (e.g., psycopg2, asyncpg).
    """
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
    Decorator to apply connection limiting to a function that creates a database session.
    This allows us to maintain backward compatibility with existing code.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        global _active_main_connections, _active_revision_connections

        # Only apply limiting if enabled
        if not CONNECTION_LIMITING_ENABLED:
            return func(*args, **kwargs)

        # Determine which semaphore to use based on the function name
        if 'main' in func.__name__.lower():
            semaphore = _main_db_semaphore
            connection_type = "main"
        else:
            semaphore = _revision_db_semaphore
            connection_type = "revision"

        # Try to acquire semaphore with timeout
        start_time = time.time()
        acquired = False

        while not acquired and (time.time() - start_time < CONNECTION_TIMEOUT):
            acquired = semaphore.acquire(blocking=False)
            if acquired:
                break
            # Log waiting status every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                logger.debug(f"Waiting for {connection_type} database connection ({int(time.time() - start_time)}s)...")
            time.sleep(0.1)  # Short sleep to prevent CPU spinning

        if not acquired:
            logger.warning(f"Timeout waiting for {connection_type} database connection after {CONNECTION_TIMEOUT}s")
            # Force acquire the semaphore - might cause issues but better than deadlock
            semaphore.acquire(blocking=True)
            logger.info(f"Forced acquisition of {connection_type} database connection after timeout")

        # Update connection counter for monitoring
        with _connection_lock:
            if connection_type == "main":
                _active_main_connections += 1
            else:
                _active_revision_connections += 1

        # Log connection acquisition
        logger.debug(
            f"Acquired {connection_type} database connection. Active: {_active_main_connections} main, {_active_revision_connections} revision")

        # Create the session
        session = func(*args, **kwargs)

        # Patch the session's close method to release the semaphore
        original_close = session.close

        def patched_close():
            global _active_main_connections, _active_revision_connections
            # Call the original close method
            result = original_close()

            # Release the semaphore
            semaphore.release()

            # Update connection counter
            with _connection_lock:
                if connection_type == "main":
                    _active_main_connections = max(0, _active_main_connections - 1)  # Prevent negative counts
                else:
                    _active_revision_connections = max(0, _active_revision_connections - 1)

            # Log connection release
            logger.debug(
                f"Released {connection_type} database connection. Active: {_active_main_connections} main, {_active_revision_connections} revision")

            return result

        # Replace the session's close method with our patched version
        session.close = patched_close

        return session

    return wrapper


class DatabaseConfig:

    def __init__(self):
        """
        Initialize the DatabaseConfig — prioritizing PostgreSQL if available,
        and falling back to SQLite only if connection fails.
        """
        # --- PRIORITY ORDER ---
        # 1. DATABASE_URL from environment (highest)
        # 2. DATABASE_URL from config.py fallback
        # -------------------------------------------
        env_url = os.getenv("DATABASE_URL")
        if env_url and env_url.strip():
            self.main_database_url = env_url.strip()
            logger.info(f"[CONFIG] Using DATABASE_URL from environment: {self.main_database_url}")
        else:
            self.main_database_url = DATABASE_URL
            logger.info(f"[CONFIG] Using DATABASE_URL from config.py: {self.main_database_url}")

        # Detect if PostgreSQL or SQLite
        self.is_postgresql = _is_postgresql_url(self.main_database_url)

        # --- ENGINE INITIALIZATION ---
        try:
            if self.is_postgresql:
                logger.info(f"[CONFIG] Attempting PostgreSQL connection: {self.main_database_url}")
                self.main_engine = create_engine(
                    self.main_database_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False,
                    connect_args={
                        "application_name": "emtac_app",
                        "options": "-c timezone=utc"
                    }
                )
                # Test the connection explicitly
                with self.main_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("[CONFIG] PostgreSQL connection verified successfully.")
            else:
                logger.warning(f"[CONFIG] Non-PostgreSQL URL detected. Using SQLite: {self.main_database_url}")
                self.main_engine = create_engine(self.main_database_url)

        except Exception as e:
            logger.error(f"[CONFIG] PostgreSQL connection failed, switching to SQLite. Error: {e}")
            self.main_database_url = "sqlite:///emtac.db"
            self.main_engine = create_engine(self.main_database_url)
            self.is_postgresql = False

        # --- ORM CONFIGURATION ---
        self.MainBase = declarative_base()
        self.MainSession = scoped_session(sessionmaker(bind=self.main_engine))
        self.MainSessionMaker = sessionmaker(bind=self.main_engine)

        # --- REVISION CONTROL DB (always SQLite) ---
        self.revision_control_db_path = REVISION_CONTROL_DB_PATH
        self.revision_control_engine = create_engine(f"sqlite:///{self.revision_control_db_path}")
        self.RevisionControlBase = declarative_base()
        self.RevisionControlSession = scoped_session(sessionmaker(bind=self.revision_control_engine))
        self.RevisionControlSessionMaker = sessionmaker(bind=self.revision_control_engine)

        # --- APPLY DATABASE-SPECIFIC SETTINGS ---
        if self.is_postgresql:
            self._apply_postgresql_settings(self.main_engine)
        else:
            self._apply_sqlite_pragmas(self.main_engine)

        # Always apply SQLite pragmas to the revision control DB
        self._apply_sqlite_pragmas(self.revision_control_engine)

        db_type = "PostgreSQL" if self.is_postgresql else "SQLite"
        logger.info(
            f"DatabaseConfig initialized with {db_type}, "
            f"connection limiting: {CONNECTION_LIMITING_ENABLED}, "
            f"max connections: {MAX_CONCURRENT_CONNECTIONS}"
        )

    @with_connection_limiting
    def get_main_session(self):
        """
        Return a session from the main database session factory.
        If connection limiting is enabled, this will automatically
        use a semaphore to limit concurrent connections.
        """
        return self.MainSession()

    @with_connection_limiting
    def get_revision_control_session(self):
        """
        Return a session from the revision control database session factory.
        If connection limiting is enabled, this will automatically
        use a semaphore to limit concurrent connections.
        """
        return self.RevisionControlSession()

    @contextmanager
    def main_session(self):
        """
        Context manager for main database sessions.
        Usage:
            with db_config.main_session() as session:
                # use session here
                session.query(...)
        """
        if CONNECTION_LIMITING_ENABLED:
            session = self.get_main_session()  # This will handle connection limiting
        else:
            session = self.MainSessionMaker()

        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    @contextmanager
    def revision_control_session(self):
        """
        Context manager for revision control database sessions.
        Usage:
            with db_config.revision_control_session() as session:
                # use session here
                session.query(...)
        """
        if CONNECTION_LIMITING_ENABLED:
            session = self.get_revision_control_session()  # This will handle connection limiting
        else:
            session = self.RevisionControlSessionMaker()

        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    # Alternative method names for clarity
    @contextmanager
    def get_main_session_context(self):
        """Alias for main_session() context manager."""
        with self.main_session() as session:
            yield session

    @contextmanager
    def get_revision_control_session_context(self):
        """Alias for revision_control_session() context manager."""
        with self.revision_control_session() as session:
            yield session

    def get_main_base(self):
        return self.MainBase

    def get_revision_control_base(self):
        return self.RevisionControlBase

    def get_main_session_registry(self):
        """Return the scoped_session registry for the main database."""
        return self.MainSession

    def get_revision_control_session_registry(self):
        """Return the scoped_session registry for the revision control database."""
        return self.RevisionControlSession

    def get_engine(self):
        """Return the main database engine. Added for compatibility with setup scripts."""
        return self.main_engine

    def get_connection_stats(self):
        """
        Return statistics about database connections.
        Useful for monitoring and debugging connection issues.
        """
        return {
            'database_type': 'PostgreSQL' if self.is_postgresql else 'SQLite',
            'database_url': self.main_database_url,
            'connection_limiting_enabled': CONNECTION_LIMITING_ENABLED,
            'max_concurrent_connections': MAX_CONCURRENT_CONNECTIONS,
            'active_main_connections': _active_main_connections,
            'active_revision_connections': _active_revision_connections,
            'connection_timeout': CONNECTION_TIMEOUT
        }

    def _apply_sqlite_pragmas(self, engine):
        """Apply SQLite-specific PRAGMA settings."""

        def set_sqlite_pragmas(dbapi_connection, connection_record):
            """Apply SQLite-specific PRAGMA settings (ignored for PostgreSQL)."""
            try:
                # Detect database type
                if dbapi_connection.__class__.__module__.startswith("sqlite3"):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL;")
                    cursor.execute("PRAGMA synchronous=NORMAL;")
                    cursor.execute("PRAGMA temp_store=MEMORY;")
                    cursor.execute("PRAGMA foreign_keys=ON;")
                    cursor.close()
            except Exception as e:
                # Safe fallback — never crash PostgreSQL connections
                import logging
                logging.getLogger(__name__).warning(f"Skipped SQLite PRAGMAs: {e}")

        event.listen(engine, 'connect', set_sqlite_pragmas)

    def _apply_postgresql_settings(self, engine):
        """Apply PostgreSQL-specific connection settings."""

        def set_postgresql_settings(dbapi_connection, connection_record):
            with dbapi_connection.cursor() as cursor:
                # Set timezone to UTC
                cursor.execute("SET timezone = 'UTC'")
                # Set statement timeout (30 seconds)
                cursor.execute("SET statement_timeout = '30s'")
                # Set idle timeout (to clean up idle connections)
                cursor.execute("SET idle_in_transaction_session_timeout = '60s'")

        event.listen(engine, 'connect', set_postgresql_settings)

    def create_documents_fts(self):
        """
        Create full-text search capabilities.
        Uses different approaches for PostgreSQL vs SQLite.
        """
        if self.is_postgresql:
            # Use PostgreSQL's built-in full-text search
            self._create_postgresql_fts()
        else:
            # Use SQLite FTS5 (original functionality)
            self._create_sqlite_fts()

    def _create_postgresql_fts(self):
        """Create PostgreSQL full-text search setup."""
        try:
            with self.main_session() as session:
                logger.info("Setting up PostgreSQL full-text search...")

                # Add tsvector columns to tables that need full-text search
                fts_tables = [
                    ('part', ['part_number', 'description']),
                    ('image', ['title', 'filename']),
                    ('drawing', ['drw_number', 'drw_spare_part_number'])
                ]

                for table_name, text_columns in fts_tables:
                    try:
                        # Add search vector column if it doesn't exist
                        session.execute(text(f"""
                            ALTER TABLE {table_name} 
                            ADD COLUMN IF NOT EXISTS search_vector tsvector
                        """))

                        # Create GIN index for full-text search
                        session.execute(text(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_fts 
                            ON {table_name} USING gin(search_vector)
                        """))

                        # Create a function to update the search vector (optional)
                        # This could be used with triggers to auto-update search vectors
                        columns_concat = " || ' ' || ".join([f"COALESCE({col}, '')" for col in text_columns])
                        session.execute(text(f"""
                            CREATE OR REPLACE FUNCTION update_{table_name}_search_vector() 
                            RETURNS trigger AS $$
                            BEGIN
                                NEW.search_vector := to_tsvector('english', {columns_concat});
                                RETURN NEW;
                            END;
                            $$ LANGUAGE plpgsql;
                        """))

                        logger.info(f"PostgreSQL FTS setup completed for table: {table_name}")

                    except Exception as e:
                        logger.warning(f"Could not setup FTS for table {table_name}: {e}")

                logger.info("PostgreSQL full-text search setup completed")

        except Exception as e:
            logger.error(f"Error setting up PostgreSQL full-text search: {e}")

    def _create_sqlite_fts(self):
        """Create SQLite FTS5 virtual table (original functionality)."""
        try:
            with self.main_session() as session:
                session.execute(
                    text(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts "
                        "USING FTS5(title, content)"
                    )
                )
                logger.info("SQLite FTS5 table created")
        except Exception as e:
            logger.error(f"Error creating SQLite FTS table: {e}")

    def test_connection(self):
        """Test the database connection and return basic info."""
        try:
            with self.main_session() as session:
                if self.is_postgresql:
                    result = session.execute(text("SELECT version()")).fetchone()
                    version_info = result[0] if result else "Unknown"

                    # Get current user
                    result = session.execute(text("SELECT current_user")).fetchone()
                    current_user = result[0] if result else "Unknown"

                    return {
                        'status': 'success',
                        'database_type': 'PostgreSQL',
                        'version': version_info,
                        'current_user': current_user,
                        'url': self.main_database_url
                    }
                else:
                    # SQLite
                    result = session.execute(text("SELECT sqlite_version()")).fetchone()
                    version_info = result[0] if result else "Unknown"

                    return {
                        'status': 'success',
                        'database_type': 'SQLite',
                        'version': version_info,
                        'url': self.main_database_url
                    }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'database_type': 'PostgreSQL' if self.is_postgresql else 'SQLite',
                'url': self.main_database_url
            }

    def get_unicode_database_url(self):
        """Get database URL with proper Unicode encoding parameters."""
        base_url = self.get_database_url()  # Your existing method

        # Add encoding parameters if not already present
        if '?' in base_url:
            return base_url + "&client_encoding=utf8&connect_timeout=30"
        else:
            return base_url + "?client_encoding=utf8&connect_timeout=30"

    def create_unicode_engine(self):
        """Create SQLAlchemy engine with proper Unicode support."""
        from sqlalchemy import create_engine

        engine = create_engine(
            self.get_unicode_database_url(),
            # Ensure connection uses UTF-8
            connect_args={
                'client_encoding': 'utf8',
            },
            # Connection pool settings
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )

        return engine