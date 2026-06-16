from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class PostgresServerConfig:
    """
    Configuration for the local PostgreSQL server controller.
    Values are loaded from environment variables at instance creation time.
    """

    bin_dir: str = field(
        default_factory=lambda: os.getenv(
            "POSTGRES_BIN_DIR",
            r"E:\emtac\databases\postgresql\pgsql\bin",
        )
    )
    data_dir: str = field(
        default_factory=lambda: os.getenv(
            "POSTGRES_DATA_DIR",
            r"E:\emtac\databases\postgresql\pgsql\data",
        )
    )
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "postgres"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))

    @property
    def pg_ctl_path(self) -> str:
        return os.path.join(self.bin_dir, "pg_ctl.exe")

    @property
    def initdb_path(self) -> str:
        return os.path.join(self.bin_dir, "initdb.exe")

    @property
    def psql_path(self) -> str:
        return os.path.join(self.bin_dir, "psql.exe")

    @property
    def log_file(self) -> str:
        return os.path.join(self.data_dir, "server.log")

    @property
    def config_file(self) -> str:
        return os.path.join(self.data_dir, "postgresql.conf")

    @property
    def version_file(self) -> str:
        return os.path.join(self.data_dir, "PG_VERSION")