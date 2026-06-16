from __future__ import annotations

from dataclasses import dataclass

from listed_server.mcp_coordinator.settings import CoordinatorSettings


@dataclass(frozen=True)
class LegacyPostgresSettings:
    postgres_host: str
    postgres_port: int
    postgres_maintenance_db: str
    postgres_default_db: str
    postgres_db: str
    postgres_default_schema: str
    postgres_admin_user: str
    postgres_admin_password: str
    postgres_read_user: str
    postgres_read_password: str
    postgres_write_user: str
    postgres_write_password: str
    max_read_rows: int


def to_legacy_postgres_settings(settings: CoordinatorSettings) -> LegacyPostgresSettings:
    return LegacyPostgresSettings(
        postgres_host=settings.postgres_host,
        postgres_port=settings.postgres_port,
        postgres_maintenance_db=settings.postgres_maintenance_db,
        postgres_default_db=settings.postgres_default_db,
        postgres_db=settings.postgres_db,
        postgres_default_schema=settings.postgres_default_schema,
        postgres_admin_user=settings.postgres_admin_user,
        postgres_admin_password=settings.postgres_admin_password,
        postgres_read_user=settings.postgres_read_user,
        postgres_read_password=settings.postgres_read_password,
        postgres_write_user=settings.postgres_write_user,
        postgres_write_password=settings.postgres_write_password,
        max_read_rows=settings.max_read_rows,
    )
