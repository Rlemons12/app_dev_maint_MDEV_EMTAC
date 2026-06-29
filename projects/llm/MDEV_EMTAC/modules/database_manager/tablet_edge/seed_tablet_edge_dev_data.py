"""
Seed development tablet records for the EMTAC Tablet Edge Agent.

Run from project root:
    python -m modules.database_manager.tablet_edge.seed_tablet_edge_dev_data
"""

from __future__ import annotations

import logging
import os
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


logger = logging.getLogger("tablet_edge_seed_dev")


DEV_TABLETS = [
    {
        "tablet_uid": "00000000-0000-0000-0000-000000000101",
        "tablet_name": "EMTAC-GALAXY-DEV-01",
        "device_make": "Samsung",
        "device_model": "Galaxy Tablet",
        "android_version": "unknown",
        "app_version": "0.1.0-dev",
        "assigned_area": "Development",
        "assigned_station": "Dev Bench",
        "assigned_role": "maintenance_tablet",
    },
    {
        "tablet_uid": "00000000-0000-0000-0000-000000000102",
        "tablet_name": "EMTAC-LENOVO-DEV-01",
        "device_make": "Lenovo",
        "device_model": "Lenovo Tablet",
        "android_version": "unknown",
        "app_version": "0.1.0-dev",
        "assigned_area": "Development",
        "assigned_station": "Dev Bench",
        "assigned_role": "maintenance_tablet",
    },
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_engine() -> Engine:
    try:
        from modules.database_manager.tablet_edge.create_tablet_edge_schema import get_engine as get_project_engine
        return get_project_engine()
    except Exception as exc:
        logger.warning("Project engine helper failed: %s", exc)

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not set and project engine helper failed.")

    return create_engine(database_url)


def seed_dev_tablets() -> None:
    engine = get_engine()

    sql = text(
        """
        INSERT INTO tablet_edge.tablet_device (
            tablet_uid,
            tablet_name,
            device_make,
            device_model,
            android_version,
            app_version,
            assigned_area,
            assigned_station,
            assigned_role,
            last_seen_at,
            is_active
        )
        VALUES (
            :tablet_uid,
            :tablet_name,
            :device_make,
            :device_model,
            :android_version,
            :app_version,
            :assigned_area,
            :assigned_station,
            :assigned_role,
            NOW(),
            TRUE
        )
        ON CONFLICT (tablet_uid)
        DO UPDATE SET
            tablet_name = EXCLUDED.tablet_name,
            device_make = EXCLUDED.device_make,
            device_model = EXCLUDED.device_model,
            android_version = EXCLUDED.android_version,
            app_version = EXCLUDED.app_version,
            assigned_area = EXCLUDED.assigned_area,
            assigned_station = EXCLUDED.assigned_station,
            assigned_role = EXCLUDED.assigned_role,
            last_seen_at = NOW(),
            is_active = TRUE,
            updated_at = NOW()
        """
    )

    with engine.begin() as conn:
        for tablet in DEV_TABLETS:
            UUID(tablet["tablet_uid"])
            conn.execute(sql, tablet)
            logger.info("Seeded dev tablet: %s", tablet["tablet_name"])

    logger.info("Development tablet seed completed.")


def main() -> None:
    configure_logging()
    seed_dev_tablets()


if __name__ == "__main__":
    main()
