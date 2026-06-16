from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from services.postgres_server.logger import configure_logging, get_logger


configure_logging()
logger = get_logger("postgres_server.launcher")


def load_project_env() -> Optional[Path]:
    """
    Load the project env file from common locations.
    """
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / ".env",
        project_root / "dev_env" / ".env",
    ]

    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            logger.info("Loaded environment from %s", env_path)
            return env_path

    logger.warning("No environment file found in expected locations")
    return None


def main() -> None:
    load_project_env()

    from services.postgres_server.postgres_server import PostgresServerManager

    manager = PostgresServerManager()
    manager.run_console_menu()


if __name__ == "__main__":
    main()