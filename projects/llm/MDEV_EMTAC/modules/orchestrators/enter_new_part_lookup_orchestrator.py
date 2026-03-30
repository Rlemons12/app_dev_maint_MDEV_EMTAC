from __future__ import annotations

from typing import Dict, Any, List

from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import db_config
from modules.configuration.log_config import (
    logger,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import Model, Position


class EnterNewPartOrchestrator:
    """
    Orchestrator for Enter New Part lookup workflows.

    RESPONSIBILITIES:
    - Own session lifecycle
    - Perform read operations
    - Assemble response-safe payloads
    """

    @with_request_id
    def get_part_form_data(self) -> Dict[str, Any]:
        logger.info("EnterNewPartOrchestrator.get_part_form_data started")

        session = db_config.get_main_session()

        try:
            models = session.query(Model).all()
            positions = session.query(Position).all()

            payload = {
                "models": [
                    {"id": model.id, "name": model.name}
                    for model in models
                ],
                "positions": [
                    {"id": position.id, "name": getattr(position, "name", str(position.id))}
                    for position in positions
                ],
                "status_code": 200,
            }

            logger.info(
                f"EnterNewPartOrchestrator.get_part_form_data success: "
                f"{len(models)} models, {len(positions)} positions"
            )

            return payload

        except SQLAlchemyError as e:
            logger.exception("Database error in get_part_form_data")

            return {
                "error": "Database error",
                "status_code": 500,
            }

        except Exception as e:
            logger.exception("Unexpected error in get_part_form_data")

            return {
                "error": str(e),
                "status_code": 500,
            }

        finally:
            session.close()