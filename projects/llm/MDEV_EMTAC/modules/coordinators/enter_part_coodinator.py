from __future__ import annotations

from typing import Dict, Any

from modules.orchestrators.enter_new_part_orchestrator import (
    EnterNewPartOrchestrator,
)
from modules.configuration.log_config import with_request_id

class EnterNewPartCoordinator:
    """
    Coordinator for Enter New Part workflows.

    Responsibilities:
    - Normalize inputs
    - Call orchestrator
    - No database logic
    """

    def __init__(self) -> None:
        self.orchestrator = EnterNewPartOrchestrator()

    def get_part_form_data(self) -> Dict[str, Any]:
        return self.orchestrator.get_part_form_data()

    @with_request_id
    def get_part_image(self, *, image_id: int):
        logger.info("EnterNewPartCoordinator.get_part_image called")
        return self.orchestrator.get_part_image(image_id=image_id)