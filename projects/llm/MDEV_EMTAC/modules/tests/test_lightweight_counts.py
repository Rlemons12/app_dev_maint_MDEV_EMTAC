# tests/contracts/test_lightweight_counts.py

import pytest

from modules.services.parts_position_image_service import PartsPositionImageService
from modules.services.drawing_position_association_service import DrawingPositionAssociationService
from modules.services.problem_position_association_service import ProblemPositionAssociationService
from modules.services.task_position_association_service import TaskPositionAssociationService
from modules.services.tool_position_association_service import ToolPositionAssociationService


def test_lightweight_count_contract():
    """
    Contract test:
    Services used by lightweight_counts MUST expose count_* methods.
    """

    contract = {
        PartsPositionImageService: "count_parts",
        DrawingPositionAssociationService: "count_drawings",
        ProblemPositionAssociationService: "count_problems",
        TaskPositionAssociationService: "count_tasks",
        ToolPositionAssociationService: "count_tools",
    }

    for service_cls, method_name in contract.items():
        svc = service_cls()
        assert hasattr(
            svc, method_name
        ), f"{service_cls.__name__} is missing {method_name}"
