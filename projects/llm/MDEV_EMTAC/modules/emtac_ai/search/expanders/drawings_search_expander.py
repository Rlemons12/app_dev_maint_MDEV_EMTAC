from typing import Dict, List

from modules.emtac_ai.search.expanders.base import BaseSearchExpander
from modules.emtac_ai.search.expanders.search_expansion_result import SearchExpansionResult

from modules.services.drawing_service import DrawingService


class DrawingsSearchExpander(BaseSearchExpander):
    """
    DRAWINGS SEARCH EXPANDER

    Given Drawing ID(s):
      → primary: drawings
      → context: positions, problems, tasks, parts
    """

    intent = "drawings"

    def __init__(self, drawing_service: DrawingService = None):
        self.drawing_service = drawing_service or DrawingService()

    # --------------------------------------------------
    # FALLBACK ENTRY (TEXT ONLY)
    # --------------------------------------------------
    def expand(self, query: str, entities: Dict) -> SearchExpansionResult:
        drawings = self.drawing_service.find(
            search_text=query,
            limit=10,
        )

        if not drawings:
            return SearchExpansionResult(intent=self.intent)

        return self.expand_from_drawing_ids([d.id for d in drawings])

    # --------------------------------------------------
    # PRIMARY GRAPH EXPANSION
    # --------------------------------------------------
    def expand_from_drawing_ids(self, drawing_ids: List[int]) -> SearchExpansionResult:
        result = SearchExpansionResult(intent=self.intent)

        drawings = []
        positions = []
        problems = []
        tasks = []
        parts = []

        for drawing_id in drawing_ids:
            drawing = self.drawing_service.get(drawing_id)
            if not drawing:
                continue

            drawings.append(drawing)

            related = self.drawing_service.find_related(drawing_id)
            if not related:
                continue

            downward = related.get("downward", {})

            positions.extend(downward.get("positions", []))
            problems.extend(downward.get("problems", []))
            tasks.extend(downward.get("tasks", []))
            parts.extend(downward.get("parts", []))

        # --------------------------------------------------
        # DE-DUPLICATE
        # --------------------------------------------------
        drawings = list({d.id: d for d in drawings}.values())
        positions = list({p.id: p for p in positions}.values())
        problems = list({p.id: p for p in problems}.values())
        tasks = list({t.id: t for t in tasks}.values())
        parts = list({p.id: p for p in parts}.values())

        # --------------------------------------------------
        # ASSEMBLE RESULT
        # --------------------------------------------------
        result.add_primary("drawings", drawings)
        result.add_context("positions", positions)
        result.add_context("problems", problems)
        result.add_context("tasks", tasks)
        result.add_context("parts", parts)

        result.metadata = {
            "drawing_ids": drawing_ids,
            "drawing_count": len(drawings),
            "position_count": len(positions),
            "problem_count": len(problems),
            "task_count": len(tasks),
            "part_count": len(parts),
        }

        return result
