from typing import Dict, List

from modules.emtac_ai.search.expanders.base import BaseSearchExpander
from modules.emtac_ai.search.expanders.search_expansion_result import SearchExpansionResult

from modules.services.part_service import PartService


class PartsSearchExpander(BaseSearchExpander):
    """
    PARTS SEARCH EXPANDER

    Given Part IDs:
      → primary: parts
      → context: positions, problems, tasks, drawings
    """

    intent = "parts"

    def __init__(self, part_service: PartService = None):
        self.part_service = part_service or PartService()

    # --------------------------------------------------
    # ENTRY POINT (ID-DRIVEN)
    # --------------------------------------------------
    def expand(self, query: str, entities: Dict) -> SearchExpansionResult:
        """
        NOTE:
        Resolver should normally be used first.
        This method exists for symmetry / fallback.
        """
        parts = self.part_service.find(
            search_text=query,
            use_fts=True,
            limit=10,
        )

        if not parts:
            return SearchExpansionResult(intent=self.intent)

        return self.expand_from_part_ids([p.id for p in parts])

    # --------------------------------------------------
    # PRIMARY GRAPH EXPANSION
    # --------------------------------------------------
    def expand_from_part_ids(self, part_ids: List[int]) -> SearchExpansionResult:
        result = SearchExpansionResult(intent=self.intent)

        parts = []
        positions = []
        problems = []
        tasks = []
        drawings = []

        for part_id in part_ids:
            part = self.part_service.get(part_id)
            if not part:
                continue

            parts.append(part)

            related = self.part_service.find_related(part_id)
            if not related:
                continue

            downward = related.get("downward", {})

            positions.extend(downward.get("positions", []))
            problems.extend(downward.get("problems", []))
            tasks.extend(downward.get("tasks", []))
            drawings.extend(downward.get("drawings", []))

        # --------------------------------------------------
        # De-duplicate
        # --------------------------------------------------
        parts = list({p.id: p for p in parts}.values())
        positions = list({p.id: p for p in positions}.values())
        problems = list({p.id: p for p in problems}.values())
        tasks = list({p.id: p for p in tasks}.values())
        drawings = list({d.id: d for d in drawings}.values())

        # --------------------------------------------------
        # Assemble result
        # --------------------------------------------------
        result.add_primary("parts", parts)
        result.add_context("positions", positions)
        result.add_context("problems", problems)
        result.add_context("tasks", tasks)
        result.add_context("drawings", drawings)

        result.metadata = {
            "part_ids": part_ids,
            "part_count": len(parts),
            "position_count": len(positions),
            "problem_count": len(problems),
            "task_count": len(tasks),
            "drawing_count": len(drawings),
        }

        return result
