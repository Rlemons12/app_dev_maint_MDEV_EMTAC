from typing import Optional, List, Dict, Any

from modules.orchestrators.base_orchestrator import BaseOrchestrator

from modules.services.tool_service import ToolService
from modules.services.tool_position_association_service import (
    ToolPositionAssociationService,
)
from modules.services.tool_image_association_service import (
    ToolImageAssociationService,
)

from modules.emtacdb.emtacdb_fts import ToolPositionAssociation


class ToolOrchestrator(BaseOrchestrator):
    """
    Tool domain workflow owner.

    Owns:
    - Tool lifecycle
    - Tool ↔ Position coordination
    - Tool ↔ Image coordination

    Does NOT:
    - Contain business logic
    - Perform persistence outside transaction()
    """

    def __init__(self):
        super().__init__()

        self.tool_service = ToolService()
        self.position_service = ToolPositionAssociationService()
        self.image_service = ToolImageAssociationService()

    # ============================================================
    # CREATE TOOL WITH RELATIONS (ATOMIC)
    # ============================================================

    def create_tool_with_relations(
        self,
        *,
        name: str,
        size: Optional[str] = None,
        type_: Optional[str] = None,
        material: Optional[str] = None,
        description: Optional[str] = None,
        tool_category_id: Optional[int] = None,
        tool_manufacturer_id: Optional[int] = None,
        image_ids: Optional[List[int]] = None,
        position_ids: Optional[List[int]] = None,
    ) -> Optional[int]:

        with self._timed("ToolOrchestrator.create_tool_with_relations"):
            with self.transaction() as session:

                # 1️⃣ Create Tool
                tool = self.tool_service.add_to_db(
                    session=session,
                    name=name,
                    size=size,
                    type_=type_,
                    material=material,
                    description=description,
                    tool_category_id=tool_category_id,
                    tool_manufacturer_id=tool_manufacturer_id,
                )

                self._debug(f"Tool created id={tool.id}")

                # 2️⃣ Associate Images
                if image_ids:
                    for img_id in image_ids:
                        self.image_service.associate(
                            session=session,
                            image_id=img_id,
                            tool_id=tool.id,
                        )

                # 3️⃣ Associate Positions
                if position_ids:
                    for pos_id in position_ids:
                        session.add(
                            ToolPositionAssociation(
                                tool_id=tool.id,
                                position_id=pos_id,
                            )
                        )

                self._info(f"Tool created with relations id={tool.id}")
                return tool.id

    # ============================================================
    # FULL TOOL PROFILE
    # ============================================================

    def get_full_tool_profile(
        self,
        *,
        tool_id: int,
    ) -> Optional[Dict[str, Any]]:

        with self._timed("ToolOrchestrator.get_full_tool_profile"):
            with self.transaction(read_only=True) as session:

                tool = self.tool_service.get(
                    session=session,
                    tool_id=tool_id,
                )

                if not tool:
                    return None

                images = self.image_service.get_images_for_tool(
                    session=session,
                    tool_id=tool_id,
                )

                position_ids = self.position_service.get_position_ids_for_tool(
                    session=session,
                    tool_id=tool_id,
                )

                return {
                    "tool": {
                        "id": tool.id,
                        "name": tool.name,
                        "size": tool.size,
                        "type": tool.type,
                        "material": tool.material,
                        "description": tool.description,
                    },
                    "relations": {
                        "images": images,
                        "positions": position_ids,
                    },
                }

    # ============================================================
    # BULK TOOL ENRICHMENT
    # ============================================================

    def enrich_tools_for_positions(
        self,
        *,
        position_ids: List[int],
    ) -> List[Dict[str, Any]]:

        if not position_ids:
            return []

        with self._timed("ToolOrchestrator.enrich_tools_for_positions"):
            with self.transaction(read_only=True) as session:

                tools = self.position_service.get_tools_for_positions(
                    session=session,
                    position_ids=position_ids,
                )

                enriched = [
                    {
                        "id": tool.id,
                        "name": tool.name,
                        "type": tool.type,
                        "size": tool.size,
                        "material": tool.material,
                    }
                    for tool in tools
                ]

                self._info(f"Enriched {len(enriched)} tools")

                return enriched
