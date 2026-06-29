# modules/services/tool_position_association_service.py

from typing import List

from sqlalchemy.orm import Session
from sqlalchemy import select, func

from modules.configuration.log_config import (
    debug_id,
    error_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import (
    Tool,
    ToolPositionAssociation,
)


class ToolPositionAssociationService:
    """
    Service for Tool ↔ Position associations.

    ✔ No session ownership
    ✔ No DatabaseConfig dependency
    ✔ No transaction control
    ✔ Orchestrator controls session lifecycle
    ✔ Safe for RAG / UI / Enrichment
    """

    # --------------------------------------------------
    # CORE QUERIES
    # --------------------------------------------------

    @with_request_id
    def get_tools_for_positions(
        self,
        session: Session,
        position_ids: List[int],
        *,
        request_id: str = None,
    ) -> List[Tool]:
        """
        Return Tool objects associated with given positions.
        """

        if not position_ids:
            return []

        try:
            stmt = (
                select(Tool)
                .join(
                    ToolPositionAssociation,
                    ToolPositionAssociation.tool_id == Tool.id,
                )
                .where(ToolPositionAssociation.position_id.in_(position_ids))
                .distinct()
            )

            tools = session.execute(stmt).scalars().all()

            debug_id(
                f"[ToolPositionAssociationService] "
                f"Resolved {len(tools)} tools for {len(position_ids)} positions",
                request_id,
            )

            return tools

        except Exception as e:
            error_id(
                f"get_tools_for_positions failed: {e}",
                request_id,
                exc_info=True,
            )
            raise  # 🚨 Do NOT swallow errors

    # --------------------------------------------------
    # COUNT (USED BY LIGHTWEIGHT SUMMARY)
    # --------------------------------------------------

    def count_tools(
        self,
        session: Session,
        position_ids: List[int],
    ) -> int:
        """
        Count distinct tools associated with given positions.

        ✔ Used by ChunkAssociationSearchExtendedService.lightweight_counts
        """

        if not position_ids:
            return 0

        stmt = (
            select(func.count(func.distinct(ToolPositionAssociation.tool_id)))
            .where(ToolPositionAssociation.position_id.in_(position_ids))
        )

        return session.execute(stmt).scalar() or 0

    # --------------------------------------------------
    # POSITION IDS FOR TOOL (REVERSE LOOKUP)
    # --------------------------------------------------

    def get_position_ids_for_tool(
        self,
        session: Session,
        tool_id: int,
    ) -> List[int]:
        """
        Tool → Position IDs

        ✔ Used for reverse chunk lookups
        """

        stmt = (
            select(ToolPositionAssociation.position_id)
            .where(ToolPositionAssociation.tool_id == tool_id)
        )

        return [row[0] for row in session.execute(stmt).all()]
