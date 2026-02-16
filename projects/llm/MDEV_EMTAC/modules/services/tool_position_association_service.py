from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import select, func

from modules.configuration.config_env import DatabaseConfig
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

    ✔ Read-only
    ✔ Service-layer safe
    ✔ Used by enrichment, UI, and RAG pipelines
    """

    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config

    # --------------------------------------------------
    # INTERNAL SESSION HANDLING
    # --------------------------------------------------

    def _get_session(self, session: Optional[Session]):
        if session:
            return session, False
        return self.db_config.get_main_session(), True

    # --------------------------------------------------
    # CORE QUERIES
    # --------------------------------------------------

    @with_request_id
    def get_tools_for_positions(
        self,
        position_ids: List[int],
        *,
        session: Optional[Session] = None,
        request_id: Optional[str] = None,
    ) -> List[Tool]:
        """
        Return Tool objects associated with given positions.
        """

        if not position_ids:
            return []

        session, created = self._get_session(session)

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
            return []

        finally:
            if created:
                session.close()

    # --------------------------------------------------
    # COUNT (USED BY LIGHTWEIGHT SUMMARY)
    # --------------------------------------------------

    def count_tools(
        self,
        position_ids: List[int],
        session: Session,
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
        tool_id: int,
        *,
        session: Optional[Session] = None,
    ) -> List[int]:
        """
        Tool → Position IDs

        ✔ Used for reverse chunk lookups
        """

        session, created = self._get_session(session)

        try:
            stmt = (
                select(ToolPositionAssociation.position_id)
                .where(ToolPositionAssociation.tool_id == tool_id)
            )

            return [row[0] for row in session.execute(stmt).all()]

        finally:
            if created:
                session.close()
