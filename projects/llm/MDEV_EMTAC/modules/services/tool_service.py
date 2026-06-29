# modules/services/tool_service.py

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.log_config import (
    info_id,
    error_id,
    warning_id,
    with_request_id,
)

from modules.emtacdb.emtacdb_fts import Tool


class ToolService:
    """
    Service layer for Tool entities.

    ✔ No session ownership
    ✔ No commit/rollback
    ✔ Orchestrator controls transaction
    """

    # ---------------------------------------------------------
    # CREATE TOOL
    # ---------------------------------------------------------

    @with_request_id
    def add_to_db(
        self,
        session: Session,
        *,
        name: str,
        size: Optional[str] = None,
        type_: Optional[str] = None,
        material: Optional[str] = None,
        description: Optional[str] = None,
        tool_category_id: Optional[int] = None,
        tool_manufacturer_id: Optional[int] = None,
    ) -> Tool:

        try:
            tool = Tool(
                name=name,
                size=size,
                type=type_,
                material=material,
                description=description,
                tool_category_id=tool_category_id,
                tool_manufacturer_id=tool_manufacturer_id,
            )
            session.add(tool)
            session.flush()

            info_id(f"Created Tool {name} (id={tool.id})", None)
            return tool

        except SQLAlchemyError as e:
            error_id(f"ToolService.add_to_db failed: {e}", None)
            raise

    # ---------------------------------------------------------
    # GET TOOL
    # ---------------------------------------------------------

    @with_request_id
    def get(self, session: Session, tool_id: int) -> Optional[Tool]:
        return session.query(Tool).filter_by(id=tool_id).first()

    # ---------------------------------------------------------
    # FIND TOOLS
    # ---------------------------------------------------------

    @with_request_id
    def find(self, session: Session, **filters) -> List[Tool]:

        query = session.query(Tool)

        for f_name, f_val in filters.items():
            if not f_val:
                continue

            if f_name in ["name", "type", "material", "size"]:
                query = query.filter(getattr(Tool, f_name).ilike(f"%{f_val}%"))
            else:
                query = query.filter(getattr(Tool, f_name) == f_val)

        return query.all()

    # ---------------------------------------------------------
    # DELETE TOOL
    # ---------------------------------------------------------

    @with_request_id
    def remove(self, session: Session, tool_id: int) -> bool:

        tool = session.query(Tool).filter_by(id=tool_id).first()

        if not tool:
            warning_id(f"Tool {tool_id} not found for delete", None)
            return False

        session.delete(tool)
        info_id(f"Tool {tool_id} deleted", None)
        return True
