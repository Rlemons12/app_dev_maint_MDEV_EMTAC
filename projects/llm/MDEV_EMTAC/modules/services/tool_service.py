# modules/services/tool_service.py

from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id, error_id, warning_id, with_request_id
)
from modules.emtacdb.emtacdb_fts import (
    Tool,
    ToolImageAssociation,
    ToolPositionAssociation,
    ToolPackage
)


class ToolService:
    """
    Service layer for Tool entities.

    Supports:
        - CRUD: add_to_db, get, find, remove
        - Relationship traversal: images, positions, tasks, packages
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ---------------------------------------------------------
    # CREATE TOOL
    # ---------------------------------------------------------
    @with_request_id
    def add_to_db(
        self,
        name: str,
        size: Optional[str] = None,
        type_: Optional[str] = None,
        material: Optional[str] = None,
        description: Optional[str] = None,
        tool_category_id: Optional[int] = None,
        tool_manufacturer_id: Optional[int] = None,
    ) -> int:

        with self.db_config.main_session() as session:
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
                return tool.id

            except SQLAlchemyError as e:
                error_id(f"ToolService.add_to_db failed: {e}", None)
                raise

    # ---------------------------------------------------------
    # GET TOOL
    # ---------------------------------------------------------
    @with_request_id
    def get(self, tool_id: int) -> Optional[Tool]:
        with self.db_config.main_session() as session:
            try:
                return session.query(Tool).filter_by(id=tool_id).first()
            except SQLAlchemyError as e:
                error_id(f"ToolService.get failed: {e}", None)
                raise

    # ---------------------------------------------------------
    # FIND TOOLS
    # ---------------------------------------------------------
    @with_request_id
    def find(self, **filters) -> List[Tool]:

        with self.db_config.main_session() as session:
            try:
                query = session.query(Tool)

                for f_name, f_val in filters.items():
                    if not f_val:
                        continue

                    # fuzzy match fields
                    if f_name in ["name", "type", "material", "size"]:
                        query = query.filter(getattr(Tool, f_name).ilike(f"%{f_val}%"))
                    else:
                        query = query.filter(getattr(Tool, f_name) == f_val)

                results = query.all()
                info_id(f"ToolService.find returned {len(results)} rows", None)
                return results

            except SQLAlchemyError as e:
                error_id(f"ToolService.find failed: {e}", None)
                raise

    # ---------------------------------------------------------
    # DELETE TOOL
    # ---------------------------------------------------------
    @with_request_id
    def remove(self, tool_id: int) -> bool:

        with self.db_config.main_session() as session:
            try:
                tool = session.query(Tool).filter_by(id=tool_id).first()
                if not tool:
                    warning_id(f"Tool {tool_id} not found for delete", None)
                    return False

                session.delete(tool)
                info_id(f"Tool {tool_id} deleted", None)
                return True

            except SQLAlchemyError as e:
                error_id(f"ToolService.remove failed: {e}", None)
                raise

    # ---------------------------------------------------------
    # RELATIONSHIPS (IMAGES, POSITIONS, PACKAGES, TASKS)
    # ---------------------------------------------------------
    @with_request_id
    def find_related(self, tool_id: int) -> Optional[Dict[str, Any]]:
        """
        Includes:
            - Images (via ToolImageAssociation)
            - Positions (via ToolPositionAssociation)
            - Tasks (via TaskToolAssociation)
            - Packages (via ToolPackage secondary table)
        """

        with self.db_config.main_session() as session:
            try:
                tool = session.query(Tool).filter_by(id=tool_id).first()
                if not tool:
                    return None

                # Expand associations cleanly
                images = [
                    {
                        "association_id": assoc.id,
                        "image_id": assoc.image_id,
                        "image_title": assoc.image.title if assoc.image else None,
                        "image_path": assoc.image.file_path if assoc.image else None,
                        "description": assoc.description
                    }
                    for assoc in tool.tool_image_association
                ]

                positions = [
                    {
                        "association_id": assoc.id,
                        "position_id": assoc.position_id,
                        "description": assoc.description
                    }
                    for assoc in tool.tool_position_association
                ]

                packages = [
                    {
                        "package_id": p.id,
                        "package_name": p.name if hasattr(p, "name") else None
                    }
                    for p in tool.tool_packages
                ]

                tasks = [
                    {
                        "task_association_id": assoc.id,
                        "task_id": assoc.task_id,
                        "role": assoc.role if hasattr(assoc, "role") else None
                    }
                    for assoc in tool.tool_tasks
                ]

                return {
                    "tool": {
                        "id": tool.id,
                        "name": tool.name,
                        "type": tool.type,
                        "size": tool.size,
                        "material": tool.material,
                        "description": tool.description,
                    },
                    "downward": {
                        "images": images,
                        "positions": positions,
                        "packages": packages,
                        "tasks": tasks,
                    }
                }

            except SQLAlchemyError as e:
                error_id(f"ToolService.find_related failed: {e}", None)
                raise
