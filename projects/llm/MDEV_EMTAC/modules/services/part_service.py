# services/part_service.py
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Part
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class PartService:
    """
    Service layer for managing Part entities.

    Provides:
      - `add_to_db`       → Get-or-create a Part, return ID
      - `find`            → Wraps Part.search() for flexible queries
      - `fts_search`      → Full Text Search with relevance scores
      - `get`             → Retrieve a Part by ID
      - `save`            → Create or update a Part
      - `remove`          → Delete a Part by ID
      - `find_or_create`  → Lookup by part_number or create new
      - `find_related`    → Traverse relationships (positions, problems, tasks, drawings)
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ------------------------
    # CORE WRAPPERS
    # ------------------------

    @with_request_id
    def add_to_db(self,
                  part_number: str,
                  name: Optional[str] = None,
                  oem_mfg: Optional[str] = None,
                  model: Optional[str] = None,
                  class_flag: Optional[str] = None,
                  ud6: Optional[str] = None,
                  type_: Optional[str] = None,
                  notes: Optional[str] = None,
                  documentation: Optional[str] = None) -> int:
        """Wrapper for Part.add_part_to_db (get-or-create, returns ID)."""
        with self.db_config.main_session() as session:
            try:
                return Part.add_part_to_db(session,
                                           part_number=part_number,
                                           name=name,
                                           oem_mfg=oem_mfg,
                                           model=model,
                                           class_flag=class_flag,
                                           ud6=ud6,
                                           type_=type_,
                                           notes=notes,
                                           documentation=documentation)
            except SQLAlchemyError as e:
                error_id(f"PartService.add_to_db failed: {e}", None)
                raise

    @with_request_id
    def find(self, **filters) -> List[Part]:
        """Search for Parts using Part.search()."""
        try:
            return Part.search(**filters)
        except SQLAlchemyError as e:
            error_id(f"PartService.find failed: {e}", None)
            raise

    @with_request_id
    def fts_search(self, search_text: str, limit: int = 100) -> List[tuple]:
        """Full Text Search with relevance scores."""
        try:
            return Part.fts_search(search_text=search_text, limit=limit)
        except SQLAlchemyError as e:
            error_id(f"PartService.fts_search failed: {e}", None)
            raise

    @with_request_id
    def get(self, part_id: int) -> Optional[Part]:
        """Retrieve a Part by ID."""
        try:
            return Part.get_by_id(part_id)
        except SQLAlchemyError as e:
            error_id(f"PartService.get failed: {e}", None)
            raise

    # ------------------------
    # CRUD
    # ------------------------

    @with_request_id
    def save(self,
             part_number: str,
             name: Optional[str] = None,
             oem_mfg: Optional[str] = None,
             model: Optional[str] = None,
             class_flag: Optional[str] = None,
             ud6: Optional[str] = None,
             type_: Optional[str] = None,
             notes: Optional[str] = None,
             documentation: Optional[str] = None,
             part_id: Optional[int] = None) -> Part:
        """
        Create or update a Part.
        - If part_id is given → update that record.
        - Otherwise → create new.
        """
        with self.db_config.main_session() as session:
            try:
                if part_id:
                    part = session.query(Part).filter_by(id=part_id).first()
                    if not part:
                        raise ValueError(f"Part with id {part_id} not found")
                    part.part_number = part_number
                    part.name = name
                    part.oem_mfg = oem_mfg
                    part.model = model
                    part.class_flag = class_flag
                    part.ud6 = ud6
                    part.type = type_
                    part.notes = notes
                    part.documentation = documentation
                    info_id(f"Updated Part id={part_id}", None)
                else:
                    part = Part(part_number=part_number,
                                name=name,
                                oem_mfg=oem_mfg,
                                model=model,
                                class_flag=class_flag,
                                ud6=ud6,
                                type=type_,
                                notes=notes,
                                documentation=documentation)
                    session.add(part)
                    info_id(f"Created Part '{part_number}'", None)
                return part
            except SQLAlchemyError as e:
                error_id(f"PartService.save failed: {e}", None)
                raise

    @with_request_id
    def remove(self, part_id: int) -> bool:
        """Delete a Part by ID."""
        with self.db_config.main_session() as session:
            try:
                part = session.query(Part).filter_by(id=part_id).first()
                if part:
                    session.delete(part)
                    info_id(f"Deleted Part id={part_id}", None)
                    return True
                return False
            except SQLAlchemyError as e:
                error_id(f"PartService.remove failed: {e}", None)
                raise

    # ------------------------
    # HELPERS
    # ------------------------

    @with_request_id
    def find_or_create(self, part_number: str, name: Optional[str] = None,
                       oem_mfg: Optional[str] = None, model: Optional[str] = None) -> Part:
        """Find a Part by part_number, or create it if missing."""
        with self.db_config.main_session() as session:
            try:
                part = session.query(Part).filter_by(part_number=part_number).first()
                if part:
                    info_id(f"Found existing Part '{part_number}'", None)
                else:
                    part = Part(part_number=part_number, name=name,
                                oem_mfg=oem_mfg, model=model)
                    session.add(part)
                    session.commit()
                    info_id(f"Created new Part '{part_number}'", None)
                return part
            except SQLAlchemyError as e:
                error_id(f"PartService.find_or_create failed: {e}", None)
                raise

    @with_request_id
    def find_related(self, part_id: int) -> Optional[Dict[str, Any]]:
        """
        Return related entities for a Part:
          - Positions
          - Problems
          - Tasks
          - Drawings
        """
        with self.db_config.main_session() as session:
            try:
                part = session.query(Part).filter_by(id=part_id).first()
                if not part:
                    return None
                return {
                    "part": part,
                    "downward": {
                        "positions": part.part_position_image,
                        "problems": part.part_problem,
                        "tasks": part.part_task,
                        "drawings": part.drawing_part,
                    }
                }
            except SQLAlchemyError as e:
                error_id(f"PartService.find_related failed: {e}", None)
                raise
