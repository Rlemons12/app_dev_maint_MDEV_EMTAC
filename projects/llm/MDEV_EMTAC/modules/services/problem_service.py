# modules/services/problem_service.py

from typing import List, Optional, Dict
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id, debug_id, warning_id, error_id, with_request_id, get_request_id
)
from modules.emtacdb.emtacdb_fts import (
    Problem, Solution, ProblemPositionAssociation,
    ImageProblemAssociation, CompleteDocumentProblemAssociation,
    DrawingProblemAssociation, PartProblemAssociation
)


class ProblemService:
    """Service layer for managing Problem objects and their associations."""

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # ----------------------------
    # CREATE
    # ----------------------------
    @with_request_id
    def add(
        self, name: str, description: str,
        session: Optional[Session] = None,
        request_id: Optional[str] = None
    ) -> Optional[int]:
        """Create a new Problem and return its ID."""
        rid = request_id or get_request_id()
        local_session = None
        if session is None:
            local_session = self.db_config.get_main_session()

        try:
            problem = Problem(name=name, description=description)
            (session or local_session).add(problem)
            (session or local_session).commit()
            info_id(f"Created Problem id={problem.id}, name='{name}'", rid)
            return problem.id
        except SQLAlchemyError as e:
            (session or local_session).rollback()
            error_id(f"Failed to add Problem: {e}", rid)
            return None
        finally:
            if local_session:
                local_session.close()

    # ----------------------------
    # RETRIEVE
    # ----------------------------
    @with_request_id
    def get_by_id(
        self, problem_id: int,
        session: Optional[Session] = None,
        request_id: Optional[str] = None
    ) -> Optional[Problem]:
        """Get a Problem by ID."""
        rid = request_id or get_request_id()
        local_session = None
        if session is None:
            local_session = self.db_config.get_main_session()
        try:
            problem = (session or local_session).query(Problem).filter_by(id=problem_id).first()
            if problem:
                debug_id(f"Found Problem {problem}", rid)
            else:
                warning_id(f"No Problem found with id={problem_id}", rid)
            return problem
        finally:
            if local_session:
                local_session.close()

    @with_request_id
    def list_all(
        self, limit: int = 100,
        session: Optional[Session] = None,
        request_id: Optional[str] = None
    ) -> List[Problem]:
        """List all Problems (with optional limit)."""
        local_session = None
        if session is None:
            local_session = self.db_config.get_main_session()
        try:
            return (session or local_session).query(Problem).limit(limit).all()
        finally:
            if local_session:
                local_session.close()

    # ----------------------------
    # UPDATE
    # ----------------------------
    @with_request_id
    def update(
        self, problem_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        session: Optional[Session] = None,
        request_id: Optional[str] = None
    ) -> bool:
        """Update an existing Problem."""
        rid = request_id or get_request_id()
        local_session = None
        if session is None:
            local_session = self.db_config.get_main_session()
        try:
            problem = (session or local_session).query(Problem).filter_by(id=problem_id).first()
            if not problem:
                warning_id(f"Problem {problem_id} not found for update", rid)
                return False
            if name:
                problem.name = name
            if description:
                problem.description = description
            (session or local_session).commit()
            info_id(f"Updated Problem id={problem_id}", rid)
            return True
        except SQLAlchemyError as e:
            (session or local_session).rollback()
            error_id(f"Failed to update Problem {problem_id}: {e}", rid)
            return False
        finally:
            if local_session:
                local_session.close()

    # ----------------------------
    # DELETE
    # ----------------------------
    @with_request_id
    def delete(
        self, problem_id: int,
        session: Optional[Session] = None,
        request_id: Optional[str] = None
    ) -> bool:
        """Delete a Problem by ID."""
        rid = request_id or get_request_id()
        local_session = None
        if session is None:
            local_session = self.db_config.get_main_session()
        try:
            problem = (session or local_session).query(Problem).filter_by(id=problem_id).first()
            if not problem:
                warning_id(f"Problem {problem_id} not found for deletion", rid)
                return False
            (session or local_session).delete(problem)
            (session or local_session).commit()
            info_id(f"Deleted Problem id={problem_id}", rid)
            return True
        except SQLAlchemyError as e:
            (session or local_session).rollback()
            error_id(f"Failed to delete Problem {problem_id}: {e}", rid)
            return False
        finally:
            if local_session:
                local_session.close()

    # ----------------------------
    # SEARCH
    # ----------------------------
    @with_request_id
    def search(
        self, text: str,
        exact: bool = False,
        limit: int = 50,
        session: Optional[Session] = None,
        request_id: Optional[str] = None
    ) -> List[Problem]:
        """Search Problems by name or description."""
        rid = request_id or get_request_id()
        local_session = None
        if session is None:
            local_session = self.db_config.get_main_session()
        try:
            query = (session or local_session).query(Problem)
            if exact:
                results = query.filter(
                    (Problem.name == text) | (Problem.description == text)
                ).limit(limit).all()
            else:
                results = query.filter(
                    (Problem.name.ilike(f"%{text}%")) | (Problem.description.ilike(f"%{text}%"))
                ).limit(limit).all()
            debug_id(f"Search for '{text}' returned {len(results)} Problems", rid)
            return results
        finally:
            if local_session:
                local_session.close()

    # ----------------------------
    # RELATIONSHIP HELPERS
    # ----------------------------
    @with_request_id
    def get_solutions(self, problem_id: int, session: Optional[Session] = None, request_id: Optional[str] = None) -> List[Solution]:
        """Get all Solutions for a Problem."""
        problem = self.get_by_id(problem_id, session=session, request_id=request_id)
        return problem.solutions if problem else []

    @with_request_id
    def get_positions(self, problem_id: int, session: Optional[Session] = None, request_id: Optional[str] = None) -> List[ProblemPositionAssociation]:
        """Get all position associations for a Problem."""
        problem = self.get_by_id(problem_id, session=session, request_id=request_id)
        return problem.problem_position if problem else []

    @with_request_id
    def get_images(self, problem_id: int, session: Optional[Session] = None, request_id: Optional[str] = None) -> List[ImageProblemAssociation]:
        """Get all image associations for a Problem."""
        problem = self.get_by_id(problem_id, session=session, request_id=request_id)
        return problem.image_problem if problem else []

    @with_request_id
    def get_documents(self, problem_id: int, session: Optional[Session] = None, request_id: Optional[str] = None) -> List[CompleteDocumentProblemAssociation]:
        """Get all complete document associations for a Problem."""
        problem = self.get_by_id(problem_id, session=session, request_id=request_id)
        return problem.complete_document_problem if problem else []

    @with_request_id
    def get_drawings(self, problem_id: int, session: Optional[Session] = None, request_id: Optional[str] = None) -> List[DrawingProblemAssociation]:
        """Get all drawing associations for a Problem."""
        problem = self.get_by_id(problem_id, session=session, request_id=request_id)
        return problem.drawing_problem if problem else []

    @with_request_id
    def get_parts(self, problem_id: int, session: Optional[Session] = None, request_id: Optional[str] = None) -> List[PartProblemAssociation]:
        """Get all part associations for a Problem."""
        problem = self.get_by_id(problem_id, session=session, request_id=request_id)
        return problem.part_problem if problem else []

    # ----------------------------
    # STATS
    # ----------------------------
    @with_request_id
    def count(self, session: Optional[Session] = None, request_id: Optional[str] = None) -> int:
        """Return total number of Problems."""
        local_session = None
        if session is None:
            local_session = self.db_config.get_main_session()
        try:
            total = (session or local_session).query(Problem).count()
            info_id(f"Total Problems: {total}", request_id)
            return total
        finally:
            if local_session:
                local_session.close()

