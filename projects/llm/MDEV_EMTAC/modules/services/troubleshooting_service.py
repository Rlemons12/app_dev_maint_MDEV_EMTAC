# modules/services/troubleshooting_service.py

from typing import List, Optional, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    info_id, error_id, warning_id, debug_id, with_request_id
)

from modules.emtacdb.emtacdb_fts import (
    Problem,
    Solution,
    Task,
    ProblemPositionAssociation
)


class TroubleshootingService:
    """
    High-level service for troubleshooting flows:

        Problem → Solution → Task
              ↘ Images / Documents / Drawings / Parts / Positions

    This class also wraps all operations involving:
        ProblemPositionAssociation
    """

    def __init__(self, db_config: DatabaseConfig = None):
        self.db_config = db_config or DatabaseConfig()

    # ---------------------------------------------------------
    # CRUD (PROBLEM)
    # ---------------------------------------------------------
    @with_request_id
    def add_problem(self, name: str, description: str) -> int:
        """Create or return an existing Problem by name."""
        with self.db_config.main_session() as session:
            return Problem.add_to_db(session, name=name, description=description)

    @with_request_id
    def get_problem(self, problem_id: int) -> Optional[Problem]:
        with self.db_config.main_session() as session:
            return Problem.get_by_id(session, problem_id)

    @with_request_id
    def update_problem(self, problem_id: int, name=None, description=None) -> bool:
        with self.db_config.main_session() as session:
            return Problem.update_problem(session, problem_id, name=name, description=description)

    @with_request_id
    def delete_problem(self, problem_id: int) -> bool:
        with self.db_config.main_session() as session:
            return Problem.delete_problem(session, problem_id)

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------
    @with_request_id
    def search_problems(self, text: str, exact: bool = False, limit: int = 50):
        """Search problems by name or description."""
        with self.db_config.main_session() as session:
            return Problem.search(session, text=text, exact=exact, limit=limit)

    # ---------------------------------------------------------
    # PROBLEM ↔ POSITION association operations
    # ---------------------------------------------------------

    @with_request_id
    def attach_problem_to_position(self, problem_id: int, position_id: int):
        """Create (or return) a ProblemPositionAssociation."""
        with self.db_config.main_session() as session:
            return ProblemPositionAssociation.add_to_db(
                session=session,
                problem_id=problem_id,
                position_id=position_id
            )

    @with_request_id
    def detach_problem_from_position(self, problem_id: int = None,
                                     position_id: int = None,
                                     association_id: int = None) -> bool:
        """Remove a problem-position association."""
        with self.db_config.main_session() as session:
            return ProblemPositionAssociation.delete_association(
                session=session,
                problem_id=problem_id,
                position_id=position_id,
                association_id=association_id
            )

    @with_request_id
    def get_positions_for_problem(self, problem_id: int):
        """Return Position objects for a given Problem."""
        with self.db_config.main_session() as session:
            return ProblemPositionAssociation.get_positions_for_problem(
                session=session,
                problem_id=problem_id
            )

    @with_request_id
    def get_problems_for_position(self, position_id: int):
        """Return Problem objects for a given Position."""
        with self.db_config.main_session() as session:
            return ProblemPositionAssociation.get_problems_for_position(
                session=session,
                position_id=position_id
            )

    @with_request_id
    def get_positions_for_problem_by_hierarchy(self, problem_id: int,
                                               level_filters: Optional[Dict[str, Any]] = None):
        """Return positions but filtered by hierarchy (area, model, equipment group, etc.)."""
        with self.db_config.main_session() as session:
            return ProblemPositionAssociation.get_positions_for_problem_by_hierarchy(
                session=session,
                problem_id=problem_id,
                level_filters=level_filters
            )

    # ---------------------------------------------------------
    # FULL RELATIONSHIP TREE (used by intent_router)
    # ---------------------------------------------------------
    @with_request_id
    def find_related(self, problem_id: int) -> Optional[Dict[str, Any]]:
        """
        Return a full troubleshooting tree for a Problem:

            problem → solutions → tasks
                    → images
                    → drawings
                    → documents
                    → parts
                    → positions
        """
        with self.db_config.main_session() as session:
            problem = Problem.get_by_id(session, problem_id)
            if not problem:
                return None

            debug_id(f"Building relationship tree for Problem {problem_id}")

            # Solutions → Tasks
            solutions = [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "tasks": [
                        {
                            "id": ts.task.id,
                            "name": ts.task.name,
                            "description": ts.task.description
                        }
                        for ts in s.task_solutions
                    ]
                }
                for s in problem.solutions
            ]

            # Images, drawings, docs, parts, positions
            images = [
                {"image_id": assoc.image_id, "description": assoc.description}
                for assoc in problem.image_problem
            ]

            drawings = [
                {"drawing_id": dp.drawing_id}
                for dp in problem.drawing_problem
            ]

            documents = [
                {"document_id": doc.complete_document_id}
                for doc in problem.complete_document_problem
            ]

            parts = [
                {"part_id": pp.part_id}
                for pp in problem.part_problem
            ]

            positions = [
                {"position_id": pos.position_id}
                for pos in problem.problem_position
            ]

            return {
                "problem": {
                    "id": problem.id,
                    "name": problem.name,
                    "description": problem.description,
                },
                "downward": {
                    "solutions": solutions,
                    "positions": positions,
                    "images": images,
                    "documents": documents,
                    "drawings": drawings,
                    "parts": parts,
                }
            }

    # ---------------------------------------------------------
    # Used by intent routing
    # ---------------------------------------------------------
    @with_request_id
    def resolve_query(self, text: str) -> Dict[str, Any]:
        """
        High-level resolver used by the Troubleshooting Router.

        1. Search by text
        2. If 0 matches → return “not found”
        3. If 1 match → return full tree
        4. If many → return choices
        """
        matches = self.search_problems(text)
        count = len(matches)

        if count == 0:
            return {
                "status": "no_match",
                "query": text,
                "message": "No problems found matching query."
            }

        if count > 1:
            return {
                "status": "multiple_matches",
                "query": text,
                "choices": [
                    {"id": p.id, "name": p.name, "description": p.description}
                    for p in matches
                ]
            }

        # Exactly one match → return full troubleshooting guide
        problem = matches[0]
        tree = self.find_related(problem.id)

        return {
            "status": "resolved",
            "query": text,
            "problem": {
                "id": problem.id,
                "name": problem.name,
                "description": problem.description
            },
            "tree": tree
        }
