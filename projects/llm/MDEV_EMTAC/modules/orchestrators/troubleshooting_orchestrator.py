"""
Troubleshooting Orchestrator

Owns troubleshooting workflows:

    - resolve_query
    - build_problem_tree
    - attach/detach flows
    - future AI troubleshooting routing

Transport agnostic.
Transaction controlled.
Session owned by orchestrator.
"""

from typing import Dict, Any, Optional

from modules.orchestrators.base_orchestrator import BaseOrchestrator
from modules.configuration.log_config import with_request_id


class TroubleshootingOrchestrator(BaseOrchestrator):

    # ---------------------------------------------------------
    # High-Level Query Resolution
    # ---------------------------------------------------------

    @with_request_id
    def resolve_query(self, text: str) -> Dict[str, Any]:

        with self._timed("TroubleshootingOrchestrator.resolve_query"):

            self._debug(f"Resolving troubleshooting query: '{text}'")

            # READ WORKFLOW — still controlled by orchestrator
            with self.transaction() as session:

                matches = self.services.troubleshooting.search_problems(
                    text=text,
                    session=session
                )

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
                            {
                                "id": p.id,
                                "name": p.name,
                                "description": p.description
                            }
                            for p in matches
                        ]
                    }

                # Exactly one match
                problem = matches[0]

                tree = self.services.troubleshooting.find_related(
                    problem_id=problem.id,
                    session=session
                )

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

    # ---------------------------------------------------------
    # Problem Tree Builder
    # ---------------------------------------------------------

    @with_request_id
    def build_problem_tree(self, problem_id: int) -> Optional[Dict[str, Any]]:

        with self._timed("TroubleshootingOrchestrator.build_problem_tree"):

            self._debug(f"Building problem tree for id={problem_id}")

            with self.transaction() as session:

                tree = self.services.troubleshooting.find_related(
                    problem_id=problem_id,
                    session=session
                )

                if not tree:
                    self._warning(f"Problem id={problem_id} not found")
                    return None

                return tree

    # ---------------------------------------------------------
    # Problem Management Workflow
    # ---------------------------------------------------------

    @with_request_id
    def create_problem_and_attach_to_position(
        self,
        name: str,
        description: str,
        position_id: int,
    ) -> Dict[str, Any]:

        with self._timed("TroubleshootingOrchestrator.create_problem_and_attach"):

            # WRITE WORKFLOW — single transaction boundary
            with self.transaction() as session:

                self._debug(f"Creating problem '{name}'")

                problem_id = self.services.troubleshooting.add_problem(
                    name=name,
                    description=description,
                    session=session
                )

                self._debug(f"Attaching problem {problem_id} to position {position_id}")

                self.services.troubleshooting.attach_problem_to_position(
                    problem_id=problem_id,
                    position_id=position_id,
                    session=session
                )

                # Commit happens automatically when context exits

                return {
                    "status": "created",
                    "problem_id": problem_id,
                    "position_id": position_id
                }
