# Updated DataService using sys.modules to dynamically import DatabaseConfig
import os
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, select  # <- make sure select is here
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from sys import modules
from modules.configuration.log_config import logger
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from modules.emtacdb.emtacdb_fts import (
    Area, EquipmentGroup, Model, AssetNumber, Location,
    Problem, Solution, Task, TaskSolutionAssociation,
    Part, Image, Drawing, CompleteDocument,
    Position, DrawingPartAssociation,
    PartsPositionImageAssociation, ImagePositionAssociation,
    ProblemPositionAssociation, DrawingPositionAssociation,
    CompletedDocumentPositionAssociation, TaskPositionAssociation, CompleteDocumentTaskAssociation,
    ImageTaskAssociation, PartTaskAssociation, DrawingTaskAssociation, CompleteDocumentProblemAssociation
)
from modules.configuration import config
# Dynamically import DatabaseConfig from the configuration module
if "modules.configuration.config_env" not in modules:
    import modules.configuration.config_env
DatabaseConfig = modules["modules.configuration.config_env"].DatabaseConfig

# Create an instance of DatabaseConfig and get a session
db_config = DatabaseConfig()
session = db_config.get_main_session()

from sys import modules
from modules.configuration.log_config import logger

# Import your specific models
from modules.emtacdb.emtacdb_fts import (
    Area, EquipmentGroup, Model, AssetNumber, Location,
    Problem, Solution, Task, TaskSolutionAssociation,
    Part, Image, Drawing, CompleteDocument,
    Position, DrawingPartAssociation,
    PartsPositionImageAssociation, ImagePositionAssociation,
    ProblemPositionAssociation, DrawingPositionAssociation,
    CompletedDocumentPositionAssociation
)

if "modules.configuration.config_env" not in modules:
    import modules.configuration.config_env
DatabaseConfig = modules["modules.configuration.config_env"].DatabaseConfig

db_config = DatabaseConfig()
session = db_config.get_main_session()

class DataService:
    """Service layer for database operations (read-only)"""

    def __init__(self, session):
        """Initialize with a database session"""
        self.session = session
        logger.info("DataService initialized with session: %s", session)

    def get_all_areas(self):
        """Get all areas"""
        logger.debug("Entering get_all_areas")
        try:
            areas = self.session.query(Area).all()
            logger.debug("Fetched %d areas", len(areas))
            logger.debug("Exiting get_all_areas")
            return areas
        except Exception as e:
            logger.exception("Error fetching all areas: %s", e)
            raise

    def get_equipment_groups_by_area(self, area_id):
        """Get equipment groups for a specific area"""
        logger.debug("Entering get_equipment_groups_by_area with area_id: %s", area_id)
        try:
            groups = self.session.query(EquipmentGroup).filter(EquipmentGroup.area_id == area_id).all()
            logger.debug("Fetched %d equipment groups for area_id %s", len(groups), area_id)
            logger.debug("Exiting get_equipment_groups_by_area")
            return groups
        except Exception as e:
            logger.exception("Error fetching equipment groups for area %s: %s", area_id, e)
            raise

    def get_models_by_equipment_group(self, group_id):
        """Get models for a specific equipment group"""
        logger.debug("Entering get_models_by_equipment_group with group_id: %s", group_id)
        try:
            models = self.session.query(Model).filter(Model.equipment_group_id == group_id).all()
            logger.debug("Fetched %d models for group_id %s", len(models), group_id)
            logger.debug("Exiting get_models_by_equipment_group")
            return models
        except Exception as e:
            logger.exception("Error fetching models for equipment group %s: %s", group_id, e)
            raise

    def get_locations_by_model(self, model_id):
        """Get locations for a specific model"""
        logger.debug("Entering get_locations_by_model with model_id: %s", model_id)
        try:
            locations = self.session.query(Location).filter(Location.model_id == model_id).all()
            logger.debug("Fetched %d locations for model_id %s", len(locations), model_id)
            logger.debug("Exiting get_locations_by_model")
            return locations
        except Exception as e:
            logger.exception("Error fetching locations for model %s: %s", model_id, e)
            raise

    def get_asset_numbers_by_model(self, model_id):
        """Get asset numbers for a specific model"""
        logger.debug("Entering get_asset_numbers_by_model with model_id: %s", model_id)
        try:
            asset_numbers = self.session.query(AssetNumber).filter(AssetNumber.model_id == model_id).all()
            logger.debug("Fetched %d asset numbers for model_id %s", len(asset_numbers), model_id)
            logger.debug("Exiting get_asset_numbers_by_model")
            return asset_numbers
        except Exception as e:
            logger.exception("Error fetching asset numbers for model %s: %s", model_id, e)
            raise

    def get_problems_by_filters(self, area_id=None, equipment_group_id=None,
                                model_id=None, asset_number_id=None, location_id=None,
                                subassembly_id=None, component_assembly_id=None, assembly_view_id=None):
        """
        Get problems for all positions matching the given filters.
        """
        logger.debug(
            "Entering get_problems_by_filters with filters: area_id=%s, equipment_group_id=%s, model_id=%s, location_id=%s, asset_number_id=%s, subassembly_id=%s, component_assembly_id=%s, assembly_view_id=%s",
            area_id, equipment_group_id, model_id, location_id, asset_number_id, subassembly_id, component_assembly_id, assembly_view_id
        )
        # For debugging purposes, check a test position
        test_position = self.session.query(Position).filter(Position.id == 37).first()
        if test_position:
            logger.debug("Position 37 exists with area_id=%s", test_position.area_id)
        else:
            logger.debug("Position 37 does not exist in the database")
        try:
            position_query = self.session.query(Position.id)
            if area_id:
                position_query = position_query.filter(Position.area_id == area_id)
            if equipment_group_id:
                position_query = position_query.filter(Position.equipment_group_id == equipment_group_id)
            if model_id:
                position_query = position_query.filter(Position.model_id == model_id)
            if location_id:
                position_query = position_query.filter(Position.location_id == location_id)
            if asset_number_id:
                position_query = position_query.filter(Position.asset_number_id == asset_number_id)
            if subassembly_id:
                position_query = position_query.filter(Position.subassembly_id == subassembly_id)
            if component_assembly_id:
                position_query = position_query.filter(Position.component_assembly_id == component_assembly_id)
            if assembly_view_id:
                position_query = position_query.filter(Position.assembly_view_id == assembly_view_id)

            position_ids = [pos_id for pos_id, in position_query.all()]
            logger.debug("Found positions with IDs: %s", position_ids)

            if not position_ids:
                logger.debug("No positions found for filters, returning empty problem list")
                logger.debug("Exiting get_problems_by_filters")
                return []

            problems_query = self.session.query(Problem).distinct().join(
                ProblemPositionAssociation,
                ProblemPositionAssociation.problem_id == Problem.id
            ).filter(
                ProblemPositionAssociation.position_id.in_(position_ids)
            )
            logger.debug("Problems query: %s", str(problems_query))
            problems = problems_query.all()

            if problems:
                for prob in problems:
                    logger.debug("Found problem: ID=%s, Name=%s", prob.id, prob.name)

            logger.debug("Fetched %d distinct problems for positions", len(problems))
            logger.debug("Exiting get_problems_by_filters")
            return problems
        except Exception as e:
            logger.error("Error in get_problems_by_filters: %s", e)
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_problems_by_position(self, position_id):
        """
        Retrieve all problems associated with a specific position.
        If no problems are found, fall back to finding problems associated with the same model.
        """
        logger.info(f"========== GETTING PROBLEMS FOR POSITION {position_id} ==========")
        try:
            # First get the position details for better logging
            position = self.session.query(Position).filter(Position.id == position_id).first()
            if position:
                logger.info(f"Position details: ID={position.id}, Area={position.area_id}, " +
                            f"Group={position.equipment_group_id}, Model={position.model_id}, " +
                            f"Location={position.location_id}")
            else:
                logger.warning(f"Position ID {position_id} not found in database")
                return []

            # Query problems associated with this position
            problems = (
                self.session.query(Problem)
                .join(ProblemPositionAssociation, Problem.id == ProblemPositionAssociation.problem_id)
                .filter(ProblemPositionAssociation.position_id == position_id)
                .all()
            )

            logger.info(f"Found {len(problems)} problems for position ID {position_id}")

            # If problems were found, log details and return them
            if problems:
                # Log details of each problem found
                for i, problem in enumerate(problems):
                    logger.debug(f"Problem #{i + 1}: ID={problem.id}, Name='{problem.name}'")

                    # Get and log solutions for this problem
                    try:
                        solutions = self.session.query(Solution).filter(Solution.problem_id == problem.id).all()
                        logger.debug(f"  -> Found {len(solutions)} solutions for problem ID {problem.id}")
                    except Exception as e:
                        logger.warning(f"Error getting solutions for problem ID {problem.id}: {e}")

                return problems

            # If no position-specific problems found and position has a model_id,
            # look for problems linked to any position with the same model_id
            elif position.model_id:
                logger.info(
                    f"No position-specific problems found. Falling back to model-based problems for model_id {position.model_id}")

                # Find positions with the same model_id (excluding the current position)
                model_positions = (
                    self.session.query(Position.id)
                    .filter(Position.model_id == position.model_id)
                    .filter(Position.id != position_id)
                    .all()
                )

                if not model_positions:
                    logger.info(f"No other positions found with model_id {position.model_id}")
                    return []

                # Extract position IDs
                model_position_ids = [pos_id for (pos_id,) in model_positions]
                logger.debug(
                    f"Found {len(model_position_ids)} other positions with the same model_id: {model_position_ids}")

                # Query problems associated with any of these positions
                model_problems = (
                    self.session.query(Problem)
                    .join(ProblemPositionAssociation, Problem.id == ProblemPositionAssociation.problem_id)
                    .filter(ProblemPositionAssociation.position_id.in_(model_position_ids))
                    .distinct()  # Avoid duplicates
                    .all()
                )

                logger.info(
                    f"Found {len(model_problems)} problems from other positions with the same model_id {position.model_id}")

                # Log details of each problem found
                for i, problem in enumerate(model_problems):
                    logger.debug(f"Model-based Problem #{i + 1}: ID={problem.id}, Name='{problem.name}'")

                    # Get and log solutions for this problem
                    try:
                        solutions = self.session.query(Solution).filter(Solution.problem_id == problem.id).all()
                        logger.debug(f"  -> Found {len(solutions)} solutions for problem ID {problem.id}")
                    except Exception as e:
                        logger.warning(f"Error getting solutions for problem ID {problem.id}: {e}")

                return model_problems

            # No problems found at all
            return []
        except Exception as e:
            logger.error(f"Error getting problems for position ID {position_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            logger.info(f"========== PROBLEM RETRIEVAL COMPLETE ==========")

    def get_problems_by_filters(self, area_id=None, equipment_group_id=None, model_id=None, location_id=None):
        """Retrieve problems based on hierarchy filters."""
        logger.info(f"========== GETTING PROBLEMS BY FILTERS ==========")
        logger.info(f"Filters: Area={area_id}, Group={equipment_group_id}, Model={model_id}, Location={location_id}")

        try:
            # Start with a query for positions that match the filters
            positions_query = self.session.query(Position.id).distinct()

            filter_count = 0
            if area_id:
                logger.debug(f"Applying area filter: {area_id}")
                positions_query = positions_query.filter(Position.area_id == area_id)
                filter_count += 1

            if equipment_group_id:
                logger.debug(f"Applying equipment group filter: {equipment_group_id}")
                positions_query = positions_query.filter(Position.equipment_group_id == equipment_group_id)
                filter_count += 1

            if model_id:
                logger.debug(f"Applying model filter: {model_id}")
                positions_query = positions_query.filter(Position.model_id == model_id)
                filter_count += 1

            if location_id:
                logger.debug(f"Applying location filter: {location_id}")
                positions_query = positions_query.filter(Position.location_id == location_id)
                filter_count += 1

            logger.debug(f"Applied {filter_count} filters to position query")

            # Execute the query to get matching positions
            position_ids = [pos_id for pos_id, in positions_query.all()]
            logger.info(f"Found {len(position_ids)} positions matching the filters: {position_ids}")

            if not position_ids:
                logger.warning("No positions match the filters, returning empty problem list")
                return []

            # Now get all problems associated with these positions
            problems = (
                self.session.query(Problem)
                .join(ProblemPositionAssociation, Problem.id == ProblemPositionAssociation.problem_id)
                .filter(ProblemPositionAssociation.position_id.in_(position_ids))
                .distinct()
                .all()
            )

            logger.info(f"Found {len(problems)} distinct problems for the {len(position_ids)} positions")

            # Log details of each problem found
            for i, problem in enumerate(problems):
                logger.debug(f"Problem #{i + 1}: ID={problem.id}, Name='{problem.name}'")

            return problems
        except Exception as e:
            logger.error(f"Error getting problems by filters: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            logger.info(f"========== PROBLEM FILTER RETRIEVAL COMPLETE ==========")

    def get_solutions_by_problem(self, problem_id):
        """Get solutions for a specific problem"""
        logger.debug("Entering get_solutions_by_problem with problem_id: %s", problem_id)
        try:
            solutions = self.session.query(Solution).filter(Solution.problem_id == problem_id).all()
            logger.debug("Fetched %d solutions for problem_id %s", len(solutions), problem_id)
            if not solutions:
                logger.warning("No solutions found for problem_id %s", problem_id)
            logger.debug("Exiting get_solutions_by_problem")
            return solutions
        except Exception as e:
            logger.exception("Error fetching solutions for problem %s: %s", problem_id, e)
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_tasks_by_solution(self, solution_id):
        logger.debug("Entering get_tasks_by_solution with solution_id: %s", solution_id)
        try:
            from sqlalchemy import text
            result = self.session.execute(
                text(f"SELECT * FROM task_solution_association WHERE solution_id = {solution_id}")
            )
            assocs = result.fetchall()
            logger.debug("Direct SQL found %d task associations", len(assocs))
            tasks = self.session.query(Task).join(
                TaskSolutionAssociation,
                TaskSolutionAssociation.task_id == Task.id
            ).filter(
                TaskSolutionAssociation.solution_id == solution_id
            ).all()
            logger.debug("ORM query found %d tasks", len(tasks))
            for task in tasks:
                logger.debug("Task: ID=%s, Name=%s", task.id, task.name)
            logger.debug("Exiting get_tasks_by_solution")
            return tasks
        except Exception as e:
            logger.exception("Error in get_tasks_by_solution: %s", e)
            return []

    def get_parts_by_position_and_task(self, position_id):
        """Get parts associated with a position (directly and via tasks), including image paths."""
        logger.debug("===> Entering get_parts_by_position_and_task(position_id=%s)", position_id)
        try:
            # STEP 1: Parts directly associated with the position
            direct_assocs = self.session.query(PartsPositionImageAssociation).filter(
                PartsPositionImageAssociation.position_id == position_id
            ).all()
            logger.debug("Retrieved %d direct PartsPositionImageAssociation records", len(direct_assocs))

            direct_parts = []
            for assoc in direct_assocs:
                part = assoc.part
                image = assoc.image
                if image and image.file_path:
                    filename = os.path.basename(image.file_path)
                    image_path = os.path.join(config.DATABASE_PATH_IMAGES_FOLDER, filename)
                    if os.path.exists(image_path):
                        part.image_path = image_path
                        logger.debug("Image resolved for part '%s' (ID: %d): %s",
                                     part.part_number, part.id, image_path)
                    else:
                        part.image_path = None
                        logger.warning("Image file missing for part '%s' (ID: %d) at path: %s",
                                       part.part_number, part.id, image_path)
                else:
                    part.image_path = None
                    logger.debug("No image associated with part '%s' (ID: %d)", part.part_number, part.id)
                direct_parts.append(part)

            logger.debug("Collected %d directly linked parts", len(direct_parts))

            # STEP 2: Get problems linked to this position
            problems = self.session.query(ProblemPositionAssociation.problem_id).filter(
                ProblemPositionAssociation.position_id == position_id
            ).all()
            problem_ids = [pid for pid, in problems]
            logger.debug("Retrieved %d problems linked to position: %s", len(problem_ids), problem_ids)

            # STEP 3: Get solutions for each problem
            solution_ids = []
            for pid in problem_ids:
                solutions = self.session.query(Solution.id).filter(Solution.problem_id == pid).all()
                new_ids = [sid for sid, in solutions]
                solution_ids.extend(new_ids)
                logger.debug("Problem ID %d -> %d solution(s): %s", pid, len(new_ids), new_ids)

            logger.debug("Aggregated total of %d solution IDs", len(solution_ids))

            # STEP 4: Get task IDs for those solutions
            task_ids = []
            if solution_ids:
                task_id_rows = self.session.query(TaskSolutionAssociation.task_id).filter(
                    TaskSolutionAssociation.solution_id.in_(solution_ids)
                ).distinct().all()
                task_ids = [tid for tid, in task_id_rows]
                logger.debug("Collected %d task IDs via TaskSolutionAssociation: %s", len(task_ids), task_ids)
            else:
                logger.debug("No solutions found. Skipping task retrieval.")

            # STEP 5: Get parts associated with those tasks
            task_parts = []
            if task_ids:
                task_assocs = self.session.query(PartTaskAssociation).filter(
                    PartTaskAssociation.task_id.in_(task_ids)
                ).all()
                logger.debug("Found %d PartTaskAssociation records for task IDs", len(task_assocs))
                for assoc in task_assocs:
                    part = assoc.part
                    part.image_path = None  # Optional: enhance this to resolve images too
                    task_parts.append(part)
                    logger.debug("Task-linked part: %s (ID: %d)", part.part_number, part.id)
            else:
                logger.debug("No task IDs found. No task-linked parts.")

            # STEP 6: Deduplicate and return combined parts
            combined_parts = {part.id: part for part in direct_parts + task_parts}
            all_parts = list(combined_parts.values())
            logger.debug("Final deduplicated part count: %d", len(all_parts))
            logger.debug("<=== Exiting get_parts_by_position_and_task")
            return all_parts

        except Exception as e:
            logger.exception(" Error in get_parts_by_position_and_task(position_id=%s): %s", position_id, e)
            raise

    def get_parts_by_position(self, position_id):
        """Get parts associated with a position directly via PartsPositionImageAssociation."""
        logger.debug("===> Entering get_parts_by_position(position_id=%s)", position_id)
        try:
            direct_assocs = self.session.query(PartsPositionImageAssociation).filter(
                PartsPositionImageAssociation.position_id == position_id
            ).all()
            logger.debug("Retrieved %d PartsPositionImageAssociation records for position_id=%s",
                         len(direct_assocs), position_id)

            parts = []
            for assoc in direct_assocs:
                part = assoc.part
                image = assoc.image
                if image and image.file_path:
                    filename = os.path.basename(image.file_path)
                    image_path = os.path.join(config.DATABASE_PATH_IMAGES_FOLDER, filename)
                    if os.path.exists(image_path):
                        part.image_path = image_path
                        logger.debug("Image found for part '%s' (ID: %d): %s",
                                     part.part_number, part.id, image_path)
                    else:
                        part.image_path = None
                        logger.warning(" Image file missing for part '%s' (ID: %d) at path: %s",
                                       part.part_number, part.id, image_path)
                else:
                    part.image_path = None
                    logger.debug("No image associated with part '%s' (ID: %d)", part.part_number, part.id)

                parts.append(part)
                logger.debug(" Added part '%s' (ID: %d) to result list", part.part_number, part.id)

            logger.debug("<=== Exiting get_parts_by_position with %d parts", len(parts))
            return parts

        except Exception as e:
            logger.exception(" Error in get_parts_by_position(position_id=%s): %s", position_id, e)
            return []

    def get_parts_by_task(self, task_id):
        """Get parts associated with a task via PartTaskAssociation."""
        logger.debug("===> Entering get_parts_by_task(task_id=%s)", task_id)
        try:
            task_assocs = self.session.query(PartTaskAssociation).filter(
                PartTaskAssociation.task_id == task_id
            ).all()
            logger.debug("Retrieved %d PartTaskAssociation records for task_id=%s",
                         len(task_assocs), task_id)

            parts = []
            for assoc in task_assocs:
                part = assoc.part
                part.image_path = None  # No image resolution in this context
                parts.append(part)
                logger.debug("Task-linked part added: '%s' (ID: %d)", part.part_number, part.id)

            logger.debug("<=== Exiting get_parts_by_task with %d parts", len(parts))
            return parts

        except Exception as e:
            logger.exception("❌ Error in get_parts_by_task(task_id=%s): %s", task_id, e)
            return []

    def get_part_details(self, part_id):
        """Get part details"""
        logger.debug("Entering get_part_details with part_id: %s", part_id)
        try:
            part = self.session.query(Part).filter(Part.id == part_id).first()
            if part:
                logger.debug("Part details retrieved for part_id %s", part_id)
            else:
                logger.warning("No part found with part_id %s", part_id)
            logger.debug("Exiting get_part_details")
            return part
        except Exception as e:
            logger.exception("Error fetching part details for part %s: %s", part_id, e)
            raise

    def get_documents_by_task_id(self, task_id):
        logger.debug("===> Entering get_documents_by_task_id(task_id=%s)", task_id)
        try:
            task_assocs = self.session.query(CompleteDocumentTaskAssociation).filter(
                CompleteDocumentTaskAssociation.task_id == task_id
            ).all()
            logger.debug("Retrieved %d CompleteDocumentTaskAssociation records for task_id=%s",
                         len(task_assocs), task_id)
            documents = []
            for assoc in task_assocs:
                doc = assoc.complete_document
                documents.append(doc)
                logger.debug("Added document '%s' (ID: %d) to result list", doc.title, doc.id)
            logger.debug("<=== Exiting get_documents_by_task_id with %d documents", len(documents))
            return documents
        except Exception as e:
            logger.exception("Error in get_documents_by_task_id(task_id=%s): %s", task_id, e)
            return []

    def get_documents_by_position(self, position_id):
        logger.debug(f"Entering get_documents_by_position with position_id: {position_id}")
        try:
            documents = (
                self.session.query(CompleteDocument)
                .join(CompletedDocumentPositionAssociation,
                      CompletedDocumentPositionAssociation.complete_document_id == CompleteDocument.id)
                .filter(CompletedDocumentPositionAssociation.position_id == position_id)
                .options(joinedload(CompleteDocument.completed_document_position_association))
                .all()
            )
            logger.debug(f"Fetched {len(documents)} documents linked to position_id: {position_id}")
            logger.debug("Exiting get_documents_by_position")
            return documents
        except Exception as e:
            logger.exception("Error in get_documents_by_position(position_id=%s): %s", position_id, e)
            return []

    def get_document_details(self, document_id):
        """Get document details"""
        logger.debug("Entering get_document_details with document_id: %s", document_id)
        try:
            document = self.session.query(CompleteDocument).filter(CompleteDocument.id == document_id).first()
            if document:
                logger.debug("Document details retrieved for document_id %s", document_id)
            else:
                logger.warning("No document found with document_id %s", document_id)
            logger.debug("Exiting get_document_details")
            return document
        except Exception as e:
            logger.exception("Error fetching document details for document %s: %s", document_id, e)
            raise

    def get_images_by_task(self, task_id):
        """Retrieve images associated with a given task ID."""
        logger.debug("Fetching images for task_id=%s", task_id)
        try:
            associations = self.session.query(ImageTaskAssociation).filter(
                ImageTaskAssociation.task_id == task_id
            ).all()

            images_with_paths = []
            for assoc in associations:
                image = assoc.image
                image_path = os.path.join(config.DATABASE_PATH_IMAGES_FOLDER, os.path.basename(image.file_path))
                if os.path.exists(image_path):
                    images_with_paths.append({
                        'id': image.id,
                        'title': image.title,
                        'description': image.description,
                        'file_path': image_path
                    })
                else:
                    logger.warning("Image file not found: %s", image_path)

            logger.debug("Retrieved %d images for task_id=%s", len(images_with_paths), task_id)
            return images_with_paths
        except Exception as e:
            logger.exception("Error fetching images for task_id=%s: %s", task_id, e)
            return []

    def get_images_by_position(self, position_id):
        """Retrieve images associated with a given position ID."""
        logger.debug("Fetching images for position_id=%s", position_id)
        try:
            associations = self.session.query(ImagePositionAssociation).filter(
                ImagePositionAssociation.position_id == position_id
            ).all()

            images_with_paths = []
            for assoc in associations:
                image = assoc.image
                image_path = os.path.join(config.DATABASE_PATH_IMAGES_FOLDER, os.path.basename(image.file_path))
                if os.path.exists(image_path):
                    images_with_paths.append({
                        'id': image.id,
                        'title': image.title,
                        'description': image.description,
                        'file_path': image_path
                    })
                else:
                    logger.warning("Image file not found: %s", image_path)

            logger.debug("Retrieved %d images for position_id=%s", len(images_with_paths), position_id)
            return images_with_paths
        except Exception as e:
            logger.exception("Error fetching images for position_id=%s: %s", position_id, e)
            return []

    def get_image_details(self, image_id):
        """Get image details"""
        logger.debug("Entering get_image_details with image_id: %s", image_id)
        try:
            image = self.session.query(Image).filter(Image.id == image_id).first()
            if image:
                logger.debug("Image details retrieved for image_id %s", image_id)
            else:
                logger.warning("No image found with image_id %s", image_id)
            logger.debug("Exiting get_image_details")
            return image
        except Exception as e:
            logger.exception("Error fetching image details for image %s: %s", image_id, e)
            raise

    def get_drawings_by_position(self, position_id):
        """Get drawings associated with a position"""
        logger.debug("Entering get_drawings_by_position with position_id: %s", position_id)
        try:
            drawings = self.session.query(Drawing).join(
                DrawingPositionAssociation,
                DrawingPositionAssociation.drawing_id == Drawing.id
            ).filter(
                DrawingPositionAssociation.position_id == position_id
            ).all()
            logger.debug("Fetched %d drawings for position_id %s", len(drawings), position_id)
            logger.debug("Exiting get_drawings_by_position")
            return drawings
        except Exception as e:
            logger.exception("Error fetching drawings for position %s: %s", position_id, e)
            raise

    def get_drawings_by_task(self, task_id):
        try:
            return (
                self.session.query(Drawing)
                .join(DrawingTaskAssociation)
                .filter(DrawingTaskAssociation.task_id == task_id)
                .all()
            )
        except Exception as e:
            logger.error(f"Error fetching drawings for task {task_id}: {e}")
            return []

    def get_drawing_details(self, drawing_id):
        """Get drawing details"""
        logger.debug("Entering get_drawing_details with drawing_id: %s", drawing_id)
        try:
            drawing = self.session.query(Drawing).filter(Drawing.id == drawing_id).first()
            if drawing:
                logger.debug("Drawing details retrieved for drawing_id %s", drawing_id)
            else:
                logger.warning("No drawing found with drawing_id %s", drawing_id)
            logger.debug("Exiting get_drawing_details")
            return drawing
        except Exception as e:
            logger.exception("Error fetching drawing details for drawing %s: %s", drawing_id, e)
            raise

    def get_drawings_by_part(self, part_id):
        """Get drawings associated with a part"""
        logger.debug("Entering get_drawings_by_part with part_id: %s", part_id)
        try:
            drawings = self.session.query(Drawing).join(
                DrawingPartAssociation,
                DrawingPartAssociation.drawing_id == Drawing.id
            ).filter(
                DrawingPartAssociation.part_id == part_id
            ).all()
            logger.debug("Fetched %d drawings for part_id %s", len(drawings), part_id)
            logger.debug("Exiting get_drawings_by_part")
            return drawings
        except Exception as e:
            logger.exception("Error fetching drawings for part %s: %s", part_id, e)
            raise

    def diagnose_problem_solution_task(self, problem_id):
        """Debug method to trace a problem's solutions and tasks"""
        logger.debug("Diagnosing Problem ID: %s", problem_id)
        try:
            solutions = self.session.query(Solution).filter(Solution.problem_id == problem_id).all()
            logger.debug("Found %d solutions for problem ID %s", len(solutions), problem_id)
            for solution in solutions:
                logger.debug("Solution ID: %s, Name: %s", solution.id, solution.name)
                task_associations = self.session.query(TaskSolutionAssociation).filter(
                    TaskSolutionAssociation.solution_id == solution.id
                ).all()
                logger.debug("Found %d task associations for Solution ID %s", len(task_associations), solution.id)
                for assoc in task_associations:
                    task = self.session.query(Task).filter(Task.id == assoc.task_id).first()
                    if task:
                        logger.debug("Task ID: %s, Name: %s", task.id, task.name)
                    else:
                        logger.warning("Task ID %s not found in database", assoc.task_id)
            logger.debug("Finished diagnosing problem_solution_task for Problem ID: %s", problem_id)
        except Exception as e:
            logger.exception("Error during diagnosis for problem %s: %s", problem_id, e)

    def diagnose_solutions_for_problem(self, problem_id):
        """Directly check solutions for a specific problem"""
        logger.debug("Diagnosing solutions for Problem ID: %s", problem_id)
        try:
            problem = self.session.query(Problem).filter(Problem.id == problem_id).first()
            if problem:
                logger.debug("Problem found: ID %s, Name: %s", problem_id, problem.name)
                logger.debug("Problem description: %s", problem.description)
                solutions = self.session.query(Solution).filter(Solution.problem_id == problem_id).all()
                logger.debug("Found %d solutions for problem ID %s", len(solutions), problem_id)
                for solution in solutions:
                    logger.debug("Solution ID: %s, Name: %s", solution.id, solution.name)
                    task_associations = self.session.query(TaskSolutionAssociation).filter(
                        TaskSolutionAssociation.solution_id == solution.id
                    ).all()
                    logger.debug("Found %d task associations for solution ID %s", len(task_associations), solution.id)
                    for ta in task_associations:
                        task = self.session.query(Task).filter(Task.id == ta.task_id).first()
                        if task:
                            logger.debug("Task ID: %s, Name: %s", task.id, task.name)
                logger.debug("Exiting diagnose_solutions_for_problem")
                return len(solutions)
            else:
                logger.warning("Problem ID %s not found during diagnosis", problem_id)
                return 0
        except Exception as e:
            logger.exception("Error in diagnose_solutions_for_problem: %s", e)
            return -1

    def get_tools_by_task(self, task_id):
        try:
            from modules.emtacdb.emtacdb_fts import TaskToolAssociation, Tool
            # Query the TaskToolAssociation table for the given task_id
            associations = self.session.query(TaskToolAssociation).filter(
                TaskToolAssociation.task_id == task_id
            ).all()
            # Extract the Tool objects from the associations
            tools = [assoc.tool for assoc in associations]
            logger.debug("Found %d tools for task ID %s", len(tools), task_id)
            return tools
        except Exception as e:
            logger.exception("Error retrieving tools for task %s: %s", task_id, e)
            return []

    def get_position_by_task_id(self, task_id):
        from modules.emtacdb.emtacdb_fts import TaskPositionAssociation
        association = self.session.query(TaskPositionAssociation).filter(
            TaskPositionAssociation.task_id == task_id
        ).first()
        if association:
            return association.position
        return None


