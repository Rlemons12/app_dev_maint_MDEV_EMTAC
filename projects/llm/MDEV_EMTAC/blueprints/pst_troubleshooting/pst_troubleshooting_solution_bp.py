# pst_troubleshooting_solution.py

from flask import Blueprint, jsonify, request
from sqlalchemy.exc import SQLAlchemyError
from modules.configuration.config_env import DatabaseConfig  # Adjust import based on your project structure
from modules.emtacdb.emtacdb_fts import (Problem, Solution, Task, TaskSolutionAssociation, ProblemPositionAssociation)
from modules.configuration.log_config import with_request_id, logger


# Initialize Database Config
db_config = DatabaseConfig()


# Define a new blueprint for solution-related routes
pst_troubleshooting_solution_bp = Blueprint('pst_troubleshooting_solution_bp', __name__)


@pst_troubleshooting_solution_bp.route('/get_solutions/<int:problem_id>', methods=['GET'])
def get_solutions(problem_id):
    """
    Retrieve solutions related to the specified problem.
    """
    print(f"=== DEBUG: get_solutions called with problem_id: {problem_id} ===")
    session = None
    try:
        print("DEBUG: Getting database session...")
        session = db_config.get_main_session()
        print(f"DEBUG: Session obtained: {session}")

        print(f"DEBUG: Querying Solution table for problem_id: {problem_id}")
        # Query the Solution table for solutions with the specified problem_id
        solutions = session.query(Solution).filter_by(problem_id=problem_id).all()
        print(f"DEBUG: Found {len(solutions)} solutions")

        if not solutions:
            print("DEBUG: No solutions found, returning message")
            return jsonify({'message': 'No solutions added yet for this problem.'}), 200

        print("DEBUG: Processing solutions data...")
        # Format the solutions data including id, name, and description
        solutions_data = []
        for i, solution in enumerate(solutions):
            print(f"DEBUG: Processing solution {i + 1}: ID={solution.id}, Name={solution.name}")
            solution_dict = {
                'id': solution.id,
                'name': solution.name,
                'description': solution.description
            }
            solutions_data.append(solution_dict)

        print(f"DEBUG: Successfully processed {len(solutions_data)} solutions")
        return jsonify(solutions_data), 200

    except SQLAlchemyError as e:
        print(f"DEBUG: SQLAlchemy error: {e}")
        logger.error(f"Database error fetching solutions: {e}")
        return jsonify({'error': 'An error occurred while fetching solutions.'}), 500
    except AttributeError as e:
        print(f"DEBUG: AttributeError (likely model issue): {e}")
        logger.error(f"Model attribute error: {e}")
        return jsonify({'error': f'Model error: {str(e)}'}), 500
    except ImportError as e:
        print(f"DEBUG: ImportError (likely import issue): {e}")
        logger.error(f"Import error: {e}")
        return jsonify({'error': f'Import error: {str(e)}'}), 500
    except Exception as e:
        print(f"DEBUG: Unexpected error: {type(e).__name__}: {e}")
        logger.error(f"Unexpected error fetching solutions: {e}")
        import traceback
        print("DEBUG: Full traceback:")
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    finally:
        if session:
            try:
                session.close()
                print("DEBUG: Session closed successfully")
            except Exception as e:
                print(f"DEBUG: Error closing session: {e}")

@pst_troubleshooting_solution_bp.route('/remove_solutions/', methods=['POST'])
def remove_solutions():
    """
    Delete solutions and their task associations for a given problem.
    """
    session = db_config.get_main_session()
    try:
        data = request.get_json()
        problem_id = data.get('problem_id')
        solution_ids = data.get('solution_ids')

        if not problem_id or not solution_ids:
            logger.error("Problem ID or Solution IDs missing in request.")
            return jsonify({'error': 'Problem ID and Solution IDs are required.'}), 400

        # Check if the problem exists
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if not problem:
            logger.error(f"Problem with ID {problem_id} not found.")
            return jsonify({'error': 'Problem not found.'}), 404

        # Fetch solutions linked to the problem that need to be deleted
        solutions_to_delete = session.query(Solution).filter(
            Solution.id.in_(solution_ids), Solution.problem_id == problem_id).all()
        if not solutions_to_delete:
            logger.error("No matching solutions found to delete.")
            return jsonify({'error': 'No matching solutions found to delete.'}), 404

        # Delete task associations in TaskSolutionAssociation for each solution
        for solution in solutions_to_delete:
            # This cascades deletion of TaskSolutionAssociation entries
            session.delete(solution)

        # Commit the transaction
        session.commit()

        logger.info(f"Deleted solutions with IDs {solution_ids} and their task associations for problem ID {problem_id}.")
        return jsonify({'message': 'Solutions and their task associations deleted successfully.'}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error while deleting solutions: {e}")
        return jsonify({'error': 'An error occurred while deleting solutions.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error while deleting solutions: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshooting_solution_bp.route('/add_solution/', methods=['POST'])
def add_solution():
    """
    Add a new solution to a problem.
    """
    session = db_config.get_main_session()
    try:
        data = request.get_json()
        problem_id = data.get('problem_id')
        solution_name = data.get('name')
        solution_description = data.get('description')  # Assuming description might be part of the payload

        # Validate required fields
        if not problem_id or not solution_name:
            logger.error("Problem ID or Solution Name missing in request.")
            return jsonify({'error': 'Problem ID and Solution Name are required.'}), 400

        # Verify problem exists
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if not problem:
            logger.error(f"Problem with ID {problem_id} not found.")
            return jsonify({'error': 'Problem not found.'}), 404

        # Create new solution instance
        new_solution = Solution(name=solution_name, description=solution_description, problem_id=problem_id)
        session.add(new_solution)

        # Commit the new solution to the database
        session.commit()

        logger.info(f"Added new solution '{solution_name}' to problem ID {problem_id}.")
        return jsonify({
            'message': 'Solution added successfully.',
            'solution': {
                'id': new_solution.id,
                'name': new_solution.name,
                'description': new_solution.description
            }
        }), 201

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error while adding solution: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred while adding the solution.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error while adding solution: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        session.close()

@pst_troubleshooting_solution_bp.route('/get_tasks/<int:solution_id>', methods=['GET'])
def get_tasks(solution_id):
    """
    Retrieve tasks associated with a specific solution.
    """
    session = db_config.get_main_session()
    try:
        # Query the TaskSolutionAssociation table to find tasks linked to the solution
        task_associations = session.query(TaskSolutionAssociation).filter_by(solution_id=solution_id).all()
        task_ids = [assoc.task_id for assoc in task_associations]

        # Retrieve tasks with the collected task IDs
        tasks = session.query(Task).filter(Task.id.in_(task_ids)).all()

        # Prepare task data for the response
        tasks_data = [{'id': task.id, 'name': task.name, 'description': task.description} for task in tasks]

        return jsonify({'tasks': tasks_data}), 200
    except SQLAlchemyError as e:
        logger.error(f"Database error fetching tasks: {e}")
        return jsonify({'error': 'An error occurred while fetching tasks.'}), 500
    finally:
        session.close()

@pst_troubleshooting_solution_bp.route('/add_task/', methods=['POST'])
def add_task():
    """
    Endpoint to add a new task to a solution and create the association.
    """
    session = db_config.get_main_session()
    try:
        # Extract data from the request
        data = request.get_json()
        solution_id = data.get('solution_id')
        task_name = data.get('name')
        task_description = data.get('description')

        # Check if required fields are provided
        if not solution_id or not task_name:
            return jsonify({"error": "Solution ID and task name are required."}), 400

        # Verify the solution exists
        solution = session.query(Solution).filter_by(id=solution_id).first()
        if not solution:
            return jsonify({"error": "Solution not found."}), 404

        # Create the new task
        new_task = Task(
            name=task_name,
            description=task_description
        )
        session.add(new_task)
        session.commit()  # Commit to generate the task ID

        # Create an association entry in TaskSolutionAssociation
        new_task_solution_association = TaskSolutionAssociation(
            task_id=new_task.id,
            solution_id=solution_id
        )
        session.add(new_task_solution_association)
        session.commit()  # Commit to save the association

        # Return a success message with new task details
        return jsonify({
            "status": "success",
            "message": "Task added successfully",
            "task": {
                "id": new_task.id,
                "name": new_task.name,
                "description": new_task.description
            }
        }), 201

    except Exception as e:
        # Handle any exceptions that may occur
        session.rollback()  # Rollback in case of error
        logger.error(f"Error adding task: {e}", exc_info=True)
        return jsonify({"error": "An error occurred while adding the task."}), 500
    finally:
        session.close()

@pst_troubleshooting_solution_bp.route('/remove_task/', methods=['POST'])
def remove_task_from_solution():

    session = db_config.get_main_session()

    try:
        data = request.get_json()
        task_id = data.get('task_id')
        solution_id = data.get('solution_id')

        if not task_id or not solution_id:
            return jsonify({'error': 'Missing task_id or solution_id.'}), 400

        # Fetch the association
        task_solution_association = session.query(TaskSolutionAssociation).filter_by(
            task_id=task_id,
            solution_id=solution_id
        ).first()

        if task_solution_association:
            session.delete(task_solution_association)
            session.commit()
            return jsonify({'status': 'success', 'message': 'Task removed from solution successfully.'}), 200
        else:
            return jsonify({'error': 'Association does not exist.'}), 404

    except Exception as e:
        session.rollback()
        return jsonify({'error': 'An unexpected error occurred.', 'details': str(e)}), 500


@pst_troubleshooting_solution_bp.route('/delete_problem', methods=['POST'])
def delete_problem():
    """
    Handle the deletion of a problem via AJAX, keeping the position entry intact,
    and deleting all associated solutions and their task associations.
    """
    # Assuming db_config.get_main_session() returns a SQLAlchemy session
    session = db_config.get_main_session()
    try:
        # Parse JSON payload
        data = request.get_json()
        logging.info(f"Received data for deletion: {data}")

        if not data or 'problem_id' not in data:
            logging.warning('Delete Problem: Missing problem_id in request.')
            return jsonify({'success': False, 'message': 'Problem ID is required.'}), 400

        problem_id = data['problem_id']
        logging.info(f"Attempting to delete Problem ID: {problem_id}")

        # Query for the Problem
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if not problem:
            logging.warning(f'Delete Problem: Problem ID {problem_id} not found.')
            return jsonify({'success': False, 'message': 'Problem not found.'}), 404

        # Fetch all solutions associated with the problem
        solutions = session.query(Solution).filter_by(problem_id=problem_id).all()

        # Delete task associations and solutions
        for solution in solutions:
            # Delete associated TaskSolutionAssociation entries
            task_solution_associations = session.query(TaskSolutionAssociation).filter_by(solution_id=solution.id).all()
            for association in task_solution_associations:
                session.delete(association)
                logging.info(f"Deleted TaskSolutionAssociation with ID {association.id} for Solution ID {solution.id}")

            # Delete the Solution
            session.delete(solution)
            logging.info(f"Deleted Solution with ID {solution.id}")

        # Delete associated entries in ProblemPositionAssociation
        associations = session.query(ProblemPositionAssociation).filter_by(problem_id=problem_id).all()
        for association in associations:
            session.delete(association)
            logging.info(f"Deleted ProblemPositionAssociation with ID {association.id}")

        # Delete the Problem
        session.delete(problem)
        logging.info(f"Deleted Problem with ID {problem_id}")

        # Commit the transaction
        session.commit()

        return jsonify({'success': True, 'message': 'Problem and associated solutions deleted successfully.'}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error during problem deletion: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while deleting the problem.'}), 500
    except Exception as e:
        session.rollback()
        logging.error(f"Unexpected error during problem deletion: {e}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred.'}), 500
    finally:
        session.close()
