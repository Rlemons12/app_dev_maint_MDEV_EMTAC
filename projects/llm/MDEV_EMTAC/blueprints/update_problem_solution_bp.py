import logging
from flask import Blueprint
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from blueprints import DATABASE_URL

# Create the blueprint
update_problem_solution_bp = Blueprint('update_problem_solution_bp', __name__)

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#@update_problem_solution_bp.route('/update_problem_solution', methods=['POST'])
'''def update_problem_solution():
    try:
        logger.info("Entered update_problem_solution route")
        print("Entered update_problem_solution route")  # Debug print statement

        # Log request headers and form data
        logger.info(f"Request headers: {request.headers}")
        logger.info(f"Request form data: {request.form}")

        # Get form data
        problem_name = request.form.get('problem_name')
        tsg_area_id = request.form.get('tsg_area')
        tsg_equipment_group_id = request.form.get('tsg_equipment_group')
        tsg_model_id = request.form.get('tsg_model')
        tsg_asset_number_id = request.form.get('tsg_asset_number')
        tsg_location_id = request.form.get('tsg_location')
        problem_description = request.form.get('problem_description')
        tsg_problem_image_search = request.form.getlist('tsg_problem_image_search')
        solution_description = request.form.get('solution_description')
        tsg_solution_image_search = request.form.getlist('tsg_solution_image_search')
        tsg_document_search = request.form.get('tsg_document_search')
        tsg_selected_part_search = request.form.get('tsg_selected_part_search')
        tsg_selected_drawing_search = request.form.get('tsg_selected_drawing_search')
        site_location_id = None  # Assuming you get site_location_id from somewhere or it’s not needed

        logger.info("Received form data")
        logger.info(f"problem_name: {problem_name}")
        logger.info(f"tsg_area_id: {tsg_area_id}")
        logger.info(f"tsg_equipment_group_id: {tsg_equipment_group_id}")
        logger.info(f"tsg_model_id: {tsg_model_id}")
        logger.info(f"tsg_asset_number_id: {tsg_asset_number_id}")
        logger.info(f"tsg_location_id: {tsg_location_id}")
        logger.info(f"problem_description: {problem_description}")
        logger.info(f"solution_description: {solution_description}")
        logger.info(f"tsg_problem_image_search: {tsg_problem_image_search}")
        logger.info(f"tsg_solution_image_search: {tsg_solution_image_search}")
        logger.info(f"tsg_document_search: {tsg_document_search}")
        logger.info(f"tsg_selected_part_search: {tsg_selected_part_search}")
        logger.info(f"tsg_selected_drawing_search: {tsg_selected_drawing_search}")
        logger.info(f"site_location_id: {site_location_id}")

        # Start a new session
        session = Session()

        try:
            # Create or get existing Position using the create_position function
            logger.info("Creating or retrieving position")
            position_id = create_position(
                session=session,  # Pass the session here
                area_id=tsg_area_id,
                equipment_group_id=tsg_equipment_group_id,
                model_id=tsg_model_id,
                asset_number_id=tsg_asset_number_id,
                location_id=tsg_location_id,
                site_location_id=site_location_id
            )

            if not position_id:
                logger.error("Failed to create or retrieve position")
                return "Failed to create or retrieve position", 500

            logger.info(f"Created or retrieved position with ID: {position_id}")

            # Add problem description and related images
            logger.info("Adding problem")
            logger.info(f"Instantiating Problem with name={problem_name}, description={problem_description}")
            problem = Problem(name=problem_name, description=problem_description)
            session.add(problem)
            session.commit()
            logger.info(f"Added problem with ID: {problem.id}")

            # Create the association between the problem and the position
            logger.info(f"Associating problem {problem.id} with position {position_id}")
            problem_position = ProblemPositionAssociation(problem_id=problem.id, position_id=position_id)
            session.add(problem_position)

            for image_id in tsg_problem_image_search:
                logger.info(f"Associating problem {problem.id} with image {image_id}")
                image = session.query(Image).get(image_id)
                image_problem = ImageProblemAssociation(image=image, problem=problem)
                session.add(image_problem)

            # Add solution description and related images
            logger.info(f"Adding solution for problem {problem.id}")
            logger.info(f"Instantiating Solution with description={solution_description}, problem_id={problem.id}")
            solution = Solution(description=solution_description, problem_id=problem.id)
            session.add(solution)
            session.commit()
            logger.info(f"Added solution with ID: {solution.id}")

            for image_id in tsg_solution_image_search:
                logger.info(f"Associating solution {solution.id} with image {image_id}")
                image = session.query(Image).get(image_id)
                image_solution = ImageSolutionAssociation(image=image, solution=solution)
                session.add(image_solution)

            # Add part and drawing associations
            if tsg_selected_part_search:
                logger.info(f"Searching for part number: {tsg_selected_part_search}")
                part = session.query(Part).filter_by(part_number=tsg_selected_part_search).first()
                if part:
                    logger.info(f"Associating part {part.id} with problem {problem.id} and solution {solution.id}")
                    part_problem = PartProblemAssociation(part=part, problem=problem)
                    part_solution = PartSolutionAssociation(part=part, solution=solution)
                    session.add(part_problem)
                    session.add(part_solution)

            if tsg_selected_drawing_search:
                logger.info(f"Searching for drawing number: {tsg_selected_drawing_search}")
                drawing = session.query(Drawing).filter_by(drw_number=tsg_selected_drawing_search).first()
                if drawing:
                    logger.info(
                        f"Associating drawing {drawing.id} with problem {problem.id} and solution {solution.id}")
                    drawing_problem = DrawingProblemAssociation(drawing=drawing, problem=problem)
                    drawing_solution = DrawingSolutionAssociation(drawing=drawing, solution=solution)
                    session.add(drawing_problem)
                    session.add(drawing_solution)

            # Commit all changes
            logger.info("Committing all changes")
            session.commit()
            logger.info("All changes committed successfully")
        except Exception as e:
            logger.error(f"An error occurred while committing the session: {e}")
            session.rollback()
            return f"An error occurred: {e}", 500
        finally:
            session.close()
    except Exception as e:
        logger.error(f"An error occurred in the route: {e}")
        return f"An error occurred: {e}", 500

    return render_template('troubleshooting_guide.html')
'''