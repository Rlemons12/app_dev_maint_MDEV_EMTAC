from flask import Blueprint, request, flash, jsonify, url_for
from modules.emtacdb.emtacdb_fts import (Problem, ImageTaskAssociation, CompleteDocument,
    drawing_part_image_model_location_association)
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from fuzzywuzzy import fuzz

# Create a SQLAlchemy engine using the DATABASE_URL from your config
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = sessionmaker(bind=engine)

search_problem_solution_bp = Blueprint('search_problem_solution_bp', __name__)

@search_problem_solution_bp.route('/search_problem_solution', methods=['GET'])
def search_problem_solution():
    try:
        session = Session()

        # Retrieve parameters from the request
        description = request.args.get('problem_description', '')
        location_id = request.args.get('problem_location', None)
        asset_number_id = request.args.get('problem_asset_number', None)
        model_id = request.args.get('problem_model', None)
        problem_title = request.args.get('problem_title', '')  # Retrieve problem title

        # Debug statements to print out the parameters
        print(f"Description: {description}")
        print(f"Location ID: {location_id}")
        print(f"Asset Number ID: {asset_number_id}")
        print(f"Model ID: {model_id}")
        print(f"Problem Title: {problem_title}")

        # Start the query with the Problem model
        query = session.query(Problem)

        # Apply filters based on other provided parameters
        if location_id:
            query = query.filter(Problem.location_id == int(location_id))
        if asset_number_id:
            query = query.filter(Problem.asset_number_id == int(asset_number_id))
        if model_id:
            query = query.filter(Problem.model_id == int(model_id))

        # Apply fuzzy matching filter on description if provided
        if description:
            filtered_problems = []
            for problem in query.all():
                # Calculate the similarity score between the problem description and the search query
                similarity_score = fuzz.token_set_ratio(problem.description.lower(), description.lower())
                # Adjust the threshold as per your requirement
                if similarity_score >= 80:  # Match only if the similarity score is exactly 100
                    filtered_problems.append(problem)

            # Use the filtered problems for further processing
            problems = filtered_problems
        else:
            # If description is not provided, use all filtered problems
            problems = query.all()

        if not problems:
            # Flash message indicating no problems found
            flash("No problems found", "error")
            return jsonify(problems=[])

        # Construct response containing the problems and associated solutions
        response = []
        for problem in problems:
            # Extract necessary attributes for each problem
            problem_info = {
                'id': problem.id,
                'name': problem.name,
                'description': problem.description,
                'location': problem.location.name if problem.location else None,
                'asset_number': problem.asset_number.number if problem.asset_number else None,
                'model': problem.model.name if problem.model else None
            }

            # Retrieve associated solutions
            solutions = []
            for solution in problem.solutions:
                # Extract necessary attributes for each solution
                solution_info = {
                    'id': solution.id,
                    'description': solution.description
                    # Add more attributes as needed
                }
                solutions.append(solution_info)
            
            # Add associated images to problem_info
            problem_info['images'] = []
            for solution in problem.solutions:
                # Query the ImageSolutionAssociation table to find associated image IDs
                image_solution_associations = session.query(ImageTaskAssociation).filter_by(solution_id=solution.id).all()
                
                # Retrieve image information for each associated image ID
                for association in image_solution_associations:
                    image_id = association.image_id
                    image = session.query(Image).get(image_id)
                    if image:
                        image_info = {
                            'id': image.id,
                            'title': image.title,
                            'description': image.description,
                            # Add other image attributes as needed
                        }
                        problem_info['images'].append(image_info)

            
            # Retrieve associated documents using the CompleteDocument model
            documents = session.query(CompleteDocument).join(Problem.complete_documents).filter(Problem.id == problem.id).all()

            serialized_documents = []
            for document in documents:
                serialized_document = {
                    'id': document.id,
                    'title': document.title                    
                }
                serialized_documents.append(serialized_document)


            # Add solutions, images, and documents to the problem info
            problem_info['solutions'] = solutions
            problem_info['documents'] = serialized_documents

            # Append the problem info to the response list
            response.append(problem_info)

        # Log the generated SQL query after applying filters
        print(f"Generated SQL query: {str(query)}")

        
        
        # Construct HTML content for problem, solution, and documents
        html_content = ""
        for problem_info in response:
            # Add problem and solution
            html_content += f"<h3>Problem:</h3><p>{problem_info['description']}</p>"
            html_content += "<h3>Solutions:</h3>"
            for solution in problem_info['solutions']:
                html_content += f"<p>{solution['description']}</p>"

            # Add associated documents
            html_content += "<h3>Associated Documents:</h3><ul>"
            for document in problem_info['documents']:
                # Generate the document link using url_for
                document_link = url_for('search_documents_bp.view_document', document_id=document['id'])
                html_content += f"<li><a href='{document_link}'>{document['title']}</a></li>"
            html_content += "</ul>"

             # Add associated images
            html_content += "<h3>Associated Images:</h3>"
            for image in problem_info['images']:
                # Generate the image link using url_for
                image_link = url_for('serve_image_route', image_id=image['id'])
                # Construct the HTML for the clickable image
                html_content += f"""
                    <div class="image-details">
                        <a href="{image_link}">
                            <img class="thumbnail" src="{image_link}" alt="{image['title']}">
                        </a>
                        <div class="description">
                            <h2>{image['title']}</h2>
                            <p>{image['description']}</p>
                        </div>
                        <div style="clear: both;"></div>
                    </div>
                """

        # Return the HTML content as a response
        return html_content

        
    except SQLAlchemyError as e:
        # Handle any SQLAlchemy errors
        print(f"An error occurred while retrieving problems: {e}")
        flash("An error occurred while retrieving problems", "error")
        return jsonify(problems=[])

    finally:
        session.close()  # Close the session in the finally block
        
        

def tsg_generate_html_links(documents):
    html_links = []
    for document in documents:
        # Retrieve the link and title from the dictionary
        link = document.get('link', '')  # Get the 'link' value or an empty string if it doesn't exist
        title = document.get('title', '')  # Get the 'title' value or an empty string if it doesn't exist
        
        # Debug print statements to check the values of link and title
        print(f"Debug: Document link: {link}")
        print(f"Debug: Document title: {title}")

        # Create HTML anchor tag with the document title as the link text and the document link as the href attribute
        html_link = f"<a href='{link}'>{title}</a>"
        html_links.append(html_link)
    return html_links

def query_association_details_by_id(id_value, id_type):
    """
    Query details by a given ID and its type.

    :param id_value: The ID value to query by.
    :param id_type: The type of ID (e.g., 'solution_id', 'image_id', 'part_id', 'location_id', 'drawing_id').
    :return: JSON response with the queried details.
    """
    # Create a session
    session = Session()
    try:
        if id_type not in ['solution_id', 'image_id', 'part_id', 'location_id', 'drawing_id']:
            return jsonify(error="Invalid id_type provided"), 400

        # Dynamically construct the filter
        filter_arg = {f"{id_type}": id_value}
        
        # Query the association table based on dynamic ID and type
        details = session.query(
            drawing_part_image_model_location_association
        ).filter_by(**filter_arg).all()

        # Check if no details found
        if not details:
            return jsonify(error=f"No details found for the provided {id_type}"), 404

        # Prepare and return JSON data with the fetched details
        results = [{'id': detail.id, 'solution_id': detail.solution_id, 'image_id': detail.image_id,
                    'part_id': detail.part_id, 'location_id': detail.location_id, 'drawing_id': detail.drawing_id}
                   for detail in details]
        return jsonify(association_details=results)
    except SQLAlchemyError as e:
        return jsonify(error=str(e)), 500
    finally:
        session.close()