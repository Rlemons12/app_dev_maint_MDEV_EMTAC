import logging
from flask import Blueprint, request, render_template, send_file, url_for
from modules.emtacdb.emtacdb_fts import CompleteDocument
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, text
import os
from modules.configuration.config import DATABASE_URL, DATABASE_DIR

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a SQLAlchemy engine using the DATABASE_URL from your config
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = sessionmaker(bind=engine)

search_documents_fts_bp = Blueprint('search_documents_fts_bp', __name__)

@search_documents_fts_bp.route('/', methods=['GET'])
def search_documents_fts():
    query = request.args.get('query', '')
    logger.debug("Received search query: %s", query)

    # Create a SQLAlchemy session
    session = Session()

    try:
        # Construct the full-text search query in documents_fts table for the title
        title_search_query = text(
            "SELECT title FROM documents_fts WHERE title MATCH :query"
        )
        logger.debug("Executing title search query: %s", title_search_query)

        # Execute the query with the user-provided search query
        title_search_results = session.execute(title_search_query, {"query": query})
        logger.debug("Title search query executed.")

        # Fetch the search results
        titles = [row.title for row in title_search_results]
        logger.debug("Found titles: %s", titles)

        # Fetch file paths from complete_documents table based on matched titles
        documents = []
        for title in titles:
            logger.debug("Searching complete_documents for title: %s", title)
            # Retrieve the corresponding document from the complete_documents table
            document = session.query(CompleteDocument).filter_by(title=title).first()
            if document:
                # Construct the full URL for viewing the document
                document.link = url_for('search_documents_fts_bp.view_document', document_id=document.id)
                documents.append(document)
                logger.debug("Document found: %s", document)
            else:
                logger.debug("No document found for title: %s", title)

        if documents:
            logger.info("Found %d documents matching the query.", len(documents))
        else:
            logger.info("No documents found matching the query.")
        
    except Exception as e:
        logger.error("An error occurred during the search process: %s", e)
        session.rollback()
        raise e

    finally:
        # Close the session
        session.close()
        logger.info("Session closed after search.")

    if documents:
        # Render the template with clickable links for each document
        return render_template('document_results.html', documents=documents)
    else:
        return "No documents found", 404


# Define the view_document route within the blueprint
@search_documents_fts_bp.route('/view_document/<int:document_id>')
def view_document(document_id):
    logger.info("Viewing document with ID: %d", document_id)

    # Create an SQLAlchemy session
    session = Session()

    try:
        # Fetch the document from the database based on the ID
        document = session.query(CompleteDocument).get(document_id)

        if document:
            file_path = os.path.join(DATABASE_DIR, document.file_path)
            logger.debug("Document file path: %s", file_path)
            if os.path.exists(file_path):
                logger.info("Serving file: %s", file_path)
                return send_file(file_path)
            else:
                logger.warning("File not found at path: %s", file_path)
                return "File not found", 404
        else:
            logger.warning("Document not found in database with ID: %d", document_id)
            return "Document not found", 404

    except Exception as e:
        logger.error("An error occurred while retrieving the document: %s", e)
        raise e

    finally:
        session.close()
        logger.info("Session closed after viewing document.")
