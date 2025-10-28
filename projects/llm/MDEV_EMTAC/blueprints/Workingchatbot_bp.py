from flask import Blueprint, request, jsonify
import openai
from modules.emtacdb.emtacdb_fts import QandA, ChatSession
from modules.emtacdb.utlity.main_database.database import find_most_relevant_document, create_session, update_session, get_session
from datetime import datetime
import logging
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as LocalSession  # Import LocalSession
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Enable SQLAlchemy SQL statement logging
sqlalchemy_logger = logging.getLogger('sqlalchemy.engine')
sqlalchemy_logger.setLevel(logging.INFO)  # Adjust the log level as needed

DATABASE_PATH = 'sqlite:///C:/Users/10169062/Desktop/AI_EMTACr3r7/Database/emtac_db.db'

engine = create_engine(DATABASE_PATH)
LocalSession = scoped_session(sessionmaker(bind=engine))
session = Session()

chatbot_bp = Blueprint('chatbot_bp', __name__)

@chatbot_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_id = data.get('userId')
        question = data.get('question').strip()
        
        # Log incoming request parameters
        logger.debug(f"Received userId: {user_id}, question: {question}")
        
        # Check if the user wants to end the session
        if question.lower() == "end session please":
            end_session(user_id)
            return jsonify({'answer': 'Your session has been ended. Thank you!'})

        with LocalSession() as session:
            # Retrieve or create the session
            latest_session = get_session(user_id, session)
            if latest_session:
                #removed user from "\nUser"
                session_id = latest_session.session_id
                session_data = latest_session.session_data + "\n: " + question
            else:
                session_id = create_session(user_id, "User: " + question, session)
                #removed User: prefix
                session_data = question
            
            # Log session information
            logger.debug(f"Session ID: {session_id}, Session Data: {session_data}")
            
            # Find the most relevant document and generate a response
            relevant_doc = find_most_relevant_document(question, session)
            if relevant_doc:
                doc_content = relevant_doc.content
                  # Include the entire conversation history in the prompt
                  #removed Chat from "\nChatbot"
                conversation_history = get_conversation_history(session_id)
                prompt = f"Conversation Summary: {conversation_history}\n: {question}"
                response = openai.Completion.create(
                    engine="gpt-3.5-turbo-instruct",
                    prompt=get_prompt_with_summary(user_id, question),  # Include conversation summary
                    max_tokens=1000
                )
                answer = response.choices[0].text.strip()
                
                # Update the conversation summary with the new summary
                update_conversation_summary(session_id, answer)
            else:
                answer = "No relevant document found."
            
            # Log before updating session and Q&A
            logger.debug('About to update the session and save the Q&A')
            
            # Update the session and save the Q&A
            #removed Chatbot from "\nChatbot
            update_session(session_id, session_data + "\n: " + answer, session)
            now = datetime.now().isoformat()
            new_qanda = QandA(user_id=user_id, question=question, answer=answer, timestamp=now)
            session.add(new_qanda)
            session.commit()
            
            # Log completion of update
            logger.debug('Completed update of QandA')
            return jsonify({'answer': answer})

    except SQLAlchemyError as e:
        LocalSession.rollback()
        logger.error(f"Database error: {e}")
        return jsonify({'error': 'Database error occurred.'}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500


def end_session(user_id):
    try:
        with LocalSession() as local_session:
            latest_session = get_session(user_id, local_session)
            if latest_session:
                latest_session.is_active = False
                local_session.commit()
    except SQLAlchemyError as e:
        local_session.rollback()
        logger.error(f"Database error while ending session: {e}")
        
        # After each question-and-answer pair:
# Calculate the new conversation summary and update the database
def update_conversation_summary(session_id, new_summary):
    with LocalSession() as session:
        chat_session = session.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if chat_session:
            # Remove prefixes like "AI:" or "Bot:" from the new_summary
            new_summary = new_summary.replace("AI:", "").replace("Bot:", "")
            
            # Update the conversation summary
            chat_session.conversation_summary = new_summary
            session.commit()


# When receiving a new question:
# Retrieve the current conversation summary and include it in the prompt
def get_prompt_with_summary(user_id, question):
    with LocalSession() as session:
        chat_session = session.query(ChatSession).filter(ChatSession.user_id == user_id).order_by(ChatSession.start_time.desc()).first()
        if chat_session:
            conversation_summary = chat_session.conversation_summary
            #removed user from \n:user
            prompt = f"Conversation Summary: {conversation_summary}\n: {question}"
        else:
            prompt = f"User: {question}"
        return prompt

def get_conversation_history(session_id):
    try:
        with LocalSession() as session:
            chat_session = session.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if chat_session:
                return chat_session.session_data
            else:
                return ""
    except SQLAlchemyError as e:
        LocalSession.rollback()
        logger.error(f"Database error while getting conversation history: {e}")
        return ""
