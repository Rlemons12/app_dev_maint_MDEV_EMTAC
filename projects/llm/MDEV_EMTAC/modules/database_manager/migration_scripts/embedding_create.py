import os
from sqlalchemy import exists
from sqlalchemy.exc import SQLAlchemyError
from modules.configuration.config_env import DatabaseConfig
from plugins.ai_modules.ai_models import ModelsConfig
from modules.configuration.log_config import info_id, error_id, get_request_id
from modules.emtacdb.emtacdb_fts import Document, DocumentEmbedding
import json 

def generate_and_store_document_embedding(document, session=None, model_name=None):
    request_id = get_request_id()

    if not document or not document.content:
        error_id("Missing document content", request_id)
        return None

    try:
        if not model_name:
            model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')
        info_id(f"Using embedding model: {model_name}", request_id)

        embedding_model = ModelsConfig.load_embedding_model(model_name)
        if not embedding_model or not hasattr(embedding_model, 'get_embeddings'):
            error_id(f"Invalid embedding model: {model_name}", request_id)
            return None

        embedding = embedding_model.get_embeddings(document.content)
        if embedding is None:
            error_id(f"Embedding generation failed for document ID {document.id}", request_id)
            return None

        created_session = False
        if session is None:
            session = DatabaseConfig().get_main_session()
            created_session = True

        exists_query = session.query(
            exists().where(
                (DocumentEmbedding.document_id == document.id) &
                (DocumentEmbedding.model_name == model_name)
            )
        ).scalar()
        if exists_query:
            info_id(f"Skipping document ID {document.id}, already embedded by model {model_name}.", request_id)
            return None

        embedding_record = DocumentEmbedding.create_with_pgvector(
            document_id=document.id,
            model_name=model_name,
            embedding=embedding
        )

        session.add(embedding_record)
        session.commit()
        info_id(f"Embedding stored for document ID {document.id}", request_id)
        return embedding_record

    except SQLAlchemyError as e:
        session.rollback()
        error_id(f"Database error: {e}", request_id)
        return None

    finally:
        if created_session:
            session.close()


def prompt_user_for_model():
    """
    Prompt user to select an embedding model from the configured list.
    """
    raw = ModelsConfig.get_config_value('embedding', 'available_models', '[]')
    available = json.loads(raw) if isinstance(raw, str) else raw

    if not available:
        print("detected_intent_id = intent_classification['intent_id'] No embedding models configured.")
        return None

    print("\n📦 Available embedding models:")
    for i, model in enumerate(available):
        print(f"{i + 1}. {model['name']} - {'[ENABLED]' if model.get('enabled', False) else '[DISABLED]'}")

    while True:
        choice = input("👉 Enter the number of the model you want to use: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(available)):
            print("detected_intent_id = intent_classification['intent_id'] Invalid selection. Try again.")
            continue

        selected = available[int(choice) - 1]
        if not selected.get("enabled", False):
            print("detected_intent_id = intent_classification['intent_id']detected_intent_id = intent_classification['intent_id'] That model is disabled. Please choose an enabled model.")
            continue

        return selected["name"]


def embed_documents_interactively(batch_size=5):
    request_id = get_request_id()
    db = DatabaseConfig()

    model_name = prompt_user_for_model()
    if not model_name:
        print("Aborting: No valid embedding model selected.")
        return

    with db.main_session() as session:
        info_id(f"Preparing to embed with model: {model_name}", request_id)

        subquery = session.query(DocumentEmbedding.document_id).filter(
            DocumentEmbedding.model_name == model_name
        )

        documents = session.query(Document).filter(
            Document.content != None,
            ~Document.id.in_(subquery)
        ).order_by(Document.id).all()

        total = len(documents)
        print(f"📄 Found {total} document chunks needing embeddings for model: {model_name}")

        if total == 0:
            print("detected_intent_id = intent_classification['intent_id'] No document chunks need embedding.")
            return

        # First 5 test
        batch = documents[:batch_size]
        print(f"\n📦 Embedding test batch 1 to {len(batch)} of {total}...")

        for doc in batch:
            generate_and_store_document_embedding(doc, session=session, model_name=model_name)

        session.commit()

        if total <= batch_size:
            print("detected_intent_id = intent_classification['intent_id'] All document chunks have been processed.")
            return

        # Ask to embed the rest
        cont = input("❓ Do you want to embed ALL remaining chunks? (y/n): ").strip().lower()
        if cont not in ['y', 'yes']:
            print("detected_intent_id = intent_classification['intent_id'] Stopping after first 5 chunks.")
            return

        # Remaining documents
        remaining = documents[batch_size:]
        print(f"\n📦 Embedding remaining {len(remaining)} document chunks...")

        for doc in remaining:
            generate_and_store_document_embedding(doc, session=session, model_name=model_name)

        session.commit()
        print("detected_intent_id = intent_classification['intent_id'] All remaining document chunks processed.")


if __name__ == "__main__":
    embed_documents_interactively()
