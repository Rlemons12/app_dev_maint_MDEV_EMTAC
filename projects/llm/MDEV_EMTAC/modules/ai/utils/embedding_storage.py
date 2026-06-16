"""Embedding generation, storage, and similarity helpers."""


from __future__ import annotations


from modules.ai.config import ModelsConfig
from modules.ai.models.embedding import NoEmbeddingModel
from modules.configuration.log_config import logger, with_request_id

def generate_embedding(document_content, model_name=None):
    """Generate embeddings for document content using the specified model."""
    logger.info(f"Starting generate_embedding")
    logger.debug(f"Document content length: {len(document_content)}")

    try:
        embedding_model = ModelsConfig.load_embedding_model(model_name)

        # If we got NoEmbeddingModel, embeddings are disabled
        if isinstance(embedding_model, NoEmbeddingModel):
            logger.info("Embeddings are currently disabled.")
            return None

        embeddings = embedding_model.get_embeddings(document_content)
        logger.info(f"Successfully generated embedding with {len(embeddings) if embeddings else 0} dimensions")
        return embeddings
    except Exception as e:
        logger.error(f"An error occurred while generating embedding: {e}")
        return None


def store_embedding_enhanced(session, document_id, embeddings, model_name=None):
    """
    Enhanced store embeddings function with pgvector support and transaction safety.
    **UPDATED** for pgvector DocumentEmbedding class compatibility.

    Args:
        session: Database session (REQUIRED - matches framework pattern)
        document_id: ID of the document
        embeddings: List of embedding values
        model_name: Name of the model used (optional)

    Returns:
        bool: Success status
    """
    if model_name is None:
        model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

    logger.info(f"Storing pgvector embedding for model {model_name} and document ID {document_id}")

    if embeddings is None or len(embeddings) == 0:
        logger.warning(f"No embeddings to store for document ID {document_id}")
        return False

    try:
        # Import DocumentEmbedding here to avoid circular import
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding

        # Use PostgreSQL savepoint for transaction safety (matches framework pattern)
        savepoint = session.begin_nested()
        try:
            # Check if embedding already exists
            existing = session.query(DocumentEmbedding).filter_by(
                document_id=document_id,
                model_name=model_name
            ).first()

            if existing:
                # Update existing embedding using the enhanced property
                existing.embedding_as_list = embeddings  # This uses the pgvector setter
                logger.info(f"Updated existing pgvector embedding for document ID {document_id}")
            else:
                # Create new embedding using the enhanced factory method
                document_embedding = DocumentEmbedding.create_with_pgvector(
                    document_id=document_id,
                    model_name=model_name,
                    embedding=embeddings
                )
                session.add(document_embedding)
                logger.info(f"Created new pgvector embedding for document ID {document_id}")

            session.flush()  # Flush within savepoint
            savepoint.commit()  # Commit savepoint
            return True

        except Exception as savepoint_error:
            savepoint.rollback()  # Rollback only the savepoint
            logger.error(f"Savepoint rolled back for pgvector embedding storage: {savepoint_error}")
            raise

    except Exception as e:
        logger.error(f"An error occurred while storing pgvector embedding: {e}")
        logger.exception("Exception details:")
        return False


def store_embedding(document_id, embeddings, model_name=None):
    """
    Legacy store embedding function for backward compatibility.
    Creates its own session - use store_embedding_enhanced() for better transaction safety.
    """
    logger.warning("store_embedding() is legacy - consider using store_embedding_enhanced() with existing session")

    try:
        from modules.configuration.config_env import DatabaseConfig
        db_config = DatabaseConfig()

        with db_config.main_session() as session:
            return store_embedding_enhanced(session, document_id, embeddings, model_name)

    except ImportError:
        logger.warning("DatabaseConfig not available, using fallback session")
        session = Session()
        try:
            result = store_embedding_enhanced(session, document_id, embeddings, model_name)
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()


def generate_and_store_embedding(session, document_id, document_content, model_name=None):
    """
    Combined function to generate and store embeddings using pgvector in one transaction.
    **UPDATED** for pgvector DocumentEmbedding class compatibility.

    Args:
        session: Database session (REQUIRED - matches framework pattern)
        document_id: ID of the document
        document_content: Text content to generate embeddings for
        model_name: Name of the model to use (optional)

    Returns:
        bool: Success status
    """
    logger.info(f"Generating and storing pgvector embedding for document ID {document_id}")

    try:
        # Generate embeddings
        embeddings = generate_embedding(document_content, model_name)

        if embeddings is None or len(embeddings) == 0:
            logger.warning(f"Failed to generate embeddings for document ID {document_id}")
            return False

        # Store embeddings using the updated pgvector method
        success = store_embedding_enhanced(session, document_id, embeddings, model_name)

        if success:
            logger.info(f"Successfully generated and stored pgvector embedding for document ID {document_id}")
        else:
            logger.error(f"Failed to store pgvector embedding for document ID {document_id}")

        return success

    except Exception as e:
        logger.error(f"Error in generate_and_store_embedding for document ID {document_id}: {e}")
        logger.exception("Exception details:")
        return False


def search_similar_embeddings(session, query_embeddings, model_name=None, limit=10, threshold=0.7):
    """
    Search for similar embeddings using pgvector cosine similarity.

    Args:
        session: Database session
        query_embeddings: Query embedding vector (list of floats)
        model_name: Embedding model name (optional)
        limit: Maximum number of results
        threshold: Minimum similarity threshold (0.0 to 1.0)

    Returns:
        List of similar documents with similarity scores
    """
    if model_name is None:
        model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

    logger.info(f"Searching similar embeddings with pgvector for model {model_name}")

    try:
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding
        from sqlalchemy import text

        # Convert query embeddings to pgvector format
        query_vector_str = '[' + ','.join(map(str, query_embeddings)) + ']'

        # Use pgvector cosine similarity operator (<=>)
        similarity_query = text("""
            SELECT
                de.document_id,
                de.model_name,
                de.embedding_vector <=> :query_vector AS distance,
                1 - (de.embedding_vector <=> :query_vector) AS similarity,
                de.created_at,
                de.updated_at
            FROM document_embedding de
            WHERE de.model_name = :model_name
              AND de.embedding_vector IS NOT NULL
              AND (1 - (de.embedding_vector <=> :query_vector)) >= :threshold
            ORDER BY de.embedding_vector <=> :query_vector ASC
            LIMIT :limit
        """)

        result = session.execute(similarity_query, {
            'query_vector': query_vector_str,
            'model_name': model_name,
            'threshold': threshold,
            'limit': limit
        })

        similar_embeddings = []
        for row in result:
            similar_embeddings.append({
                'document_id': row[0],
                'model_name': row[1],
                'distance': float(row[2]),
                'similarity': float(row[3]),
                'created_at': row[4].isoformat() if row[4] else None,
                'updated_at': row[5].isoformat() if row[5] else None
            })

        logger.info(f"Found {len(similar_embeddings)} similar embeddings above threshold {threshold}")
        return similar_embeddings

    except Exception as e:
        logger.error(f"pgvector similarity search failed: {e}")
        return []


def get_embedding_with_similarity(session, document_id, query_embeddings, model_name=None):
    """
    Get a specific embedding and calculate its similarity to a query.

    Args:
        session: Database session
        document_id: ID of the document
        query_embeddings: Query embedding vector for similarity calculation
        model_name: Embedding model name (optional)

    Returns:
        dict: Embedding info with similarity score, or None if not found
    """
    if model_name is None:
        model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

    try:
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding

        embedding = session.query(DocumentEmbedding).filter_by(
            document_id=document_id,
            model_name=model_name
        ).first()

        if not embedding:
            return None

        # Get the embedding as a list
        embedding_vector = embedding.embedding_as_list

        if not embedding_vector:
            return None

        # Calculate similarity using the enhanced method
        similarity = embedding.cosine_similarity(query_embeddings)

        return {
            'id': embedding.id,
            'document_id': embedding.document_id,
            'model_name': embedding.model_name,
            'embedding': embedding_vector,
            'similarity': similarity,
            'storage_type': embedding.get_storage_type(),
            'dimensions': len(embedding_vector),
            'created_at': embedding.created_at.isoformat() if embedding.created_at else None,
            'updated_at': embedding.updated_at.isoformat() if embedding.updated_at else None
        }

    except Exception as e:
        logger.error(f"Error getting embedding with similarity for document {document_id}: {e}")
        return None


def get_pgvector_statistics():
    """
    Get statistics about pgvector usage in the DocumentEmbedding table.

    Returns:
        dict: Statistics about embedding storage
    """
    try:
        from modules.configuration.config_env import DatabaseConfig
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding
        from sqlalchemy import func

        db_config = DatabaseConfig()
        with db_config.main_session() as session:

            total_embeddings = session.query(DocumentEmbedding).count()

            pgvector_embeddings = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.embedding_vector.isnot(None)
            ).count()

            legacy_embeddings = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.model_embedding.isnot(None),
                DocumentEmbedding.embedding_vector.is_(None)
            ).count()

            # Get model distribution for pgvector embeddings
            pgvector_models = session.query(
                DocumentEmbedding.model_name,
                func.count(DocumentEmbedding.id).label('count')
            ).filter(
                DocumentEmbedding.embedding_vector.isnot(None)
            ).group_by(DocumentEmbedding.model_name).all()

            statistics = {
                'total_embeddings': total_embeddings,
                'pgvector_embeddings': pgvector_embeddings,
                'legacy_embeddings': legacy_embeddings,
                'pgvector_percentage': (pgvector_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0,
                'pgvector_models': {model: count for model, count in pgvector_models},
                'needs_migration': legacy_embeddings > 0
            }

            logger.info(
                f"pgvector statistics: {pgvector_embeddings}/{total_embeddings} using pgvector ({statistics['pgvector_percentage']:.1f}%)")
            return statistics

    except Exception as e:
        logger.error(f"Failed to get pgvector statistics: {e}")
        return {}


def example_completeDocument_integration():
    """
    Example showing proper integration with updated pgvector DocumentEmbedding class.
    This demonstrates the correct usage patterns for the pgvector framework.
    """
    from modules.configuration.config_env import DatabaseConfig
    from modules.emtacdb.emtacdb_fts import DocumentEmbedding

    db_config = DatabaseConfig()

    with db_config.main_session() as session:
        # Example 1: Generate and store pgvector embedding for a document chunk
        document_id = 123
        content = "This is sample document content for pgvector embedding generation."

        success = generate_and_store_embedding(session, document_id, content)
        if success:
            print("pgvector embedding generated and stored successfully")

        # Example 2: Store pre-generated embeddings using pgvector
        embeddings = [0.1, 0.2, 0.3, 0.4, 0.5] * 307  # 1536 dimensions for OpenAI
        success = store_embedding_enhanced(session, document_id, embeddings, "OpenAIEmbeddingModel")
        if success:
            print("Pre-generated pgvector embedding stored successfully")

        # Example 3: Query stored pgvector embeddings
        embedding_record = session.query(DocumentEmbedding).filter_by(
            document_id=document_id,
            model_name="OpenAIEmbeddingModel"
        ).first()

        if embedding_record:
            stored_embeddings = embedding_record.embedding_as_list  # Uses pgvector property
            storage_type = embedding_record.get_storage_type()
            print(f"Retrieved {len(stored_embeddings)} dimension embedding using {storage_type}")

        # Example 4: Perform similarity search with pgvector
        query_embeddings = [0.1, 0.2, 0.3, 0.4, 0.5] * 307  # Query vector
        similar_docs = search_similar_embeddings(
            session, query_embeddings, "OpenAIEmbeddingModel", limit=5, threshold=0.8
        )
        print(f"Found {len(similar_docs)} similar documents using pgvector")

        # Example 5: Get embedding statistics
        stats = get_pgvector_statistics()
        print(f"pgvector usage: {stats.get('pgvector_percentage', 0):.1f}% of embeddings")
