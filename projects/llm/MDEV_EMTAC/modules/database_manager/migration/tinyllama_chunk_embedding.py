# add_tinyllama_embeddings_from_chunks.py
"""
Create TinyLlama embeddings directly from document chunks in the 'document' table.
This will create variable-dimension embeddings alongside your existing 1536d OpenAI embeddings.
"""

from sentence_transformers import SentenceTransformer
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger, with_request_id, info_id, error_id, warning_id
from sqlalchemy import text
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import time


class DocumentChunkEmbeddingService:
    """
    Service to create TinyLlama embeddings from document chunks
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model

        Popular choices:
        - "all-MiniLM-L6-v2" (384 dimensions, fast, good general purpose)
        - "all-mpnet-base-v2" (768 dimensions, higher quality)
        - "paraphrase-MiniLM-L6-v2" (384 dimensions, good for paraphrases)
        - "multi-qa-MiniLM-L6-cos-v1" (384 dimensions, optimized for Q&A)
        """
        self.model_name = model_name
        self.st_model = None
        self.db_config = DatabaseConfig()
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.st_model = SentenceTransformer(self.model_name)
            dimensions = self.st_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            logger.info(f"Model embedding dimensions: {dimensions}")
            return True
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {e}")
            raise

    @with_request_id
    def get_documents_without_tinyllama_embeddings(self, limit: Optional[int] = None, request_id=None) -> List[Dict]:
        """
        Get document chunks that don't have TinyLlama embeddings yet

        Args:
            limit: Maximum number of documents to return
            request_id: Request ID for logging

        Returns:
            List of document dictionaries
        """
        try:
            with self.db_config.main_session() as session:
                # Find documents that have content but no TinyLlama embeddings
                query = """
                SELECT d.id, d.name, d.content, d.complete_document_id, d.file_path
                FROM document d
                WHERE d.content IS NOT NULL 
                  AND d.content != ''
                  AND d.id NOT IN (
                      SELECT DISTINCT de.document_id 
                      FROM document_embedding de 
                      WHERE de.model_name LIKE %s
                  )
                ORDER BY d.id
                """

                if limit:
                    query += f" LIMIT {limit}"

                # Look for any TinyLlama-style model names
                tinyllama_pattern = f"%tinyllama%"

                results = session.execute(text(query), (tinyllama_pattern,)).fetchall()

                documents = []
                for row in results:
                    documents.append({
                        'id': row.id,
                        'name': row.name,
                        'content': row.content,
                        'complete_document_id': row.complete_document_id,
                        'file_path': row.file_path,
                        'content_length': len(row.content) if row.content else 0
                    })

                info_id(f"Found {len(documents)} documents without TinyLlama embeddings", request_id)
                return documents

        except Exception as e:
            error_id(f"Error getting documents without TinyLlama embeddings: {e}", request_id)
            return []

    @with_request_id
    def generate_embedding(self, text: str, request_id=None) -> Optional[List[float]]:
        """
        Generate embedding for text

        Args:
            text: Input text to embed
            request_id: Request ID for logging

        Returns:
            List of floats representing the embedding, or None if error
        """
        if not self.st_model:
            error_id("Model not loaded", request_id)
            return None

        try:
            # Truncate very long texts to avoid memory issues
            if len(text) > 8000:  # Reasonable limit for most sentence transformers
                text = text[:8000] + "..."
                warning_id(f"Text truncated to 8000 characters", request_id)

            embedding = self.st_model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            error_id(f"Error generating embedding: {e}", request_id)
            return None

    @with_request_id
    def store_embedding(self, document_id: int, embedding: List[float],
                        model_identifier: str = None, request_id=None) -> Optional[int]:
        """
        Store embedding in the document_embedding table

        Args:
            document_id: Document ID
            embedding: Embedding vector
            model_identifier: Custom model identifier
            request_id: Request ID for logging

        Returns:
            Embedding ID if successful, None otherwise
        """
        if not model_identifier:
            model_identifier = f"tinyllama_{self.model_name.replace('/', '_').replace('-', '_')}"

        try:
            with self.db_config.main_session() as session:
                # Check if this document already has this type of embedding
                check_sql = """
                SELECT id FROM document_embedding 
                WHERE document_id = :document_id AND model_name = :model_name
                """

                existing = session.execute(text(check_sql), {
                    'document_id': document_id,
                    'model_name': model_identifier
                }).fetchone()

                if existing:
                    warning_id(f"Document {document_id} already has embedding with model {model_identifier}",
                               request_id)
                    return existing.id

                # Insert new embedding using your migrated table structure
                insert_sql = """
                INSERT INTO document_embedding 
                (document_id, model_name, embedding_vector, created_at, updated_at)
                VALUES (:document_id, :model_name, :embedding_vector, NOW(), NOW())
                RETURNING id
                """

                result = session.execute(text(insert_sql), {
                    'document_id': document_id,
                    'model_name': model_identifier,
                    'embedding_vector': embedding
                })

                embedding_id = result.fetchone()[0]
                session.commit()

                info_id(
                    f"Stored TinyLlama embedding: doc_id={document_id}, embedding_id={embedding_id}, dims={len(embedding)}",
                    request_id)
                return embedding_id

        except Exception as e:
            error_id(f"Error storing embedding for document {document_id}: {e}", request_id)
            return None

    @with_request_id
    def process_single_document(self, document: Dict, request_id=None) -> bool:
        """
        Process a single document to create TinyLlama embedding

        Args:
            document: Document dictionary with id, content, etc.
            request_id: Request ID for logging

        Returns:
            True if successful, False otherwise
        """
        try:
            doc_id = document['id']
            content = document['content']

            if not content or not content.strip():
                warning_id(f"Document {doc_id} has no content to embed", request_id)
                return False

            # Generate embedding
            embedding = self.generate_embedding(content, request_id)
            if not embedding:
                error_id(f"Failed to generate embedding for document {doc_id}", request_id)
                return False

            # Store embedding
            embedding_id = self.store_embedding(doc_id, embedding, request_id=request_id)
            if not embedding_id:
                error_id(f"Failed to store embedding for document {doc_id}", request_id)
                return False

            info_id(f"Successfully processed document {doc_id}: {len(embedding)} dimensions", request_id)
            return True

        except Exception as e:
            error_id(f"Error processing document {document.get('id', 'unknown')}: {e}", request_id)
            return False

    @with_request_id
    def batch_process_documents(self, batch_size: int = 50, total_limit: Optional[int] = None, request_id=None) -> Dict:
        """
        Process multiple documents in batches

        Args:
            batch_size: Number of documents to process in each batch
            total_limit: Maximum total documents to process (None for all)
            request_id: Request ID for logging

        Returns:
            Dictionary with processing results
        """
        results = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'processing_time': 0,
            'avg_time_per_doc': 0
        }

        start_time = time.time()

        try:
            # Get documents to process
            documents = self.get_documents_without_tinyllama_embeddings(limit=total_limit, request_id=request_id)

            if not documents:
                info_id("No documents found that need TinyLlama embeddings", request_id)
                return results

            info_id(f"Starting batch processing of {len(documents)} documents", request_id)

            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size

                info_id(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)", request_id)

                for doc in batch:
                    try:
                        if self.process_single_document(doc, request_id):
                            results['successful'] += 1
                        else:
                            results['failed'] += 1
                            results['errors'].append(f"Document {doc['id']}: Processing failed")

                        results['total_processed'] += 1

                        # Progress update every 10 documents
                        if results['total_processed'] % 10 == 0:
                            progress = (results['total_processed'] / len(documents)) * 100
                            info_id(f"Progress: {progress:.1f}% ({results['total_processed']}/{len(documents)})",
                                    request_id)

                    except Exception as e:
                        results['failed'] += 1
                        results['total_processed'] += 1
                        error_msg = f"Document {doc['id']}: {str(e)}"
                        results['errors'].append(error_msg)
                        error_id(error_msg, request_id)

                # Small delay between batches to not overwhelm the system
                if batch_num < total_batches:
                    time.sleep(0.1)

            # Calculate final statistics
            end_time = time.time()
            results['processing_time'] = round(end_time - start_time, 2)
            if results['total_processed'] > 0:
                results['avg_time_per_doc'] = round(results['processing_time'] / results['total_processed'], 3)

            info_id(f"Batch processing completed: {results['successful']} successful, {results['failed']} failed",
                    request_id)
            info_id(f"Total time: {results['processing_time']}s, Avg per doc: {results['avg_time_per_doc']}s",
                    request_id)

            return results

        except Exception as e:
            error_id(f"Error in batch processing: {e}", request_id)
            results['errors'].append(f"Batch processing error: {str(e)}")
            return results

    @with_request_id
    def get_statistics(self, request_id=None) -> Dict:
        """
        Get statistics about TinyLlama embeddings

        Args:
            request_id: Request ID for logging

        Returns:
            Dictionary with statistics
        """
        try:
            with self.db_config.main_session() as session:
                # Count total documents
                total_docs = session.execute(text("""
                    SELECT COUNT(*) FROM document 
                    WHERE content IS NOT NULL AND content != ''
                """)).fetchone()[0]

                # Count documents with TinyLlama embeddings
                tinyllama_docs = session.execute(text("""
                    SELECT COUNT(DISTINCT de.document_id) 
                    FROM document_embedding de 
                    WHERE de.model_name LIKE '%tinyllama%'
                """)).fetchone()[0]

                # Count documents with OpenAI embeddings
                openai_docs = session.execute(text("""
                    SELECT COUNT(DISTINCT de.document_id) 
                    FROM document_embedding de 
                    WHERE de.model_name LIKE '%OpenAI%'
                """)).fetchone()[0]

                # Get model breakdown
                model_stats = session.execute(text("""
                    SELECT 
                        model_name,
                        COUNT(*) as count,
                        AVG(array_length(string_to_array(trim(both '[]' from embedding_vector::text), ','), 1)) as avg_dimensions
                    FROM document_embedding 
                    WHERE model_name LIKE '%tinyllama%'
                    GROUP BY model_name
                    ORDER BY count DESC
                """)).fetchall()

                stats = {
                    'total_documents_with_content': total_docs,
                    'documents_with_tinyllama_embeddings': tinyllama_docs,
                    'documents_with_openai_embeddings': openai_docs,
                    'documents_needing_tinyllama': total_docs - tinyllama_docs,
                    'completion_percentage': round((tinyllama_docs / total_docs * 100), 2) if total_docs > 0 else 0,
                    'model_breakdown': {}
                }

                for row in model_stats:
                    stats['model_breakdown'][row.model_name] = {
                        'count': row.count,
                        'avg_dimensions': round(row.avg_dimensions, 0) if row.avg_dimensions else 0
                    }

                return stats

        except Exception as e:
            error_id(f"Error getting statistics: {e}", request_id)
            return {'error': str(e)}

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.st_model:
            return {"status": "not_loaded"}

        return {
            "model_name": self.model_name,
            "dimensions": self.st_model.get_sentence_embedding_dimension(),
            "status": "loaded",
            "model_type": "sentence_transformer"
        }


# Convenience functions for easy usage
@with_request_id
def create_tinyllama_embeddings_for_all_documents(model_name: str = "all-MiniLM-L6-v2",
                                                  batch_size: int = 50,
                                                  limit: Optional[int] = None,
                                                  request_id=None) -> Dict:
    """
    Create TinyLlama embeddings for all documents that don't have them

    Args:
        model_name: SentenceTransformer model to use
        batch_size: Documents to process per batch
        limit: Maximum documents to process (None for all)
        request_id: Request ID for logging

    Returns:
        Processing results dictionary
    """
    service = DocumentChunkEmbeddingService(model_name)
    return service.batch_process_documents(batch_size, limit, request_id)


@with_request_id
def get_tinyllama_statistics(request_id=None) -> Dict:
    """
    Get statistics about TinyLlama embeddings

    Args:
        request_id: Request ID for logging

    Returns:
        Statistics dictionary
    """
    service = DocumentChunkEmbeddingService()
    return service.get_statistics(request_id)


@with_request_id
def create_embeddings_for_specific_documents(document_ids: List[int],
                                             model_name: str = "all-MiniLM-L6-v2",
                                             request_id=None) -> Dict:
    """
    Create TinyLlama embeddings for specific document IDs

    Args:
        document_ids: List of document IDs to process
        model_name: SentenceTransformer model to use
        request_id: Request ID for logging

    Returns:
        Processing results dictionary
    """
    service = DocumentChunkEmbeddingService(model_name)

    results = {
        'total_processed': 0,
        'successful': 0,
        'failed': 0,
        'errors': []
    }

    try:
        with service.db_config.main_session() as session:
            for doc_id in document_ids:
                # Get document content
                doc_query = """
                SELECT id, name, content, complete_document_id, file_path
                FROM document 
                WHERE id = :doc_id AND content IS NOT NULL AND content != ''
                """

                doc_result = session.execute(text(doc_query), {'doc_id': doc_id}).fetchone()

                if not doc_result:
                    results['failed'] += 1
                    results['errors'].append(f"Document {doc_id}: Not found or has no content")
                    continue

                document = {
                    'id': doc_result.id,
                    'name': doc_result.name,
                    'content': doc_result.content,
                    'complete_document_id': doc_result.complete_document_id,
                    'file_path': doc_result.file_path
                }

                if service.process_single_document(document, request_id):
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Document {doc_id}: Processing failed")

                results['total_processed'] += 1

        info_id(
            f"Processed {len(document_ids)} specific documents: {results['successful']} successful, {results['failed']} failed",
            request_id)
        return results

    except Exception as e:
        error_id(f"Error processing specific documents: {e}", request_id)
        results['errors'].append(f"Processing error: {str(e)}")
        return results


# Usage examples and main execution
if __name__ == "__main__":
    print("TinyLlama Embedding Service for Document Chunks")
    print("=" * 50)

    # Get current statistics
    print("\n1. Current Statistics:")
    stats = get_tinyllama_statistics()
    print(f"   Total documents with content: {stats.get('total_documents_with_content', 0)}")
    print(f"   Documents with OpenAI embeddings: {stats.get('documents_with_openai_embeddings', 0)}")
    print(f"   Documents with TinyLlama embeddings: {stats.get('documents_with_tinyllama_embeddings', 0)}")
    print(f"   Documents needing TinyLlama: {stats.get('documents_needing_tinyllama', 0)}")
    print(f"   Completion percentage: {stats.get('completion_percentage', 0)}%")

    # Option to process all remaining documents
    if stats.get('documents_needing_tinyllama', 0) > 0:
        user_input = input(
            f"\nDo you want to create TinyLlama embeddings for {stats['documents_needing_tinyllama']} documents? (y/n): ")

        if user_input.lower() == 'y':
            print("\n2. Creating TinyLlama embeddings...")

            # Choose model
            print("\nAvailable models:")
            print("1. all-MiniLM-L6-v2 (384d, fast)")
            print("2. all-mpnet-base-v2 (768d, higher quality)")
            print("3. paraphrase-MiniLM-L6-v2 (384d, good for paraphrases)")

            model_choice = input("Choose model (1-3, default 1): ").strip()

            model_map = {
                '1': 'all-MiniLM-L6-v2',
                '2': 'all-mpnet-base-v2',
                '3': 'paraphrase-MiniLM-L6-v2'
            }

            selected_model = model_map.get(model_choice, 'all-MiniLM-L6-v2')
            print(f"Using model: {selected_model}")

            # Process documents
            results = create_tinyllama_embeddings_for_all_documents(
                model_name=selected_model,
                batch_size=20,  # Smaller batches for initial run
                limit=100  # Process only 100 documents initially
            )

            print(f"\n3. Processing Results:")
            print(f"   Total processed: {results['total_processed']}")
            print(f"   Successful: {results['successful']}")
            print(f"   Failed: {results['failed']}")
            print(f"   Processing time: {results['processing_time']}s")
            print(f"   Average time per document: {results['avg_time_per_doc']}s")

            if results['errors']:
                print(f"\n   Errors:")
                for error in results['errors'][:5]:  # Show first 5 errors
                    print(f"     - {error}")
                if len(results['errors']) > 5:
                    print(f"     ... and {len(results['errors']) - 5} more errors")

        else:
            print("Skipping embedding creation.")

    else:
        print("\nAll documents already have TinyLlama embeddings!")

    # Show final statistics
    print("\n4. Final Statistics:")
    final_stats = get_tinyllama_statistics()
    if 'model_breakdown' in final_stats:
        for model, info in final_stats['model_breakdown'].items():
            print(f"   {model}: {info['count']} embeddings, {info['avg_dimensions']} dimensions")