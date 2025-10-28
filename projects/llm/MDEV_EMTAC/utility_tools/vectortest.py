# Vector Search Fix for Your PostgreSQL Schema
# This addresses the most likely issues based on your database structure

def fix_vector_search_issues():
    """Fix common vector search issues for your specific setup."""
    print("🔧 VECTOR SEARCH FIX")
    print("=" * 40)

    from modules.configuration.config_env import DatabaseConfig
    from sqlalchemy import text

    db_config = DatabaseConfig()
    with db_config.main_session() as session:

        # 1. Check if embeddings are in legacy format
        print("1. Checking embedding storage format...")

        counts = session.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(embedding_vector) as pgvector_count,
                COUNT(CASE WHEN model_embedding IS NOT NULL AND embedding_vector IS NULL THEN 1 END) as legacy_count
            FROM document_embedding
        """)).fetchone()

        total, pgvector_count, legacy_count = counts
        print(f"   Total: {total}, pgvector: {pgvector_count}, Legacy: {legacy_count}")

        if legacy_count > 0 and pgvector_count == 0:
            print("   ⚠️  Issue found: All embeddings are in legacy format!")
            print("   Solution: Migrate to pgvector format")
            return "migration_needed"

        # 2. Check model name consistency
        print("\n2. Checking model name consistency...")

        from plugins.ai_modules import ModelsConfig
        current_model = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

        model_counts = session.execute(text("""
            SELECT model_name, COUNT(*) 
            FROM document_embedding 
            WHERE embedding_vector IS NOT NULL
            GROUP BY model_name
            ORDER BY COUNT(*) DESC
        """)).fetchall()

        print(f"   Current search model: {current_model}")
        print("   Available models in database:")
        for model, count in model_counts:
            status = "✅ MATCH" if model == current_model else "❌ MISMATCH"
            print(f"     - {model}: {count} embeddings {status}")

        if not any(model == current_model for model, count in model_counts):
            if model_counts:
                most_common_model = model_counts[0][0]
                print(f"   ⚠️  Issue found: Current model '{current_model}' has no embeddings!")
                print(f"   Solution: Use model '{most_common_model}' which has {model_counts[0][1]} embeddings")
                return "model_mismatch", most_common_model

        # 3. Test similarity thresholds
        print("\n3. Testing similarity thresholds...")

        # Generate a test embedding
        from plugins.ai_modules import generate_embedding
        test_query = "enzyme protein"
        query_embeddings = generate_embedding(test_query, current_model)

        if query_embeddings:
            query_vector_str = '[' + ','.join(map(str, query_embeddings)) + ']'

            # Test different thresholds
            thresholds = [0.0, 0.1, 0.3, 0.5, 0.7]

            for threshold in thresholds:
                result_count = session.execute(text("""
                    SELECT COUNT(*)
                    FROM document_embedding de
                    WHERE de.model_name = :model_name
                      AND de.embedding_vector IS NOT NULL
                      AND (1 - (de.embedding_vector <=> :query_vector)) >= :threshold
                """), {
                    'query_vector': query_vector_str,
                    'model_name': current_model,
                    'threshold': threshold
                }).scalar()

                print(f"   Threshold {threshold}: {result_count} results")

                if result_count > 0 and threshold > 0.5:
                    print(f"   ✅ Good threshold found: {threshold}")
                    return "threshold_ok"

            if result_count == 0:
                print("   ⚠️  Issue found: No results even with threshold 0.0!")
                return "no_matches"

        return "unknown_issue"


def create_optimized_vector_search_client():
    """Create an optimized VectorSearchClient for your schema."""

    class OptimizedVectorSearchClient:
        """Optimized vector search client for your document_embedding schema."""

        def __init__(self):
            from modules.configuration.config_env import DatabaseConfig
            from plugins.ai_modules import ModelsConfig

            self.db_config = DatabaseConfig()
            self.current_model = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'OpenAIEmbeddingModel')

        def search(self, query, limit=10, threshold=0.1, model_override=None):
            """
            Optimized search method that handles your schema properly.

            Args:
                query: Search query string
                limit: Maximum results to return
                threshold: Minimum similarity threshold (lowered default)
                model_override: Override the current model

            Returns:
                List of search results with content and similarity scores
            """
            try:
                from plugins.ai_modules import generate_embedding
                from sqlalchemy import text

                # Use override model or current model
                search_model = model_override or self.current_model

                # Generate query embedding
                query_embeddings = generate_embedding(query, search_model)
                if not query_embeddings:
                    return []

                query_vector_str = '[' + ','.join(map(str, query_embeddings)) + ']'

                with self.db_config.main_session() as session:

                    # First, find which models actually have pgvector embeddings
                    available_models = session.execute(text("""
                        SELECT model_name, COUNT(*) as count
                        FROM document_embedding 
                        WHERE embedding_vector IS NOT NULL
                        GROUP BY model_name
                        ORDER BY count DESC
                    """)).fetchall()

                    if not available_models:
                        print("No pgvector embeddings found in database")
                        return []

                    # Try the search model first, then fall back to most common model
                    search_order = [search_model] + [model for model, count in available_models if
                                                     model != search_model]

                    for model_to_try in search_order:
                        try:
                            # Enhanced query that joins with document and complete_document for content
                            similarity_query = text("""
                                SELECT 
                                    de.document_id,
                                    de.model_name,
                                    1 - (de.embedding_vector <=> :query_vector) AS similarity,
                                    d.title,
                                    d.content,
                                    cd.title as doc_title,
                                    cd.file_path
                                FROM document_embedding de
                                LEFT JOIN document d ON de.document_id = d.id
                                LEFT JOIN complete_document cd ON d.complete_document_id = cd.id
                                WHERE de.model_name = :model_name
                                  AND de.embedding_vector IS NOT NULL
                                  AND (1 - (de.embedding_vector <=> :query_vector)) >= :threshold
                                ORDER BY de.embedding_vector <=> :query_vector ASC
                                LIMIT :limit
                            """)

                            result = session.execute(similarity_query, {
                                'query_vector': query_vector_str,
                                'model_name': model_to_try,
                                'threshold': threshold,
                                'limit': limit
                            })

                            results = []
                            for row in result:
                                doc_id, model_name, similarity, title, content, doc_title, file_path = row

                                # Create content from available sources
                                display_content = content or title or doc_title or f"Document {doc_id}"
                                if len(display_content) > 500:
                                    display_content = display_content[:500] + "..."

                                results.append({
                                    'id': doc_id,
                                    'content': display_content,
                                    'similarity': float(similarity),
                                    'model_name': model_name,
                                    'title': doc_title or title,
                                    'file_path': file_path,
                                    'source': 'document_embedding'
                                })

                            if results:
                                if model_to_try != search_model:
                                    print(
                                        f"Found {len(results)} results using model '{model_to_try}' instead of '{search_model}'")
                                return results

                        except Exception as e:
                            print(f"Error searching with model '{model_to_try}': {e}")
                            continue

                    return []

            except Exception as e:
                print(f"Vector search error: {e}")
                return []

    return OptimizedVectorSearchClient()


def apply_vector_search_fix():
    """Apply the appropriate fix based on diagnosis."""

    print("🔍 Diagnosing vector search issues...")
    result = fix_vector_search_issues()

    if result == "migration_needed":
        print("\n🔄 Applying migration fix...")
        # Apply migration here if needed
        print("Migration script provided separately")

    elif isinstance(result, tuple) and result[0] == "model_mismatch":
        print(f"\n🔧 Applying model fix...")
        _, correct_model = result
        print(f"Updating VectorSearchClient to use model: {correct_model}")

        # Create optimized client
        optimized_client = create_optimized_vector_search_client()

        # Test with correct model
        test_results = optimized_client.search("enzyme protein", model_override=correct_model)
        print(f"Test search with correct model: {len(test_results)} results")

        return optimized_client

    elif result == "no_matches":
        print("\n🔧 Applying threshold fix...")

        # Create client with very low threshold
        optimized_client = create_optimized_vector_search_client()
        test_results = optimized_client.search("enzyme protein", threshold=0.0)
        print(f"Test search with threshold 0.0: {len(test_results)} results")

        return optimized_client

    else:
        print("\n✅ Creating optimized client anyway...")
        return create_optimized_vector_search_client()


# Quick test function
def test_optimized_search():
    """Test the optimized vector search."""
    print("\n🧪 TESTING OPTIMIZED VECTOR SEARCH")
    print("=" * 40)

    client = apply_vector_search_fix()

    test_queries = ["enzyme", "protein", "cofactor", "what is enzyme"]

    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        results = client.search(query, limit=3, threshold=0.1)
        print(f"Results: {len(results)}")

        for i, result in enumerate(results[:2], 1):
            similarity = result.get('similarity', 0)
            content = result.get('content', '')[:100]
            print(f"  {i}. Similarity: {similarity:.4f} - {content}...")


if __name__ == "__main__":
    test_optimized_search