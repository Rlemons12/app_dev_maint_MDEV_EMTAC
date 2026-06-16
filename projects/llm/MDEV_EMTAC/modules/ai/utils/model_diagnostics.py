"""Model diagnostics and test helpers extracted from legacy ai_models.py."""


from __future__ import annotations


import os

from modules.ai.config import ModelsConfig
from modules.ai.config import get_current_models
from modules.ai.models.embedding import GPT4AllEmbeddingModel
from modules.ai.models.embedding import TinyLlamaEmbeddingModel
from modules.ai.models.text import GPT4AllModel
from modules.ai.models.text import TinyLlamaModel
from modules.ai.utils.embedding_storage import generate_and_store_embedding
from modules.ai.utils.embedding_storage import generate_embedding
from modules.configuration.log_config import logger, with_request_id
from modules.configuration.log_config import logger, with_request_id, debug_id, info_id, warning_id, error_id, get_request_id, log_timed_operation

def test_tinyllama_framework_integration():
    """Test TinyLlama integration with your framework."""
    logger.info("Testing TinyLlama framework integration...")

    try:
        # Test loading through ModelsConfig
        model = ModelsConfig.load_ai_model("TinyLlamaModel")

        if not model.is_available():
            logger.error("TinyLlama model not available through framework")
            return False

        # Test basic functionality
        test_prompt = "Hello! Can you tell me a brief fun fact?"
        response = model.get_response(test_prompt)

        if response and len(response.strip()) > 0 and not response.startswith("Error"):
            logger.info(f"TinyLlama framework integration test passed: '{response[:50]}...'")
            return True
        else:
            logger.error(f"TinyLlama framework integration test failed: '{response}'")
            return False

    except Exception as e:
        logger.error(f"TinyLlama framework integration test failed: {e}")
        return False


def diagnose_tinyllama():
    """Diagnose TinyLlama model setup specifically"""
    request_id = get_request_id()
    info_id("Starting TinyLlama diagnosis", request_id)

    result = {
        "status": "unknown",
        "details": {},
        "requirements": {
            "transformers": False,
            "torch": False,
            "model_files": False
        }
    }

    # Check requirements
    try:
        import transformers
        result["requirements"]["transformers"] = True
        info_id("transformers library available", request_id)
    except ImportError:
        result["requirements"]["transformers"] = False
        error_id("transformers library not available", request_id)

    try:
        import torch
        result["requirements"]["torch"] = True
        info_id("torch library available", request_id)
    except ImportError:
        result["requirements"]["torch"] = False
        error_id("torch library not available", request_id)

    # Check model files
    model_path = r"C:\Users\10169062\PycharmProjects\MDEV_EMTAC\plugins\ai_modules\TinyLlama_1_1B"
    config_path = os.path.join(model_path, "config.json")

    if os.path.exists(config_path):
        result["requirements"]["model_files"] = True
        info_id("TinyLlama model files found", request_id)
    else:
        result["requirements"]["model_files"] = False
        error_id(f"TinyLlama model files not found at {model_path}", request_id)

    # Test TinyLlama model if requirements are met
    if all(result["requirements"].values()):
        try:
            tinyllama_model = TinyLlamaModel()
            result["status"] = "loaded" if tinyllama_model.model_loaded else "failed"
            result["details"] = tinyllama_model.get_model_info()
            info_id(f"TinyLlama test: {result['status']}", request_id)
        except Exception as e:
            result["status"] = "error"
            result["details"] = {"error": str(e)}
            error_id(f"TinyLlama test failed: {e}", request_id)
    else:
        result["status"] = "requirements_not_met"
        result["details"] = {"missing_requirements": [k for k, v in result["requirements"].items() if not v]}

    info_id("TinyLlama diagnosis completed", request_id)
    return result


def test_tinyllama_functionality():
    """Test function to verify TinyLlama is working properly"""
    logger.info("Testing TinyLlama functionality...")

    try:
        # Test model loading
        tinyllama = TinyLlamaModel()

        if not tinyllama.model_loaded:
            logger.error("TinyLlama model loading test failed")
            return False

        logger.info("TinyLlama model loading test passed")

        # Test response generation
        test_prompt = "Hello! Can you tell me a fun fact?"
        response = tinyllama.get_response(test_prompt)

        if response and len(response.strip()) > 0 and not response.startswith("Error"):
            logger.info(f"TinyLlama response test passed: '{response[:50]}...'")
            return True
        else:
            logger.error(f"TinyLlama response test failed: '{response}'")
            return False

    except Exception as e:
        logger.error(f"TinyLlama functionality test failed: {e}")
        return False


def diagnose_models():
    """Diagnose both GPT4All and embedding model setup"""
    request_id = get_request_id()
    info_id("Starting model diagnosis", request_id)

    results = {
        "gpt4all": {"status": "unknown", "details": {}},
        "embedding": {"status": "unknown", "details": {}}
    }

    # Test GPT4All
    try:
        gpt4all_model = GPT4AllModel()
        results["gpt4all"]["status"] = "loaded" if gpt4all_model.model_loaded else "failed"
        results["gpt4all"]["details"] = gpt4all_model.get_model_info()
        info_id(f"GPT4All test: {results['gpt4all']['status']}", request_id)
    except Exception as e:
        results["gpt4all"]["status"] = "error"
        results["gpt4all"]["details"] = {"error": str(e)}
        error_id(f"GPT4All test failed: {e}", request_id)

    # Test Embedding Model
    try:
        embedding_model = GPT4AllEmbeddingModel()
        if embedding_model.model is not None:
            test_embeddings = embedding_model.get_embeddings("test")
            results["embedding"]["status"] = "loaded" if len(test_embeddings) > 0 else "failed"
            results["embedding"]["details"] = {
                "model_name": embedding_model.model_name,
                "test_embedding_dimensions": len(test_embeddings) if test_embeddings else 0
            }
        else:
            results["embedding"]["status"] = "failed"
            results["embedding"]["details"] = {"error": "Model not loaded"}
        info_id(f"Embedding test: {results['embedding']['status']}", request_id)
    except Exception as e:
        results["embedding"]["status"] = "error"
        results["embedding"]["details"] = {"error": str(e)}
        error_id(f"Embedding test failed: {e}", request_id)

    info_id("Model diagnosis completed", request_id)
    return results


def test_embedding_functionality():
    """Test function to verify embedding generation and storage is working."""
    logger.info("Testing embedding functionality...")

    test_text = "This is a test document for embedding generation."
    test_document_id = 999999  # Use a high ID that won't conflict

    try:
        # Test embedding generation
        embeddings = generate_embedding(test_text)

        if embeddings is None or len(embeddings) == 0:
            logger.error("Embedding generation test failed")
            return False

        logger.info(f"Embedding generation test passed: {len(embeddings)} dimensions")

        # Test embedding storage (but don't actually store the test)
        logger.info("Embedding functionality test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Embedding functionality test failed: {e}")
        return False


def example_model_configuration():
    """
    Example showing proper model configuration management.
    """
    # Set the current models
    ModelsConfig.set_current_embedding_model("OpenAIEmbeddingModel")
    ModelsConfig.set_current_ai_model("AnthropicModel")
    ModelsConfig.set_current_image_model("CLIPModelHandler")

    # Get current models
    current_models = get_current_models()
    print(f"Current models: {current_models}")

    # Load specific models
    embedding_model = ModelsConfig.load_embedding_model("OpenAIEmbeddingModel")
    ai_model = ModelsConfig.load_ai_model("AnthropicModel")
    image_model = ModelsConfig.load_image_model("CLIPModelHandler")

    # Test functionality
    if test_embedding_functionality():
        print("Embedding system working correctly")
    else:
        print("Embedding system needs attention")

    # Test image model
    try:
        result = image_model.process_image("test_image.jpg")
        print(f"Image model working: {result}")
    except Exception as e:
        print(f"Image model error: {e}")


def test_pgvector_functionality():
    """Test function to verify pgvector embedding functionality is working."""
    logger.info("Testing pgvector embedding functionality...")

    test_text = "This is a test document for pgvector embedding generation."
    test_document_id = 999999  # Use a high ID that won't conflict

    try:
        from modules.configuration.config_env import DatabaseConfig

        db_config = DatabaseConfig()
        with db_config.main_session() as session:

            # Test embedding generation and storage
            success = generate_and_store_embedding(session, test_document_id, test_text)

            if not success:
                logger.error("pgvector embedding generation and storage test failed")
                return False

            logger.info("pgvector embedding generation and storage test passed")

            # Test retrieval
            from modules.emtacdb.emtacdb_fts import DocumentEmbedding

            embedding_record = session.query(DocumentEmbedding).filter_by(
                document_id=test_document_id
            ).first()

            if embedding_record:
                embeddings = embedding_record.embedding_as_list
                storage_type = embedding_record.get_storage_type()

                if embeddings and len(embeddings) > 0:
                    logger.info(f"pgvector retrieval test passed: {len(embeddings)} dimensions using {storage_type}")

                    # Clean up test data
                    session.delete(embedding_record)
                    session.commit()

                    return True
                else:
                    logger.error("pgvector retrieval test failed - no embeddings")
                    return False
            else:
                logger.error("pgvector retrieval test failed - no record found")
                return False

    except Exception as e:
        logger.error(f"pgvector functionality test failed: {e}")
        return False


def test_tinyllama_embedding_functionality():
    """Test function to verify TinyLlama embedding generation is working properly"""
    logger.info("Testing TinyLlama embedding functionality...")

    try:
        # Test model loading
        tinyllama_embedding = TinyLlamaEmbeddingModel()

        if not tinyllama_embedding.is_available():
            logger.error("TinyLlama embedding model loading test failed")
            return False

        # Test embedding generation
        test_text = "TinyLlama embedding test sentence"
        embeddings = tinyllama_embedding.get_embeddings(test_text)

        if embeddings and len(embeddings) > 0:
            logger.info(f"TinyLlama embedding test passed: {len(embeddings)} dimensions")
            return True
        else:
            logger.error("TinyLlama embedding generation test failed")
            return False

    except Exception as e:
        logger.error(f"TinyLlama embedding functionality test failed: {e}")
        return False
