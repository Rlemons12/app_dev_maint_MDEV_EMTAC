"""Model recommendation and availability helpers."""


from __future__ import annotations


import os

from modules.ai.config import ModelsConfig
from modules.ai.utils.model_downloads import get_available_local_models
from modules.configuration.log_config import logger, with_request_id

def switch_to_local_models():
    """Switch to local models for complete offline operation"""
    try:
        # Set GPT4All as current AI model
        local_models = get_available_local_models()

        if local_models["gpt4all"]:
            best_gpt4all = local_models["gpt4all"][0]["name"]  # Use first available
            ModelsConfig.set_config_value('ai', 'GPT4ALL_MODEL_FILE', best_gpt4all)
            ModelsConfig.set_current_ai_model('GPT4AllModel')
            logger.info(f"Switched to local GPT4All model: {best_gpt4all}")

        if local_models["sentence_transformer"]:
            best_st = local_models["sentence_transformer"][0]["name"]
            ModelsConfig.set_config_value('embedding', 'SENTENCE_TRANSFORMER_MODEL', best_st)
            ModelsConfig.set_current_embedding_model('GPT4AllEmbeddingModel')
            logger.info(f"Switched to local SentenceTransformer: {best_st}")

        return True
    except Exception as e:
        logger.error(f"Error switching to local models: {e}")
        return False


def check_model_availability():
    """Check which models are actually available on the system."""
    availability = {
        "ai_models": {},
        "embedding_models": {},
        "image_models": {}
    }

    # --- Check AI Models ---
    ai_models = [
        ("NoAIModel", "Always available"),
        ("OpenAIModel", "Requires OPENAI_API_KEY"),
        ("AnthropicModel", "Requires ANTHROPIC_API_KEY"),
        ("Llama3Model", "Requires transformers and model files"),
        ("GPT4AllModel", "Requires gpt4all and model files"),
        ("TinyLlamaModel", "Requires transformers and model files")
    ]

    for model_name, requirement in ai_models:
        try:
            if model_name == "NoAIModel":
                availability["ai_models"][model_name] = {"available": True, "status": "Always available"}

            elif model_name == "OpenAIModel":
                from modules.configuration.config import OPENAI_API_KEY
                available = bool(OPENAI_API_KEY and OPENAI_API_KEY.strip())
                status = "API key configured" if available else "API key missing"
                availability["ai_models"][model_name] = {"available": available, "status": status}

            elif model_name == "AnthropicModel":
                from modules.configuration.config import ANTHROPIC_API_KEY
                available = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.strip())
                status = "API key configured" if available else "API key missing"
                availability["ai_models"][model_name] = {"available": available, "status": status}

            elif model_name == "TinyLlamaModel":
                from modules.configuration.model_config import get_tinyllama_config
                config = get_tinyllama_config()
                model_exists = os.path.exists(os.path.join(config['model_path'], 'config.json'))

                try:
                    import transformers
                    import torch
                    transformers_available = True
                except ImportError:
                    transformers_available = False

                available = model_exists and transformers_available
                if available:
                    status = "Ready"
                elif not transformers_available:
                    status = "Missing transformers/torch"
                else:
                    status = f"Model files not found at {config['model_path']}"

                availability["ai_models"][model_name] = {"available": available, "status": status}

            elif model_name == "GPT4AllModel":
                try:
                    import gpt4all
                    from modules.configuration.config import GPT4ALL_MODELS_PATH
                    model_files = [f for f in os.listdir(GPT4ALL_MODELS_PATH) if f.endswith('.gguf')] \
                        if os.path.exists(GPT4ALL_MODELS_PATH) else []
                    available = len(model_files) > 0
                    status = f"Ready ({len(model_files)} models)" if available else "No model files found"
                except ImportError:
                    available = False
                    status = "gpt4all not installed"
                except Exception:
                    available = False
                    status = "Configuration error"

                availability["ai_models"][model_name] = {"available": available, "status": status}

            elif model_name == "Llama3Model":
                try:
                    import transformers
                    import torch
                    availability["ai_models"][model_name] = {"available": True, "status": "Dependencies available"}
                except ImportError:
                    availability["ai_models"][model_name] = {"available": False, "status": "Missing dependencies"}

        except Exception as e:
            availability["ai_models"][model_name] = {"available": False, "status": f"Error: {str(e)}"}

    # --- Check Embedding Models ---
    embedding_models = [
        ("NoEmbeddingModel", "Always available"),
        ("OpenAIEmbeddingModel", "Requires OPENAI_API_KEY"),
        ("GPT4AllEmbeddingModel", "Requires sentence-transformers"),
        ("TinyLlamaEmbeddingModel", "Requires sentence-transformers (TinyLlama optimized)")
    ]

    for model_name, requirement in embedding_models:
        try:
            if model_name == "NoEmbeddingModel":
                availability["embedding_models"][model_name] = {"available": True, "status": "Always available"}

            elif model_name == "OpenAIEmbeddingModel":
                from modules.configuration.config import OPENAI_API_KEY
                available = bool(OPENAI_API_KEY and OPENAI_API_KEY.strip())
                status = "API key configured" if available else "API key missing"
                availability["embedding_models"][model_name] = {"available": available, "status": status}

            elif model_name in ["GPT4AllEmbeddingModel", "TinyLlamaEmbeddingModel"]:
                try:
                    from sentence_transformers import SentenceTransformer

                    # Test loading a sample model
                    if model_name == "TinyLlamaEmbeddingModel":
                        SentenceTransformer("all-MiniLM-L6-v2")
                        status = "Ready (TinyLlama optimized)"
                    else:
                        SentenceTransformer("all-MiniLM-L6-v2")
                        status = "Ready"

                    availability["embedding_models"][model_name] = {"available": True, "status": status}
                except ImportError:
                    availability["embedding_models"][model_name] = {
                        "available": False,
                        "status": "sentence-transformers not installed"
                    }

        except Exception as e:
            availability["embedding_models"][model_name] = {"available": False, "status": f"Error: {str(e)}"}

    # --- Check Image Models ---
    image_models = [
        ("NoImageModel", "Always available"),
        ("CLIPModelHandler", "Requires transformers and PIL")
    ]

    for model_name, requirement in image_models:
        try:
            if model_name == "NoImageModel":
                availability["image_models"][model_name] = {"available": True, "status": "Always available"}

            elif model_name == "CLIPModelHandler":
                try:
                    from transformers import CLIPModel, CLIPProcessor
                    from PIL import Image
                    availability["image_models"][model_name] = {"available": True, "status": "Ready"}
                except ImportError:
                    availability["image_models"][model_name] = {"available": False, "status": "Missing dependencies"}

        except Exception as e:
            availability["image_models"][model_name] = {"available": False, "status": f"Error: {str(e)}"}

    return availability


def get_recommended_model_setup():
    """Get recommendations for model setup based on system capabilities, with smart AI-embedding pairing."""
    availability = check_model_availability()
    recommendations = {
        "ai_model": None,
        "embedding_model": None,
        "image_model": None,
        "reasoning": {}
    }

    # AI model recommendations with smart embedding pairing
    if availability["ai_models"].get("TinyLlamaModel", {}).get("available", False):
        recommendations["ai_model"] = "TinyLlamaModel"
        recommendations["reasoning"]["ai"] = "TinyLlama is available and provides local, private AI without API costs"

        # Pair TinyLlama AI with TinyLlama embedding for optimal workflow
        if availability["embedding_models"].get("TinyLlamaEmbeddingModel", {}).get("available", False):
            recommendations["embedding_model"] = "TinyLlamaEmbeddingModel"
            recommendations["reasoning"][
                "embedding"] = "TinyLlama embeddings are optimized for the TinyLlama workflow with lightweight models"
        elif availability["embedding_models"].get("GPT4AllEmbeddingModel", {}).get("available", False):
            recommendations["embedding_model"] = "GPT4AllEmbeddingModel"
            recommendations["reasoning"]["embedding"] = "Local embeddings complement TinyLlama for complete privacy"

    elif availability["ai_models"].get("GPT4AllModel", {}).get("available", False):
        recommendations["ai_model"] = "GPT4AllModel"
        recommendations["reasoning"]["ai"] = "GPT4All is available for local AI processing"

        # Pair GPT4All AI with GPT4All embedding for local workflow
        if availability["embedding_models"].get("GPT4AllEmbeddingModel", {}).get("available", False):
            recommendations["embedding_model"] = "GPT4AllEmbeddingModel"
            recommendations["reasoning"]["embedding"] = "Local embeddings complement GPT4All for complete privacy"
        elif availability["embedding_models"].get("TinyLlamaEmbeddingModel", {}).get("available", False):
            recommendations["embedding_model"] = "TinyLlamaEmbeddingModel"
            recommendations["reasoning"]["embedding"] = "TinyLlama embeddings provide fast local processing"

    elif availability["ai_models"].get("OpenAIModel", {}).get("available", False):
        recommendations["ai_model"] = "OpenAIModel"
        recommendations["reasoning"]["ai"] = "OpenAI API is configured and provides high-quality responses"

        # Pair OpenAI AI with OpenAI embedding for consistency
        if availability["embedding_models"].get("OpenAIEmbeddingModel", {}).get("available", False):
            recommendations["embedding_model"] = "OpenAIEmbeddingModel"
            recommendations["reasoning"]["embedding"] = "OpenAI embeddings provide high-quality vector representations"

    elif availability["ai_models"].get("AnthropicModel", {}).get("available", False):
        recommendations["ai_model"] = "AnthropicModel"
        recommendations["reasoning"]["ai"] = "Anthropic API is configured and provides excellent AI capabilities"

        # For Anthropic, suggest best available embedding
        if availability["embedding_models"].get("OpenAIEmbeddingModel", {}).get("available", False):
            recommendations["embedding_model"] = "OpenAIEmbeddingModel"
            recommendations["reasoning"]["embedding"] = "OpenAI embeddings provide quality vectors for Anthropic AI"
        elif availability["embedding_models"].get("TinyLlamaEmbeddingModel", {}).get("available", False):
            recommendations["embedding_model"] = "TinyLlamaEmbeddingModel"
            recommendations["reasoning"]["embedding"] = "Local embeddings provide privacy with Anthropic AI"

    else:
        recommendations["ai_model"] = "NoAIModel"
        recommendations["reasoning"]["ai"] = "No AI models are properly configured"

    # Embedding model fallback (if not set above by AI model pairing)
    if not recommendations["embedding_model"]:
        if availability["embedding_models"].get("TinyLlamaEmbeddingModel", {}).get("available", False):
            recommendations["embedding_model"] = "TinyLlamaEmbeddingModel"
            recommendations["reasoning"]["embedding"] = "TinyLlama embeddings are available for fast local processing"
        elif availability["embedding_models"].get("GPT4AllEmbeddingModel", {}).get("available", False):
            recommendations["embedding_model"] = "GPT4AllEmbeddingModel"
            recommendations["reasoning"]["embedding"] = "Local embeddings are available for privacy and no API costs"
        elif availability["embedding_models"].get("OpenAIEmbeddingModel", {}).get("available", False):
            recommendations["embedding_model"] = "OpenAIEmbeddingModel"
            recommendations["reasoning"]["embedding"] = "OpenAI embeddings provide high-quality vector representations"
        else:
            recommendations["embedding_model"] = "NoEmbeddingModel"
            recommendations["reasoning"]["embedding"] = "No embedding models are properly configured"

    # Image model recommendations (unchanged)
    if availability["image_models"].get("CLIPModelHandler", {}).get("available", False):
        recommendations["image_model"] = "CLIPModelHandler"
        recommendations["reasoning"]["image"] = "CLIP provides excellent image processing capabilities"
    else:
        recommendations["image_model"] = "NoImageModel"
        recommendations["reasoning"]["image"] = "No image models are properly configured"

    return recommendations


def apply_recommended_models():
    """Apply the recommended model configuration based on system availability."""
    recommendations = get_recommended_model_setup()

    try:
        success_count = 0

        if recommendations["ai_model"]:
            if ModelsConfig.set_current_ai_model(recommendations["ai_model"]):
                logger.info(f"Set AI model to: {recommendations['ai_model']}")
                success_count += 1

        if recommendations["embedding_model"]:
            if ModelsConfig.set_current_embedding_model(recommendations["embedding_model"]):
                logger.info(f"Set embedding model to: {recommendations['embedding_model']}")
                success_count += 1

        if recommendations["image_model"]:
            if ModelsConfig.set_current_image_model(recommendations["image_model"]):
                logger.info(f"Set image model to: {recommendations['image_model']}")
                success_count += 1

        logger.info(f"Successfully applied {success_count}/3 recommended model configurations")
        return success_count == 3

    except Exception as e:
        logger.error(f"Error applying recommended models: {e}")
        return False


def configure_tinyllama_workflow():
    """Configure both TinyLlama AI and TinyLlama embedding models for optimal local workflow."""
    try:
        success_count = 0

        # Set TinyLlama as AI model
        if ModelsConfig.set_current_ai_model("TinyLlamaModel"):
            logger.info("Set AI model to: TinyLlamaModel")
            success_count += 1

        # Set TinyLlama as embedding model
        if ModelsConfig.set_current_embedding_model("TinyLlamaEmbeddingModel"):
            logger.info("Set embedding model to: TinyLlamaEmbeddingModel")
            success_count += 1

        logger.info(f"TinyLlama workflow configured: {success_count}/2 models set")
        return success_count == 2

    except Exception as e:
        logger.error(f"Error configuring TinyLlama workflow: {e}")
        return False
