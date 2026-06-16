"""Bootstrap helpers for model config initialization and defaults."""


from __future__ import annotations


import json

from datetime import datetime, timedelta
from modules.ai.config import ModelsConfig
from modules.configuration.log_config import logger, with_request_id
from sqlalchemy import (
    Column, String, Integer, DateTime, Enum,
    UniqueConstraint, create_engine,inspect
)

def register_default_models_with_tinyllama_updated():
    """Register the default models including TinyLlama in the database using DatabaseConfig."""
    default_configs = [
        # AI models - including TinyLlama
        {"model_type": "ai", "key": "available_models", "value": json.dumps([
            {"name": "NoAIModel", "display_name": "Disabled", "enabled": True},
            {"name": "OpenAIModel", "display_name": "OpenAI GPT", "enabled": True},
            {"name": "Llama3Model", "display_name": "Meta Llama 3", "enabled": True},
            {"name": "AnthropicModel", "display_name": "Anthropic Claude", "enabled": True},
            {"name": "GPT4AllModel", "display_name": "GPT4All (Local)", "enabled": True},
            {"name": "TinyLlamaModel", "display_name": "TinyLlama 1.1B (Local)", "enabled": True}
        ])},
        {"model_type": "ai", "key": "CURRENT_MODEL", "value": "OpenAIModel"},

        # TinyLlama specific configuration
        {"model_type": "ai", "key": "TINYLLAMA_MODEL_PATH",
         "value": r"C:\Users\10169062\PycharmProjects\MDEV_EMTAC\plugins\ai_modules\TinyLlama_1_1B"},
        {"model_type": "ai", "key": "TINYLLAMA_TIMEOUT", "value": "120"},
        {"model_type": "ai", "key": "TINYLLAMA_MAX_TOKENS", "value": "256"},

        # Embedding models - including TinyLlama
        {"model_type": "embedding", "key": "available_models", "value": json.dumps([
            {"name": "NoEmbeddingModel", "display_name": "Disabled", "enabled": True},
            {"name": "OpenAIEmbeddingModel", "display_name": "OpenAI Embedding", "enabled": True},
            {"name": "GPT4AllEmbeddingModel", "display_name": "Local Embeddings (SentenceTransformers)",
             "enabled": True},
            {"name": "TinyLlamaEmbeddingModel", "display_name": "TinyLlama Embeddings (Optimized)", "enabled": True}
        ])},
        {"model_type": "embedding", "key": "CURRENT_MODEL", "value": "OpenAIEmbeddingModel"},

        # Image models (unchanged)
        {"model_type": "image", "key": "available_models", "value": json.dumps([
            {"name": "NoImageModel", "display_name": "Disabled", "enabled": True},
            {"name": "CLIPModelHandler", "display_name": "CLIP Model Handler", "enabled": True}
        ])},
        {"model_type": "image", "key": "CURRENT_MODEL", "value": "CLIPModelHandler"}
    ]

    try:
        from modules.configuration.config_env import DatabaseConfig
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
    except ImportError:
        logger.warning("DatabaseConfig not available, using fallback session")
        session = Session()

    try:
        for config in default_configs:
            existing = session.query(ModelsConfig).filter_by(
                model_type=config["model_type"],
                key=config["key"]
            ).first()

            if not existing:
                config_entry = ModelsConfig(**config)
                session.add(config_entry)
                logger.info(f"Registered config: {config['model_type']}.{config['key']}")
            else:
                # Update existing available_models to include TinyLlama if missing
                if config["key"] == "available_models":
                    try:
                        existing_models = json.loads(existing.value)
                        new_models = json.loads(config["value"])

                        # Get existing model names
                        existing_names = {model["name"] for model in existing_models}

                        # Add any missing models (like TinyLlama)
                        for new_model in new_models:
                            if new_model["name"] not in existing_names:
                                existing_models.append(new_model)
                                logger.info(f"Added missing model: {new_model['name']}")

                        # Update the database
                        existing.value = json.dumps(existing_models)
                        existing.updated_at = datetime.utcnow()

                    except json.JSONDecodeError:
                        logger.error(f"Error parsing existing models, replacing with defaults")
                        existing.value = config["value"]
                        existing.updated_at = datetime.utcnow()

        session.commit()
        logger.info("Default configurations with TinyLlama registered successfully")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error registering default configurations: {e}")
        return False
    finally:
        session.close()


def register_default_models_with_tinyllama():
    """Register the default models including TinyLlama in the database using DatabaseConfig."""
    default_configs = [
        # AI models - including TinyLlama
        {"model_type": "ai", "key": "available_models", "value": json.dumps([
            {"name": "NoAIModel", "display_name": "Disabled", "enabled": True},
            {"name": "OpenAIModel", "display_name": "OpenAI GPT", "enabled": True},
            {"name": "Llama3Model", "display_name": "Meta Llama 3", "enabled": True},
            {"name": "AnthropicModel", "display_name": "Anthropic Claude", "enabled": True},
            {"name": "GPT4AllModel", "display_name": "GPT4All (Local)", "enabled": True},
            {"name": "TinyLlamaModel", "display_name": "TinyLlama 1.1B (Local)", "enabled": True}  # NEW
        ])},
        {"model_type": "ai", "key": "CURRENT_MODEL", "value": "OpenAIModel"},

        # Embedding models - ADD TinyLlama embedding here
        {"model_type": "embedding", "key": "available_models", "value": json.dumps([
            {"name": "NoEmbeddingModel", "display_name": "Disabled", "enabled": True},
            {"name": "OpenAIEmbeddingModel", "display_name": "OpenAI Embedding", "enabled": True},
            {"name": "GPT4AllEmbeddingModel", "display_name": "Local Embeddings (SentenceTransformers)",
             "enabled": True},
            {"name": "TinyLlamaEmbeddingModel", "display_name": "TinyLlama Embeddings (Optimized)", "enabled": True}
        ])},
        {"model_type": "embedding", "key": "CURRENT_MODEL", "value": "OpenAIEmbeddingModel"},

        # Image models (unchanged)
        {"model_type": "image", "key": "available_models", "value": json.dumps([
            {"name": "NoImageModel", "display_name": "Disabled", "enabled": True},
            {"name": "CLIPModelHandler", "display_name": "CLIP Model Handler", "enabled": True}
        ])},
        {"model_type": "image", "key": "CURRENT_MODEL", "value": "CLIPModelHandler"}
    ]

    try:
        from modules.configuration.config_env import DatabaseConfig
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
    except ImportError:
        logger.warning("DatabaseConfig not available, using fallback session")
        session = Session()

    try:
        for config in default_configs:
            existing = session.query(ModelsConfig).filter_by(
                model_type=config["model_type"],
                key=config["key"]
            ).first()

            if not existing:
                config_entry = ModelsConfig(**config)
                session.add(config_entry)
                logger.info(f"Registered config: {config['model_type']}.{config['key']}")
            else:
                # Update existing available_models to include TinyLlama if missing
                if config["key"] == "available_models":
                    try:
                        existing_models = json.loads(existing.value)
                        new_models = json.loads(config["value"])

                        # Get existing model names
                        existing_names = {model["name"] for model in existing_models}

                        # Add any missing models (like TinyLlama)
                        for new_model in new_models:
                            if new_model["name"] not in existing_names:
                                existing_models.append(new_model)
                                logger.info(f"Added missing model: {new_model['name']}")

                        # Update the database
                        existing.value = json.dumps(existing_models)
                        existing.updated_at = datetime.utcnow()

                    except json.JSONDecodeError:
                        logger.error(f"Error parsing existing models, replacing with defaults")
                        existing.value = config["value"]
                        existing.updated_at = datetime.utcnow()

        session.commit()
        logger.info("Default configurations with TinyLlama registered successfully")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error registering default configurations: {e}")
        return False
    finally:
        session.close()


def initialize_models_config():
    """
    Create the models configuration table if it doesn't exist and register default models.
    This function uses DatabaseConfig for proper session management.
    """

    try:
        logger.info("Initializing models configuration table...")

        # Create an inspector to check if table exists
        inspector = inspect(engine)

        # Check if the table already exists
        if not inspector.has_table(ModelsConfig.__tablename__):
            try:
                # Create the table
                ModelsConfig.__table__.create(engine)
                logger.info(f"Successfully created table {ModelsConfig.__tablename__}")
            except Exception as e:
                logger.error(f"Error creating ModelsConfig table: {str(e)}")
                return False

        # Initialize with default configurations including TinyLlama
        success = register_default_models()
        if success:
            logger.info("Default model configurations registered successfully")
        else:
            logger.warning("Some issues occurred while registering default model configurations")

        return True

    except Exception as e:
        logger.error(f"Unexpected error initializing ModelsConfig: {str(e)}")
        logger.exception("Exception details:")
        return False


def register_default_models():
    """Register the default models in the database using DatabaseConfig, including TinyLlama."""
    default_configs = [
        # AI models - including GPT4All and TinyLlama as local options
        {"model_type": "ai", "key": "available_models", "value": json.dumps([
            {"name": "NoAIModel", "display_name": "Disabled", "enabled": True},
            {"name": "OpenAIModel", "display_name": "OpenAI GPT", "enabled": True},
            {"name": "Llama3Model", "display_name": "Meta Llama 3", "enabled": True},
            {"name": "AnthropicModel", "display_name": "Anthropic Claude", "enabled": True},
            {"name": "GPT4AllModel", "display_name": "GPT4All (Local)", "enabled": True},
            {"name": "TinyLlamaModel", "display_name": "TinyLlama 1.1B (Local)", "enabled": True}
        ])},
        {"model_type": "ai", "key": "CURRENT_MODEL", "value": "OpenAIModel"},

        # Embedding models - including GPT4All and TinyLlama embedding options
        {"model_type": "embedding", "key": "available_models", "value": json.dumps([
            {"name": "NoEmbeddingModel", "display_name": "Disabled", "enabled": True},
            {"name": "OpenAIEmbeddingModel", "display_name": "OpenAI Embedding", "enabled": True},
            {"name": "GPT4AllEmbeddingModel", "display_name": "Local Embeddings (SentenceTransformers)",
             "enabled": True},
            {"name": "TinyLlamaEmbeddingModel", "display_name": "TinyLlama Embeddings (Optimized)", "enabled": True}
            # ADD THIS LINE
        ])},

        # Image models (unchanged)
        {"model_type": "image", "key": "available_models", "value": json.dumps([
            {"name": "NoImageModel", "display_name": "Disabled", "enabled": True},
            {"name": "CLIPModelHandler", "display_name": "CLIP Model Handler", "enabled": True}
        ])},
        {"model_type": "image", "key": "CURRENT_MODEL", "value": "CLIPModelHandler"},

        # TinyLlama specific configuration
        {"model_type": "ai", "key": "TINYLLAMA_MODEL_PATH",
         "value": r"C:\Users\10169062\PycharmProjects\MDEV_EMTAC\plugins\ai_modules\TinyLlama_1_1B"},
        {"model_type": "ai", "key": "TINYLLAMA_TIMEOUT", "value": "120"},
        {"model_type": "ai", "key": "TINYLLAMA_MAX_TOKENS", "value": "256"}
    ]

    try:
        from modules.configuration.config_env import DatabaseConfig
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
    except ImportError:
        logger.warning("DatabaseConfig not available, using fallback session")
        session = Session()

    try:
        for config in default_configs:
            existing = session.query(ModelsConfig).filter_by(
                model_type=config["model_type"],
                key=config["key"]
            ).first()

            if not existing:
                config_entry = ModelsConfig(**config)
                session.add(config_entry)
                logger.info(f"Registered config: {config['model_type']}.{config['key']}")
            else:
                # Update existing available_models to include new models if missing
                if config["key"] == "available_models":
                    try:
                        existing_models = json.loads(existing.value)
                        new_models = json.loads(config["value"])

                        # Get existing model names
                        existing_names = {model["name"] for model in existing_models}

                        # Add any missing models (like TinyLlama)
                        models_added = False
                        for new_model in new_models:
                            if new_model["name"] not in existing_names:
                                existing_models.append(new_model)
                                logger.info(f"Added missing model: {new_model['name']}")
                                models_added = True

                        # Update the database if models were added
                        if models_added:
                            existing.value = json.dumps(existing_models)
                            existing.updated_at = datetime.utcnow()

                    except json.JSONDecodeError:
                        logger.error(
                            f"Error parsing existing models for {config['model_type']}, replacing with defaults")
                        existing.value = config["value"]
                        existing.updated_at = datetime.utcnow()

                # Update other configuration values if they don't exist or are different
                elif config["key"] in ["TINYLLAMA_MODEL_PATH", "TINYLLAMA_TIMEOUT", "TINYLLAMA_MAX_TOKENS"]:
                    if existing.value != config["value"]:
                        logger.info(f"Updating config: {config['model_type']}.{config['key']} = {config['value']}")
                        existing.value = config["value"]
                        existing.updated_at = datetime.utcnow()

        session.commit()
        logger.info("Default configurations with TinyLlama registered successfully")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error registering default configurations: {e}")
        return False
    finally:
        session.close()
