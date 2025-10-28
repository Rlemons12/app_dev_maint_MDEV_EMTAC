
"""
AI Models Module - Unified Import Block
Aligned with E:\emtac\models architecture
"""

# -------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------
import os
import sys
import time
import json
import base64
import logging
import threading
import importlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import defaultdict

# -------------------------------------------------------
# Third-Party Libraries
# -------------------------------------------------------
import torch
from torch import compile as torch_compile  # alias for clarity
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import openai
import requests
import numpy
from sentence_transformers import SentenceTransformer
from sqlalchemy import (
    Column, String, Integer, DateTime, Enum,
    UniqueConstraint, create_engine, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

# -------------------------------------------------------
# Project Imports
# -------------------------------------------------------
from modules.configuration.config import (
    OPENAI_API_KEY, OPENAI_MODEL_NAME, DATABASE_URL, ANTHROPIC_API_KEY,
    MODELS_DIR, MODELS_LLM_DIR, MODELS_IMAGE_DIR, MODELS_CLIP_DIR,
    MODELS_TINY_LLAMA_DIR, MODELS_QWEN_DIR,
    HF_HOME, HF_HUB_CACHE, SENTENCE_TRANSFORMERS_MODELS_PATH
)
from modules.configuration.log_config import logger, with_request_id
from modules.configuration.base import Base


# Feature flags - explicit since dependencies are installed and verified
QUANTIZATION_AVAILABLE = True       # BitsAndBytes quantization supported
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")
TRANSFORMERS_AVAILABLE = True


# Safe imports with fallbacks
try:
    from modules.configuration.log_config import (
        with_request_id, debug_id, info_id, warning_id, error_id,
        get_request_id, log_timed_operation
    )
    LOGGING_AVAILABLE = True
except ImportError:
    # Fallback if logging imports fail
    LOGGING_AVAILABLE = False
    def with_request_id(func): return func
    def debug_id(msg, req_id=None): pass
    def info_id(msg, req_id=None): pass
    def warning_id(msg, req_id=None): pass
    def error_id(msg, req_id=None): pass
    def get_request_id(): return "unknown"
    def log_timed_operation(name, req_id=None):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContext()


logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))


class ModelsConfig(Base):
    __tablename__ = 'models_config'

    id = Column(Integer, primary_key=True)
    model_type = Column(Enum('ai', 'image', 'embedding', name='model_type_enum'), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(String(1000), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        # Composite unique constraint on model_type and key
        UniqueConstraint('model_type', 'key', name='unique_model_type_key'),
    )

    def __repr__(self):
        return f"<ModelsConfig(model_type='{self.model_type}', key='{self.key}')>"

    @staticmethod
    def load_config_from_db():
        """
        Load AI model configuration from the database using DatabaseConfig.

        Returns:
            Tuple of (current_ai_model, current_embedding_model)
        """
        try:
            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
        except ImportError:
            logger.warning("DatabaseConfig not available, using fallback session")
            session = Session()

        try:
            ai_model_config = session.query(ModelsConfig).filter_by(
                model_type='ai',
                key="CURRENT_MODEL"
            ).first()

            embedding_model_config = session.query(ModelsConfig).filter_by(
                model_type='embedding',
                key="CURRENT_MODEL"
            ).first()

            current_ai_model = ai_model_config.value if ai_model_config else "NoAIModel"
            current_embedding_model = embedding_model_config.value if embedding_model_config else "NoEmbeddingModel"

            return current_ai_model, current_embedding_model
        finally:
            session.close()

    @staticmethod
    def load_image_model_config_from_db():
        """
        Load image model configuration from the database using DatabaseConfig.

        Returns:
            String representing the current image model
        """
        try:
            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
        except ImportError:
            logger.warning("DatabaseConfig not available, using fallback session")
            session = Session()

        try:
            image_model_config = session.query(ModelsConfig).filter_by(
                model_type='image',
                key="CURRENT_MODEL"
            ).first()

            current_image_model = image_model_config.value if image_model_config else "no_model"

            return current_image_model
        finally:
            session.close()

    @classmethod
    def set_config_value(cls, model_type, key, value):
        """
        Set a configuration value in the database using DatabaseConfig.

        Args:
            model_type: Type of model ('ai', 'image', 'embedding')
            key: Configuration key
            value: Configuration value

        Returns:
            Boolean indicating success
        """
        try:
            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
        except ImportError:
            logger.warning("DatabaseConfig not available, using fallback session")
            session = Session()

        try:
            # Check if config already exists
            config = session.query(cls).filter_by(
                model_type=model_type,
                key=key
            ).first()

            if config:
                # Update existing config
                config.value = value
                config.updated_at = datetime.utcnow()
            else:
                # Create new config
                config = cls(
                    model_type=model_type,
                    key=key,
                    value=value
                )
                session.add(config)

            session.commit()
            logger.info(f"Successfully set config {model_type}.{key} = {value}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error setting config {model_type}.{key}: {e}")
            return False
        finally:
            session.close()

    @classmethod
    def get_config_value(cls, model_type, key, default=None):
        """
        Get a configuration value from the database using DatabaseConfig.

        Args:
            model_type: Type of model ('ai', 'image', 'embedding')
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value or default if not found
        """
        try:
            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
        except ImportError:
            logger.warning("DatabaseConfig not available, using fallback session")
            session = Session()

        try:
            config = session.query(cls).filter_by(
                model_type=model_type,
                key=key
            ).first()

            if config:
                return config.value
            return default
        except Exception as e:
            logger.error(f"Error getting config {model_type}.{key}: {e}")
            return default
        finally:
            session.close()

    @classmethod
    def set_current_ai_model(cls, model_name):
        """Set the current AI model to use."""
        return cls.set_config_value('ai', 'CURRENT_MODEL', model_name)

    @classmethod
    def set_current_embedding_model(cls, model_name):
        """Set the current embedding model to use."""
        return cls.set_config_value('embedding', 'CURRENT_MODEL', model_name)

    @classmethod
    def set_current_image_model(cls, model_name):
        """Set the current image model to use."""
        return cls.set_config_value('image', 'CURRENT_MODEL', model_name)

    @classmethod
    def initialize_models_config_table(cls):
        """Initialize the model configurations with default values if they don't exist."""
        # Set default AI model if not set
        if not cls.get_config_value('ai', 'CURRENT_MODEL'):
            cls.set_current_ai_model('OpenAIModel')

        # Set default embedding model if not set
        if not cls.get_config_value('embedding', 'CURRENT_MODEL'):
            cls.set_current_embedding_model('OpenAIEmbeddingModel')

        # Set default image model if not set
        if not cls.get_config_value('image', 'CURRENT_MODEL'):
            cls.set_current_image_model('CLIPModelHandler')

        logger.info("Model configurations initialized")

    @classmethod
    def get_available_models(cls, model_type):
        """Get list of available models for a specific type with their details."""
        models_json = cls.get_config_value(model_type, "available_models", "[]")
        try:
            return json.loads(models_json)
        except json.JSONDecodeError:
            logger.error(f"Error parsing available models for {model_type}")
            return []

    @classmethod
    def get_enabled_models(cls, model_type):
        """Get list of enabled models for a specific type."""
        models = cls.get_available_models(model_type)
        return [model for model in models if model.get("enabled", True)]

    @classmethod
    def get_current_model_info(cls, model_type):
        """Get detailed information about the current model of a specific type."""
        current_model = cls.get_config_value(model_type, "CURRENT_MODEL")
        if not current_model:
            return None

        models = cls.get_available_models(model_type)
        for model in models:
            if model["name"] == current_model:
                return model

        return None

    @classmethod
    def load_ai_model(cls, model_name=None):
        """Load an AI model from E:\\emtac\\models by name."""
        if model_name is None:
            model_name = cls.get_config_value('ai', 'CURRENT_MODEL', 'NoAIModel')
        try:
            module = importlib.import_module('models')
            model_class = getattr(module, model_name)
            logger.info(f"Loaded AI model: {model_name}")
            return model_class()
        except Exception as e:
            logger.error(f"Error loading AI model {model_name}: {e}")
            return cls._fallback_ai_model()

    @classmethod
    def load_embedding_model(cls, model_name=None):
        """Load an embedding model from E:\\emtac\\models by name."""
        if model_name is None:
            model_name = cls.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')
        try:
            module = importlib.import_module('models')
            model_class = getattr(module, model_name)
            logger.info(f"Loaded embedding model: {model_name}")
            return model_class()
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")
            return cls._fallback_embedding_model()

    @classmethod
    def load_image_model(cls, model_name=None):
        """Load an image model from E:\\emtac\\models\\image_models."""
        if model_name is None:
            model_name = cls.get_config_value('image', 'CURRENT_MODEL', 'NoImageModel')
        try:
            module = importlib.import_module('models.image_models')
            model_class = getattr(module, model_name)
            logger.info(f"Loaded image model: {model_name}")
            return model_class()
        except Exception as e:
            logger.error(f"Error loading image model {model_name}: {e}")
            return cls._fallback_image_model()

    # -------------------------------------------------------
    # Fallback Implementations
    # -------------------------------------------------------
    @staticmethod
    def _fallback_ai_model():
        class FallbackAI:
            def get_response(self, prompt): return "AI service unavailable."

            def generate_description(self, image_path): return "Unavailable."

        logger.warning("Using fallback AI model")
        return FallbackAI()

    @staticmethod
    def _fallback_embedding_model():
        class FallbackEmbedding:
            def get_embeddings(self, text): return []

        logger.warning("Using fallback embedding model")
        return FallbackEmbedding()

    @staticmethod
    def _fallback_image_model():
        class FallbackImage:
            def process_image(self, image_path): return "Image processing unavailable."

            def compare_images(self, a, b): return {"similarity": 0.0}

            def generate_description(self, image_path): return "Unavailable."

        logger.warning("Using fallback image model")
        return FallbackImage()


# Define the AIModel interface
class AIModel(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_description(self, image_path: str) -> str:
        pass


# Define the EmbeddingModel interface
class EmbeddingModel(ABC):
    @abstractmethod
    def get_embeddings(self, text: str) -> list:
        pass


# Define the ImageModel interface
class ImageModel(ABC):
    @abstractmethod
    def process_image(self, image_path: str) -> str:
        pass

    @abstractmethod
    def compare_images(self, image1_path: str, image2_path: str) -> dict:
        pass

    @abstractmethod
    def generate_description(self, image_path: str) -> str:
        pass


# Implementations of the AI model classes
class NoAIModel(AIModel):
    def get_response(self, prompt: str) -> str:
        return "AI is currently disabled."

    def generate_description(self, image_path: str) -> str:
        return "AI description generation is currently disabled."


class AnthropicModel(AIModel):
    def __init__(self):
        self.api_key = ANTHROPIC_API_KEY
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-5-sonnet-20241022"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        logger.debug(f"Anthropic API Key: {self.api_key[:5] if self.api_key else 'None'}...")

    def get_response(self, prompt: str) -> str:
        logger.debug(f"Using Anthropic model: {self.model}")
        logger.debug(f"Sending prompt to Anthropic: {prompt}")

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )

            if response.status_code == 200:
                response_data = response.json()
                answer = response_data['content'][0]['text']
                logger.debug(f"Anthropic response: {answer}")
                return answer
            else:
                logger.error(f"Error from Anthropic API: {response.status_code} - {response.text}")
                return f"An error occurred: {response.status_code}"

        except Exception as e:
            logger.error(f"Error while getting response from Anthropic: {e}")
            return "An error occurred while processing your request."

    def generate_description(self, image_path: str) -> str:
        logger.debug(f"Generating image description with Anthropic")

        try:
            # Convert the image to base64
            base64_image = self.encode_image(image_path)

            payload = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": "What's in this image?"
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )

            if response.status_code == 200:
                response_data = response.json()
                description = response_data['content'][0]['text']
                return description
            else:
                logger.error(f"Error from Anthropic API: {response.status_code} - {response.text}")
                return f"An error occurred: {response.status_code}"

        except Exception as e:
            logger.error(f"Error while generating description with Anthropic: {e}")
            return "An error occurred while processing the image."

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return base64_encoded


class OpenAIModel(AIModel):
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        logger.debug(f"OpenAI API Key: {OPENAI_API_KEY[:5] if OPENAI_API_KEY else 'None'}...")

    def get_response(self, prompt: str) -> str:
        logger.debug(f"Using OpenAI model: {OPENAI_MODEL_NAME}")
        logger.debug(f"Sending prompt to OpenAI: {prompt}")
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=1000
            )
            answer = response.choices[0].text.strip()
            logger.debug(f"OpenAI response: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error while getting response from OpenAI: {e}")
            return "An error occurred while processing your request."

    def generate_description(self, image_path: str) -> str:
        base64_image = self.encode_image(image_path)
        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                return "No description available."
        else:
            return "Error in API request."

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return base64_encoded


class Llama3Model(AIModel):
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logger.debug(f"Loaded Hugging Face model: {self.model_id}")
        except Exception as e:
            logger.error(f"Error loading Llama3 model: {e}")
            self.model = None
            self.tokenizer = None

    def get_response(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            return "Llama3 model is not available."

        logger.debug(f"Using Hugging Face model: {self.model_id}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("")
            ]

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=56,
                #eos_token_id=terminators,
                do_sample=False,
                #temperature=0.6,
                #top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache
                early_stopping=True  # Stop at EOS token
            )
            response = outputs[0][input_ids.shape[-1]:]
            decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)
            logger.debug(f"Hugging Face response: {decoded_response}")
            return decoded_response
        except Exception as e:
            logger.error(f"Error generating response with Llama3: {e}")
            return "An error occurred while processing your request."

    def generate_description(self, image_path: str) -> str:
        return "Image description not supported for Llama model."


# Implementation of the embedding model classes
class NoEmbeddingModel(EmbeddingModel):
    def get_embeddings(self, text: str) -> list:
        return []


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        logger.debug(f"OpenAI API Key for embedding: {OPENAI_API_KEY[:5] if OPENAI_API_KEY else 'None'}...")

    def get_embeddings(self, text: str) -> list:
        logger.debug(f"Generating embeddings using OpenAI model: {OPENAI_MODEL_NAME}")
        try:
            response = openai.Embedding.create(
                input=text,
                model=OPENAI_MODEL_NAME
            )
            embeddings = response['data'][0]['embedding']
            logger.debug(f"Generated embeddings: {len(embeddings)} dimensions")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            return []


# Implementation of the image model classes
class NoImageModel(ImageModel):
    def __init__(self):
        self.model_name = "NoImageModel"

    def process_image(self, image_path: str) -> str:
        return "Image processing is currently disabled."

    def compare_images(self, image1_path: str, image2_path: str) -> dict:
        return {
            "similarity": 0.0,
            "message": "Image comparison is currently disabled.",
            "model": self.model_name
        }

    def generate_description(self, image_path: str) -> str:
        return "Image description is currently disabled."

class CLIPModelHandler:  # Add (ImageModel) if you have a base class
    """Optimized CLIP model handler with offline mode and intelligent caching"""

    # Class-level cache to avoid reloading models across instances
    _model_cache = {}
    _processor_cache = {}
    _cache_initialized = False

    def __init__(self):
        self.model_name = "CLIPModelHandler"
        self.clip_model_name = "openai/clip-vit-base-patch32"

        # Configure offline mode FIRST to prevent network checks
        if not self._cache_initialized:
            self._configure_offline_mode()
            CLIPModelHandler._cache_initialized = True

        logger.info("Initializing optimized CLIP model handler")

        # Load model and processor with intelligent caching
        self.model, self.processor = self._load_or_get_cached_model()

        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.model, 'to'):
            self.model.to(self.device)

        logger.info(f"CLIP model ready on {self.device}")

    def _configure_offline_mode(self):
        """Configure environment to disable online checks - MAJOR SPEEDUP!"""
        # Set environment variables to force offline mode
        offline_env_vars = {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false"  # Disable warnings
        }

        for key, value in offline_env_vars.items():
            os.environ[key] = value

        logger.info("Configured offline mode - network checks disabled")

    def _load_or_get_cached_model(self):
        """Load model with intelligent caching - avoids repeated loading"""
        cache_key = self.clip_model_name

        # Return cached model if available
        if cache_key in self._model_cache:
            logger.info("detected_intent_id = intent_classification['intent_id']Using cached CLIP model (instant load)")
            return self._model_cache[cache_key], self._processor_cache[cache_key]

        logger.info("Loading CLIP model for first time...")
        start_time = time.time()

        try:
            # ATTEMPT 1: Try offline loading first (fastest - no network)
            processor = CLIPProcessor.from_pretrained(
                self.clip_model_name,
                local_files_only=True,
                cache_dir="./model_cache"
            )
            model = CLIPModel.from_pretrained(
                self.clip_model_name,
                local_files_only=True,
                cache_dir="./model_cache"
            )
            logger.info("Loaded CLIP model from local cache (offline)")

        except Exception as offline_error:
            logger.warning(f"Offline loading failed: {offline_error}")
            logger.info("📥 Downloading model from HuggingFace (first time only)...")

            # ATTEMPT 2: Download if not cached (temporarily allow network)
            # Temporarily disable offline mode for download
            temp_offline_vars = {}
            for key in ["TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"]:
                if key in os.environ:
                    temp_offline_vars[key] = os.environ.pop(key)

            try:
                processor = CLIPProcessor.from_pretrained(
                    self.clip_model_name,
                    cache_dir="./model_cache"
                )
                model = CLIPModel.from_pretrained(
                    self.clip_model_name,
                    cache_dir="./model_cache"
                )
                logger.info("📥 Successfully downloaded and cached CLIP model")

            except Exception as download_error:
                logger.error(f"Failed to download model: {download_error}")
                raise

            finally:
                # Restore offline mode
                for key, value in temp_offline_vars.items():
                    os.environ[key] = value

        # Cache the loaded models in memory
        self._model_cache[cache_key] = model
        self._processor_cache[cache_key] = processor

        load_time = time.time() - start_time
        logger.info(f"CLIP model loaded and cached in {load_time:.2f}s")

        return model, processor

    def is_valid_image(self, image):
        """Check if image meets requirements for CLIP processing"""
        try:
            if not isinstance(image, PILImage.Image):
                return False

            # Check minimum dimensions (CLIP is quite flexible)
            width, height = image.size
            min_size = 32  # CLIP can handle small images
            max_size = 2048  # Reasonable upper limit

            if width < min_size or height < min_size:
                logger.debug(f"Image too small: {width}x{height}")
                return False

            if width > max_size or height > max_size:
                logger.debug(f"Image very large: {width}x{height} (will resize)")
                # CLIP preprocessor will handle resizing

            return True

        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False

    def get_image_embedding(self, image):
        """Generate CLIP embedding for an image - CORE FUNCTIONALITY"""
        try:
            if not self.is_valid_image(image):
                logger.warning("Invalid image for embedding generation")
                return None

            # Preprocess image using CLIP processor
            inputs = self.processor(images=image, return_tensors="pt", padding=True)

            # Move inputs to correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding with no gradient computation (faster)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Normalize the embedding (important for similarity comparisons)
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy array for storage/compatibility
            embedding_np = embedding.cpu().numpy().flatten()

            logger.debug(f"Generated embedding with shape: {embedding_np.shape}")
            return embedding_np

        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return None

    def process_image(self, image_path: str) -> str:
        """Process image and return status info"""
        try:
            logger.info(f"Processing image with CLIP: {image_path}")

            # Load and validate image
            image = PILImage.open(image_path).convert("RGB")

            if not self.is_valid_image(image):
                return f"Invalid image: {image_path}"

            # Generate embedding
            embedding = self.get_image_embedding(image)

            if embedding is not None:
                return f"Successfully processed: {image_path} (embedding: {embedding.shape})"
            else:
                return f"Failed to generate embedding: {image_path}"

        except Exception as e:
            logger.error(f"Error processing image with CLIP: {e}")
            return f"Error processing image: {str(e)}"

    def compare_images(self, image1_path: str, image2_path: str) -> dict:
        """Compare two images using CLIP embeddings with cosine similarity"""
        try:
            logger.info(f"Comparing images: {image1_path} vs {image2_path}")

            # Load both images
            image1 = PILImage.open(image1_path).convert("RGB")
            image2 = PILImage.open(image2_path).convert("RGB")

            # Generate embeddings
            embedding1 = self.get_image_embedding(image1)
            embedding2 = self.get_image_embedding(image2)

            if embedding1 is None or embedding2 is None:
                raise ValueError("Failed to generate embeddings for one or both images")

            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]

            # Convert numpy types to Python types for JSON serialization
            similarity = float(similarity)

            # Interpret similarity score
            if similarity > 0.9:
                interpretation = "Very similar"
            elif similarity > 0.7:
                interpretation = "Similar"
            elif similarity > 0.5:
                interpretation = "Somewhat similar"
            else:
                interpretation = "Different"

            return {
                "similarity": similarity,
                "interpretation": interpretation,
                "image1": image1_path,
                "image2": image2_path,
                "model": self.model_name,
                "message": "Comparison completed successfully"
            }

        except Exception as e:
            logger.error(f"Error comparing images with CLIP: {e}")
            return {
                "similarity": 0.0,
                "image1": image1_path,
                "image2": image2_path,
                "model": self.model_name,
                "error": str(e),
                "message": "Comparison failed"
            }

    def generate_description(self, image_path: str) -> str:
        """Generate basic image description"""
        try:
            logger.info(f"Generating description for: {image_path}")

            # Load image
            image = PILImage.open(image_path).convert("RGB")

            if not self.is_valid_image(image):
                return f"Invalid image for description: {image_path}"

            # Get basic image properties
            width, height = image.size
            aspect_ratio = width / height

            # Determine orientation
            if aspect_ratio > 1.3:
                orientation = "landscape"
            elif aspect_ratio < 0.77:
                orientation = "portrait"
            else:
                orientation = "square"

            # Calculate megapixels
            megapixels = (width * height) / 1_000_000

            # Generate description
            description = f"A {orientation} image with {width}×{height} pixels ({megapixels:.1f}MP)"

            # Add file info
            import os
            file_size = os.path.getsize(image_path) / 1024  # KB
            if file_size > 1024:
                size_str = f"{file_size / 1024:.1f}MB"
            else:
                size_str = f"{file_size:.0f}KB"

            description += f", file size: {size_str}"

            return description

        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return f"Error generating description: {str(e)}"

    @classmethod
    def get_cache_stats(cls):
        """Get information about model cache status"""
        return {
            "models_cached": len(cls._model_cache),
            "cache_initialized": cls._cache_initialized,
            "cached_model_names": list(cls._model_cache.keys())
        }

    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory"""
        cls._model_cache.clear()
        cls._processor_cache.clear()
        cls._cache_initialized = False
        logger.info("🗑Cleared CLIP model cache")

    @classmethod
    def preload_model(cls, model_name="openai/clip-vit-base-patch32"):
        """Preload model at application startup for fastest first access"""
        logger.info("Preloading CLIP model...")
        handler = cls()  # This will load and cache the model
        logger.info("CLIP model preloaded and cached")
        return handler

class GPT4AllModel(AIModel):
    """GPT4All local LLM implementation with request ID tracking and timeout protection"""

    def __init__(self, model_name="Meta-Llama-3-8B-Instruct.Q4_0.gguf"):
        print(f"GPT4AllModel.__init__ called with model_name: {model_name}")

        try:
            # Log at the very start
            if LOGGING_AVAILABLE:
                request_id = get_request_id()
                info_id(f"GPT4AllModel.__init__ starting with model: {model_name}", request_id)
            else:
                print(f"Logging not available, using print: GPT4AllModel.__init__ starting")

        except Exception as e:
            print(f"Error in init logging: {e}")

        self.model_name = model_name
        self.model = None
        self.model_loaded = False
        self.last_error = None
        self.generation_timeout = 350  # in seconds timeout

        print(f"About to call _initialize_model()")
        try:
            self._initialize_model()
            print(f"_initialize_model() completed. model_loaded: {self.model_loaded}")
        except Exception as e:
            print(f"_initialize_model() failed: {e}")
            import traceback
            traceback.print_exc()

    @with_request_id
    def _initialize_model(self):
        """Initialize GPT4All model with detailed debugging"""
        request_id = get_request_id()

        try:
            info_id("=== GPT4All Initialization Debug ===", request_id)

            # Test imports first
            try:
                from gpt4all import GPT4All
                info_id("GPT4All import successful", request_id)
            except ImportError as e:
                error_id(f"GPT4All import failed: {e}", request_id)
                self.model = None
                self.last_error = f"GPT4All import failed: {e}"
                return

            # Check paths
            info_id(f"Model name: {self.model_name}", request_id)
            info_id(f"GPT4ALL_MODELS_PATH: {GPT4ALL_MODELS_PATH}", request_id)

            model_path = os.path.join(GPT4ALL_MODELS_PATH, self.model_name)
            info_id(f"Full model path: {model_path}", request_id)
            info_id(f"Path exists: {os.path.exists(model_path)}", request_id)

            if not os.path.exists(model_path):
                error_id(f"Model file not found: {model_path}", request_id)
                self.last_error = f"Model file not found: {model_path}"
                self.model = None
                return

            # Check file size
            file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
            info_id(f"Model file found, size: {file_size_gb:.2f} GB", request_id)

            # Try model creation
            info_id("Creating GPT4All model instance...", request_id)
            try:
                self.model = GPT4All(
                    self.model_name,
                    model_path=GPT4ALL_MODELS_PATH,
                    allow_download=False,
                    device='cpu',
                    n_threads=2
                )
                info_id("GPT4All model instance created successfully", request_id)
            except Exception as model_error:
                error_id(f"Model creation failed: {model_error}", request_id)
                self.last_error = f"Model creation failed: {model_error}"
                self.model = None
                return

            # Test the model
            info_id("Testing model with simple prompt...", request_id)
            if self._test_model_quick():
                self.model_loaded = True
                info_id("GPT4All model loaded and tested successfully!", request_id)
            else:
                error_id("Model test failed", request_id)
                self.last_error = "Model test failed"
                self.model = None

        except Exception as e:
            error_id(f"Unexpected error in GPT4All initialization: {e}", request_id)
            import traceback
            error_id(f"Traceback: {traceback.format_exc()}", request_id)
            self.last_error = str(e)
            self.model = None

    def _test_model_quick(self):
        """Quick model test with detailed logging"""
        request_id = get_request_id()

        try:
            info_id("Starting quick model test...", request_id)
            result = {"success": False, "response": None, "error": None}

            def test():
                try:
                    info_id("Creating chat session...", request_id)
                    with self.model.chat_session():
                        info_id("Generating test response...", request_id)
                        response = self.model.generate("Hi", max_tokens=5, temp=0.1, streaming=False)
                        result["response"] = response
                        result["success"] = bool(response and len(response.strip()) > 0)
                        info_id(f"Test response: '{response}'", request_id)
                except Exception as e:
                    result["error"] = str(e)
                    error_id(f"Test generation failed: {e}", request_id)

            thread = threading.Thread(target=test)
            thread.daemon = True
            thread.start()
            thread.join(timeout=10)  # Increased timeout for debugging

            if thread.is_alive():
                warning_id("Model test timed out after 10 seconds", request_id)
                return False

            if result["error"]:
                error_id(f"Model test error: {result['error']}", request_id)
                return False

            success = result["success"] and not thread.is_alive()
            info_id(f"Test result: {'SUCCESS' if success else 'FAILED'}", request_id)
            return success

        except Exception as e:
            error_id(f"Error during model test: {e}", request_id)
            return False

    @with_request_id
    def get_response(self, prompt: str) -> str:
        """Generate response with comprehensive timeout and error handling"""
        request_id = get_request_id()

        if self.model is None:
            error_id("GPT4All model not available", request_id)
            return "GPT4All model is not available. Please check installation and model loading."

        debug_id(f"GPT4All request - Model: {self.model_name}, Prompt length: {len(prompt)}", request_id)
        info_id("Starting GPT4All generation - this may take several minutes", request_id)

        try:
            with log_timed_operation("gpt4all_response_generation", request_id):
                result = {"response": None, "error": None, "completed": False}

                def generate():
                    try:
                        info_id("GPT4All processing request", request_id)
                        with self.model.chat_session():
                            result["response"] = self.model.generate(
                                prompt=prompt,
                                max_tokens=100,  # Very short for testing
                                temp=0.5,
                                streaming=False
                            )
                        result["completed"] = True
                        info_id("GPT4All generation completed", request_id)
                    except Exception as e:
                        result["error"] = str(e)
                        error_id(f"Generation error: {e}", request_id)

                # Execute with timeout
                thread = threading.Thread(target=generate, daemon=True)
                thread.start()
                thread.join(timeout=self.generation_timeout)

                # Check results
                if not result["completed"] and thread.is_alive():
                    warning_id(f"GPT4All generation timed out after {self.generation_timeout}s", request_id)
                    return "AI response timed out."

                if result["error"]:
                    error_id(f"GPT4All generation error: {result['error']}", request_id)
                    return f"Error: {result['error']}"

                if result["response"]:
                    info_id(f"GPT4All response generated: {len(result['response'])} chars", request_id)
                    return result["response"]
                else:
                    return "No response generated."

        except Exception as e:
            error_id(f"Unexpected error: {e}", request_id)
            return f"Unexpected error: {str(e)}"

    def generate_description(self, image_path: str) -> str:
        """Image description not supported"""
        return "Image description not supported by GPT4All text models. Use a vision model instead."

    def get_model_info(self):
        """Get model status information"""
        return {
            "status": "loaded" if self.model_loaded else "not_loaded",
            "model_name": self.model_name,
            "backend": "llama.cpp",
            "local": True,
            "privacy": "full - no data sent externally",
            "timeout": f"{self.generation_timeout}s",
            "error": self.last_error
        }

class GPT4AllEmbeddingModel(EmbeddingModel):
    """Simplified embedding model - let sentence-transformers handle caching automatically"""

    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1"):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model using sentence-transformers default behavior"""

        logger.info(f"Loading embedding model: GPT4AllEmbeddingModel")

        try:
            from sentence_transformers import SentenceTransformer

            # Set basic performance settings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # Try models in order of preference
            models_to_try = [
                "nomic-ai/nomic-embed-text-v1",  # Primary model
                "all-MiniLM-L6-v2",  # Small, fast fallback
                "all-mpnet-base-v2",  # High quality fallback
                "paraphrase-MiniLM-L6-v2"  # Another small fallback
            ]

            for model_name in models_to_try:
                if self._try_load_model(model_name):
                    logger.info(f"Successfully loaded embedding model: {self.model_name}")
                    return

            # If all models fail
            logger.warning("SentenceTransformer embedding model not available")
            logger.warning("   All embedding models failed to load")
            logger.warning("   Vector search will not be available")
            self.model = None
            self.is_loaded = False

        except ImportError:
            logger.error("SentenceTransformers not installed. Install with: pip install sentence-transformers")
            self.model = None
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Unexpected error initializing embedding model: {e}")
            self.model = None
            self.is_loaded = False

    def _try_load_model(self, model_name):
        """Try to load a specific embedding model"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Attempting to load: {model_name}")

            # Let sentence-transformers handle everything automatically
            self.model = SentenceTransformer(model_name)

            # Test the model
            logger.info("Testing embedding model...")
            test_embedding = self.model.encode(["test sentence"], show_progress_bar=False)

            if len(test_embedding) > 0 and len(test_embedding[0]) > 0:
                logger.info(f"Model test successful!")
                logger.info(f"   - Model: {model_name}")
                logger.info(f"   - Embedding dimensions: {len(test_embedding[0])}")

                # Get cache location if available
                if hasattr(self.model, 'cache_folder'):
                    logger.info(f"   - Cache location: {self.model.cache_folder}")

                self.model_name = model_name
                self.is_loaded = True
                return True
            else:
                logger.warning(f"Model {model_name} loaded but failed to generate test embedding")
                return False

        except Exception as e:
            logger.debug(f"Failed to load {model_name}: {e}")
            return False

    def get_embeddings(self, text: str) -> list:
        """Generate embeddings for the given text"""
        if self.model is None or not self.is_loaded:
            logger.warning("SentenceTransformer embedding model not available")
            return []

        logger.debug(f"Generating embeddings with SentenceTransformer model: {self.model_name}")

        try:
            # Handle both single text and list inputs
            if isinstance(text, str):
                embeddings = self.model.encode([text], show_progress_bar=False)[0]
            elif isinstance(text, list):
                embeddings = self.model.encode(text, show_progress_bar=False)
                return [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
            else:
                logger.error(f"Invalid text type for embeddings: {type(text)}")
                return []

            logger.debug(f"Generated embeddings: {len(embeddings)} dimensions")
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings with SentenceTransformer: {e}")
            return []

    def is_available(self):
        """Check if the embedding model is available and loaded"""
        return self.model is not None and self.is_loaded

    def get_model_info(self):
        """Get information about the loaded model"""
        if self.is_available():
            try:
                test_embedding = self.model.encode(["test"], show_progress_bar=False)

                # Try to get cache info
                cache_info = "automatic"
                if hasattr(self.model, 'cache_folder'):
                    cache_info = self.model.cache_folder
                elif hasattr(self.model, '_cache_folder'):
                    cache_info = self.model._cache_folder

                return {
                    "model_name": self.model_name,
                    "embedding_dim": len(test_embedding[0]),
                    "cache_location": cache_info,
                    "status": "available"
                }
            except Exception as e:
                return {
                    "model_name": self.model_name,
                    "status": f"error: {e}"
                }
        else:
            return {
                "model_name": self.model_name,
                "status": "not_available"
            }


class TinyLlamaModel(AIModel):  # NOW INHERITS FROM AIModel!
    """
    TinyLlama model implementation integrated with your AI framework.

    Features:
    - Proper inheritance from AIModel interface
    - ModelsConfig integration for configuration
    - Enhanced error handling and logging
    - Memory-efficient caching
    - Framework-compatible methods
    """

    # Class-level caching for memory efficiency across instances
    _model_cache = {}
    _tokenizer_cache = {}
    _cache_initialized = False

    def __init__(self, model_path=None):
        """
        Initialize TinyLlama with framework integration.

        Args:
            model_path: Custom model path (optional, uses ModelsConfig if not provided)
        """
        # Load configuration using your framework's ModelsConfig system
        self.config = self._load_configuration(model_path)

        # Model attributes
        self.model_path = self.config['model_path']
        self.model_name = "TinyLlama-1.1B-Chat-v1.0"
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.last_error = None
        self.generation_timeout = self.config['timeout']
        self.max_tokens = self.config['max_tokens']

        # Initialize logging (using your framework's logging)
        if LOGGING_AVAILABLE:
            self.request_id = get_request_id()
            logger.info(f"Initializing TinyLlama model", extra={'request_id': self.request_id})

        # Configure environment for optimal performance
        if not self._cache_initialized:
            self._configure_environment()
            TinyLlamaModel._cache_initialized = True

        # Initialize the model
        self._initialize_model()

    def _load_configuration(self, model_path=None):
        """Load configuration using your framework's ModelsConfig system."""
        try:
            # Use ModelsConfig to get configuration values from database
            config_model_path = ModelsConfig.get_config_value(
                'ai', 'TINYLLAMA_MODEL_PATH',
                r"C:\Users\10169062\PycharmProjects\MDEV_EMTAC\plugins\ai_modules\TinyLlama_1_1B"
            )
            config_timeout = int(ModelsConfig.get_config_value('ai', 'TINYLLAMA_TIMEOUT', '120'))
            config_max_tokens = int(ModelsConfig.get_config_value('ai', 'TINYLLAMA_MAX_TOKENS', '256'))

            # Disable quantization on CPU automatically
            enable_quantization = torch.cuda.is_available() and QUANTIZATION_AVAILABLE

            return {
                'model_path': model_path or config_model_path,
                'timeout': config_timeout,
                'max_tokens': config_max_tokens,
                'enable_quantization': enable_quantization,  # Auto-detect
                'enable_compile': True
            }
        except Exception as e:
            logger.error(f"Error loading TinyLlama configuration: {e}")
            # Fallback configuration
            return {
                'model_path': model_path or r"C:\Users\10169062\PycharmProjects\MDEV_EMTAC\plugins\ai_modules\TinyLlama_1_1B",
                'timeout': 120,
                'max_tokens': 256,
                'enable_quantization': False,  # Safe default for CPU
                'enable_compile': True
            }

    def _configure_environment(self):
        """Configure environment for optimal offline performance."""
        offline_env_vars = {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false"
        }

        for key, value in offline_env_vars.items():
            os.environ[key] = value

        if LOGGING_AVAILABLE:
            logger.debug("Configured TinyLlama environment for offline operation")

    @with_request_id
    def _initialize_model(self):
        """Initialize TinyLlama model with comprehensive error handling."""
        if not TRANSFORMERS_AVAILABLE:
            self.last_error = f"Transformers not available"
            logger.error(self.last_error)
            return

        # Validate model path
        if not self._validate_model_path():
            return

        # Check cache first
        cache_key = self.model_path
        if cache_key in self._model_cache:
            self.model = self._model_cache[cache_key]
            self.tokenizer = self._tokenizer_cache[cache_key]
            self.model_loaded = True
            logger.info("Using cached TinyLlama model")
            return

        try:
            logger.info(f"Loading TinyLlama model from {self.model_path}")
            start_time = time.time()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            # Configure model loading parameters
            model_kwargs = {
                'local_files_only': True,
                'device_map': {"": "cpu"},
                'torch_dtype': torch.float32,
                'trust_remote_code': True
            }

            # Skip quantization on CPU - it's not well supported
            if QUANTIZATION_AVAILABLE and self.config['enable_quantization'] and torch.cuda.is_available():
                try:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    model_kwargs['quantization_config'] = quant_config
                    logger.info("Enabled 8-bit quantization for TinyLlama")
                except Exception as e:
                    logger.warning(f"Quantization failed, using standard loading: {e}")
            else:
                logger.info("Using standard loading (no quantization on CPU)")

            # Load model
            raw_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            # Apply torch.compile if available and enabled
            if TORCH_COMPILE_AVAILABLE and self.config['enable_compile']:
                try:
                    self.model = torch_compile(raw_model)
                    logger.info("Applied torch.compile optimization to TinyLlama")
                except Exception as e:
                    self.model = raw_model
                    logger.warning(f"torch.compile failed, using uncompiled model: {e}")
            else:
                self.model = raw_model

            # Cache the loaded models
            self._model_cache[cache_key] = self.model
            self._tokenizer_cache[cache_key] = self.tokenizer
            self.model_loaded = True

            load_time = time.time() - start_time
            logger.info(f"TinyLlama model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            self.last_error = f"Model loading failed: {e}"
            logger.error(self.last_error)
            self.model = None
            self.tokenizer = None

    def _validate_model_path(self):
        """Validate that the model path contains required files."""
        if not os.path.exists(self.model_path):
            self.last_error = f"Model path not found: {self.model_path}"
            logger.error(self.last_error)
            return False

        required_files = ["config.json"]
        for file_name in required_files:
            file_path = os.path.join(self.model_path, file_name)
            if not os.path.exists(file_path):
                self.last_error = f"Required model file missing: {file_name}"
                logger.error(self.last_error)
                return False

        return True

    @with_request_id
    def get_response(self, prompt: str) -> str:
        """
        Generate response using TinyLlama model with detailed timing.
        Implements your framework's AIModel interface.
        """
        method_start = time.time()
        request_id = get_request_id()

        logger.info(f"[TINYLLAMA TIMING] Starting get_response for prompt length: {len(prompt)}")

        if not self.model_loaded or self.model is None or self.tokenizer is None:
            error_msg = f"TinyLlama model not available. Error: {self.last_error}"
            logger.error(error_msg)
            return error_msg

        if not prompt or len(prompt.strip()) == 0:
            return "Please provide a valid prompt."

        logger.info(f"[TINYLLAMA TIMING] Model validation completed in {time.time() - method_start:.3f}s")

        result = {"response": None, "error": None, "completed": False}

        def generate():
            try:
                gen_start = time.time()
                logger.info(f"[TINYLLAMA TIMING] Starting generation thread...")

                # 1. Chat template formatting timing
                template_start = time.time()
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]

                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                template_time = time.time() - template_start
                logger.info(f"[TINYLLAMA TIMING] Chat template formatting: {template_time:.3f}s")

                # 2. Tokenization timing
                tokenize_start = time.time()
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=768
                )
                tokenize_time = time.time() - tokenize_start
                input_token_count = inputs['input_ids'].shape[-1]
                logger.info(f"[TINYLLAMA TIMING] Tokenization: {tokenize_time:.3f}s ({input_token_count} tokens)")

                # 3. Model generation timing (the big one!)
                generation_start = time.time()
                logger.info(f"[TINYLLAMA TIMING] Starting model.generate() with max_tokens={self.max_tokens}...")

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                generation_time = time.time() - generation_start
                output_token_count = outputs[0].shape[-1]
                new_token_count = output_token_count - input_token_count

                logger.info(f"[TINYLLAMA TIMING] MODEL GENERATION: {generation_time:.3f}s")
                logger.info(f"[TINYLLAMA TIMING] Generated {new_token_count} new tokens (total: {output_token_count})")

                if generation_time > 30:
                    logger.warning(
                        f"[TINYLLAMA TIMING] SLOW GENERATION WARNING: {generation_time:.3f}s for {new_token_count} tokens")
                    logger.warning(f"[TINYLLAMA TIMING] Tokens per second: {new_token_count / generation_time:.2f}")

                # 4. Response extraction and decoding timing
                decode_start = time.time()
                response_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
                response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                decode_time = time.time() - decode_start
                logger.info(f"[TINYLLAMA TIMING] Response decoding: {decode_time:.3f}s")

                # 5. Total generation timing
                total_gen_time = time.time() - gen_start
                logger.info(f"[TINYLLAMA TIMING] Total generation thread: {total_gen_time:.3f}s")
                logger.info(f"[TINYLLAMA TIMING] BREAKDOWN - Template: {template_time:.3f}s, "
                            f"Tokenize: {tokenize_time:.3f}s, "
                            f"Generate: {generation_time:.3f}s, "
                            f"Decode: {decode_time:.3f}s")

                result["response"] = response_text
                result["completed"] = True
                logger.info(f"[TINYLLAMA TIMING] Response generated: {len(response_text)} characters")

            except Exception as e:
                error_time = time.time() - gen_start
                result["error"] = str(e)
                logger.error(f"[TINYLLAMA TIMING] Generation error after {error_time:.3f}s: {e}")

        # Thread execution timing
        thread_start = time.time()
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

        logger.info(f"[TINYLLAMA TIMING] Thread started, waiting up to {self.generation_timeout}s...")
        thread.join(timeout=self.generation_timeout)
        thread_time = time.time() - thread_start

        logger.info(f"[TINYLLAMA TIMING] Thread execution completed in {thread_time:.3f}s")

        # Handle results with timing
        result_start = time.time()

        if not result["completed"]:
            warning_msg = f"TinyLlama response timed out after {self.generation_timeout}s"
            logger.warning(f"[TINYLLAMA TIMING] TIMEOUT: {warning_msg}")
            return warning_msg

        if result["error"]:
            error_msg = f"TinyLlama generation error: {result['error']}"
            logger.error(f"[TINYLLAMA TIMING] ERROR: {error_msg}")
            return error_msg

        result_time = time.time() - result_start
        total_method_time = time.time() - method_start

        logger.info(f"[TINYLLAMA TIMING] Result handling: {result_time:.3f}s")
        logger.info(f"[TINYLLAMA TIMING] TOTAL get_response METHOD: {total_method_time:.3f}s")

        # Performance analysis
        if total_method_time > 30:
            logger.warning(f"[TINYLLAMA TIMING] PERFORMANCE WARNING: Total time {total_method_time:.3f}s is very slow")
            logger.warning(
                f"[TINYLLAMA TIMING] Consider reducing max_tokens (current: {self.max_tokens}) or switching models")

        return result["response"] or "No response generated."

    @with_request_id
    def generate_description(self, image_path: str) -> str:
        """
        Generate image description with timing.
        Implements your framework's AIModel interface.
        """
        method_start = time.time()
        request_id = get_request_id()

        logger.info(f"[TINYLLAMA TIMING] generate_description called for: {image_path}")

        # Validate input
        validation_start = time.time()
        if not image_path:
            logger.warning(f"[TINYLLAMA TIMING] Empty image_path provided")
            return "No image path provided."

        validation_time = time.time() - validation_start
        logger.debug(f"[TINYLLAMA TIMING] Input validation: {validation_time:.6f}s")

        # Generate response
        response_start = time.time()
        response = "Image description not supported by TinyLlama text model. Please use a vision model."
        response_time = time.time() - response_start

        total_time = time.time() - method_start

        logger.info(f"[TINYLLAMA TIMING] generate_description completed in {total_time:.6f}s")
        logger.debug(
            f"[TINYLLAMA TIMING] BREAKDOWN - Validation: {validation_time:.6f}s, Response: {response_time:.6f}s")

        return response

    @with_request_id
    def get_model_info(self):
        """Get comprehensive model information for your framework."""
        info = {
            "status": "loaded" if self.model_loaded else "not_loaded",
            "model_name": self.model_name,
            "model_path": self.model_path,
            "backend": "transformers",
            "quantized": QUANTIZATION_AVAILABLE and self.config['enable_quantization'],
            "compiled": TORCH_COMPILE_AVAILABLE and self.config['enable_compile'],
            "device": "CPU",
            "parameters": "1.1B",
            "timeout": f"{self.generation_timeout}s",
            "max_tokens": self.max_tokens,
            "local": True,
            "privacy": "full - no data sent externally",
            "error": self.last_error,
            "framework_integrated": True  # New flag
        }

        if self.model_loaded:
            info["cache_status"] = "cached" if self.model_path in self._model_cache else "not_cached"

        return info

    @classmethod
    def clear_cache(cls):
        """Clear model cache to free memory."""
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
        cls._cache_initialized = False
        if LOGGING_AVAILABLE:
            logger.info("Cleared TinyLlama model cache")

    @classmethod
    def get_cache_stats(cls):
        """Get cache statistics."""
        return {
            "models_cached": len(cls._model_cache),
            "cache_initialized": cls._cache_initialized,
            "cached_paths": list(cls._model_cache.keys())
        }

    # Framework compatibility methods
    def is_available(self):
        """Check if the model is available and ready."""
        return self.model_loaded and self.model is not None

    def get_capabilities(self):
        """Get model capabilities for your framework."""
        return {
            "text_generation": True,
            "image_description": False,
            "chat": True,
            "streaming": False,
            "offline": True,
            "local": True
        }

# Update your register_default_models function to ensure TinyLlama configs are included
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


# Test function for TinyLlama integration
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


class TinyLlamaEmbeddingModel(EmbeddingModel):
    _instance = None
    _model_loaded = False
    _cached_model = None

    def __new__(cls, model_name="all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Only initialize once
        if self._model_loaded:
            logger.info(f"Using cached TinyLlama embedding model: {model_name}")
            return

        logger.info(f"Loading TinyLlama embedding model: {model_name}")
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        self._initialize_model()
        TinyLlamaEmbeddingModel._model_loaded = True

    def _initialize_model(self):
        """Initialize the model only if not already cached"""
        if TinyLlamaEmbeddingModel._cached_model is not None:
            logger.info("Using existing cached TinyLlama model")
            self.model = TinyLlamaEmbeddingModel._cached_model
            self.is_loaded = True
            return

        try:
            logger.info(f"Attempting to load TinyLlama embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            TinyLlamaEmbeddingModel._cached_model = self.model  # Cache it
            self.is_loaded = True

            # Test the model
            logger.info("Testing TinyLlama embedding model...")
            test_embedding = self.model.encode("test")
            logger.info("TinyLlama embedding model test successful!")
            logger.info(f"   - Model: {self.model_name}")
            logger.info(f"   - Embedding dimensions: {len(test_embedding)}")
            logger.info(f"   - Optimized for: Fast local inference")
            logger.info(f"Successfully loaded TinyLlama embedding model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load TinyLlama embedding model: {e}")
            self.model = None
            self.is_loaded = False
            raise

    def preload_embedding_model_properly():
        """Ensure embedding model is properly preloaded and cached"""
        try:
            logger.info("Force-preloading TinyLlama embedding model...")

            # Force creation of singleton instance
            embedding_model = TinyLlamaEmbeddingModel()

            # Force initialization if not already done
            if not embedding_model.is_loaded:
                embedding_model._initialize_model()

            # Test it works
            test_result = embedding_model.get_embeddings("preload test")

            if test_result and len(test_result) > 0:
                logger.info(f"Embedding model preloaded successfully! Dimensions: {len(test_result)}")
                return True
            else:
                logger.error("Embedding model preload failed - no test result")
                return False

        except Exception as e:
            logger.error(f"Failed to preload embedding model: {e}")
            return False

    def _initialize_model(self):
        """Initialize the TinyLlama-optimized embedding model"""
        logger.info(f"Loading TinyLlama embedding model: {self.model_name}")

        try:
            from sentence_transformers import SentenceTransformer

            # Set performance settings for TinyLlama workflow
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # Try the specified model first
            if self._try_load_model(self.model_name):
                logger.info(f"Successfully loaded TinyLlama embedding model: {self.model_name}")
                return

            # If specified model fails, try TinyLlama-compatible models
            logger.warning(f"Primary model {self.model_name} failed, trying TinyLlama-compatible alternatives...")

            for model_name in self.tinyllama_compatible_models:
                if self._try_load_model(model_name):
                    logger.info(f"Successfully loaded fallback TinyLlama embedding model: {self.model_name}")
                    return

            # If all models fail
            logger.warning("All TinyLlama embedding models failed to load")
            logger.warning("   Vector search will not be available for TinyLlama")
            self.model = None
            self.is_loaded = False

        except ImportError:
            logger.error("SentenceTransformers not installed. Install with: pip install sentence-transformers")
            self.model = None
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Unexpected error initializing TinyLlama embedding model: {e}")
            self.model = None
            self.is_loaded = False

    def _try_load_model(self, model_name):
        """Try to load a specific embedding model optimized for TinyLlama"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Attempting to load TinyLlama embedding model: {model_name}")

            # Load model with TinyLlama-optimized settings
            self.model = SentenceTransformer(model_name)

            # Test the model
            logger.info("Testing TinyLlama embedding model...")
            test_embedding = self.model.encode(["TinyLlama test sentence"], show_progress_bar=False)

            if len(test_embedding) > 0 and len(test_embedding[0]) > 0:
                logger.info(f"TinyLlama embedding model test successful!")
                logger.info(f"   - Model: {model_name}")
                logger.info(f"   - Embedding dimensions: {len(test_embedding[0])}")
                logger.info(f"   - Optimized for: Fast local inference")

                # Get cache location if available
                if hasattr(self.model, 'cache_folder'):
                    logger.info(f"   - Cache location: {self.model.cache_folder}")

                self.model_name = model_name
                self.is_loaded = True
                return True
            else:
                logger.warning(f"TinyLlama embedding model {model_name} loaded but failed test")
                return False

        except Exception as e:
            logger.debug(f"Failed to load TinyLlama embedding model {model_name}: {e}")
            return False

    def get_embeddings(self, text):
        """Get embeddings using cached model"""
        if not self.is_loaded or self.model is None:
            logger.warning("TinyLlama embedding model not loaded, attempting to initialize...")
            self._initialize_model()

        if not self.is_loaded:
            logger.error("Failed to load TinyLlama embedding model")
            return None

        try:
            logger.debug(f"Generating TinyLlama embeddings with model: {self.model_name}")
            embedding = self.model.encode(text)
            logger.debug(f"Generated TinyLlama embeddings: {len(embedding)} dimensions")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating TinyLlama embeddings: {e}")
            return None

    def is_available(self):
        """Check if the TinyLlama embedding model is available and loaded"""
        return self.model is not None and self.is_loaded

    def get_model_info(self):
        """Get information about the loaded TinyLlama embedding model"""
        if self.is_available():
            try:
                test_embedding = self.model.encode(["test"], show_progress_bar=False)

                # Try to get cache info
                cache_info = "automatic"
                if hasattr(self.model, 'cache_folder'):
                    cache_info = self.model.cache_folder
                elif hasattr(self.model, '_cache_folder'):
                    cache_info = self.model._cache_folder

                return {
                    "model_name": self.model_name,
                    "embedding_dim": len(test_embedding[0]),
                    "cache_location": cache_info,
                    "status": "available",
                    "optimized_for": "TinyLlama local workflow",
                    "compatible_models": self.tinyllama_compatible_models
                }
            except Exception as e:
                return {
                    "model_name": self.model_name,
                    "status": f"error: {e}",
                    "optimized_for": "TinyLlama local workflow"
                }
        else:
            return {
                "model_name": self.model_name,
                "status": "not_available",
                "optimized_for": "TinyLlama local workflow",
                "compatible_models": self.tinyllama_compatible_models
            }


def test_tinyllama_embedding_functionality():
    """Test function to verify TinyLlama embedding generation is working properly"""
    logger.info("Testing TinyLlama embedding functionality...")

    try:
        # Test model loading
        tinyllama_embedding = TinyLlamaEmbeddingModel()

        if not tinyllama_embedding.is_available():
            logger.error("TinyLlama embedding model loading test failed")
            return False

        logger.info("TinyLlama embedding model loading test passed")

        # Test embedding generation
        test_text = "TinyLlama embedding test sentence"
        embeddings = tinyllama_embedding.get_embeddings(test_text)

        if embeddings and len(embeddings) > 0:
            logger.info(f"TinyLlama embedding generation test passed: {len(embeddings)} dimensions")
            return True
        else:
            logger.error("TinyLlama embedding generation test failed")
            return False

    except Exception as e:
        logger.error(f"TinyLlama embedding functionality test failed: {e}")
        return False



# Update the register_default_models function to include TinyLlama
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


# Add diagnostic function for TinyLlama
@with_request_id
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


# Add test function for TinyLlama
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

@with_request_id
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


def download_recommended_models():
    """Download recommended models for offline use"""
    recommended_models = {
        "gpt4all": [
            "Meta-Llama-3-8B-Instruct.Q4_0.gguf",  # 4.66GB, good balance
            "mistral-7b-openorca.gguf2.Q4_0.gguf"  # Alternative model
        ],
        "sentence_transformer": [
            "nomic-ai/nomic-embed-text-v1",  # Your current choice
            "all-MiniLM-L6-v2"  # Lighter alternative
        ]
    }

    setup_info = check_gpt4all_setup()
    downloads = []

    # Ensure directories exist
    os.makedirs(GPT4ALL_MODELS_PATH, exist_ok=True)
    os.makedirs(SENTENCE_TRANSFORMERS_MODELS_PATH, exist_ok=True)

    # Download GPT4All models
    if setup_info["gpt4all_installed"]:
        try:
            from gpt4all import GPT4All
            for model_name in recommended_models["gpt4all"]:
                model_path = os.path.join(GPT4ALL_MODELS_PATH, model_name)
                if not os.path.exists(model_path):
                    logger.info(f"📥 Downloading GPT4All model: {model_name}")
                    try:
                        # Download to specified directory
                        model = GPT4All(model_name, model_path=GPT4ALL_MODELS_PATH, allow_download=True)
                        downloads.append(f"Downloaded {model_name}")
                        logger.info(f"Successfully downloaded {model_name}")
                    except Exception as e:
                        downloads.append(f"Failed to download {model_name}: {e}")
                        logger.error(f"Failed to download {model_name}: {e}")
                else:
                    downloads.append(f"detected_intent_id = intent_classification['intent_id']{model_name} already exists")
        except Exception as e:
            downloads.append(f"GPT4All download error: {e}")

    # Download SentenceTransformer models
    if setup_info["sentence_transformers_installed"]:
        try:
            from sentence_transformers import SentenceTransformer
            for model_name in recommended_models["sentence_transformer"]:
                local_path = os.path.join(SENTENCE_TRANSFORMERS_MODELS_PATH, model_name.split('/')[-1])
                if not os.path.exists(local_path):
                    logger.info(f"📥 Downloading SentenceTransformer: {model_name}")
                    try:
                        model = SentenceTransformer(model_name, cache_folder=SENTENCE_TRANSFORMERS_MODELS_PATH)
                        downloads.append(f"Downloaded {model_name}")
                        logger.info(f"Successfully downloaded {model_name}")
                    except Exception as e:
                        downloads.append(f"Failed to download {model_name}: {e}")
                        logger.error(f"Failed to download {model_name}: {e}")
                else:
                    downloads.append(f"detected_intent_id = intent_classification['intent_id']{model_name} already exists")
        except Exception as e:
            downloads.append(f"SentenceTransformer download error: {e}")

    return downloads


def get_available_local_models():
    """Get list of locally available models for configuration"""
    local_models = {
        "gpt4all": [],
        "sentence_transformer": []
    }

    # Scan GPT4All models
    if os.path.exists(GPT4ALL_MODELS_PATH):
        for file in os.listdir(GPT4ALL_MODELS_PATH):
            if file.endswith('.gguf') or file.endswith('.bin'):
                model_path = os.path.join(GPT4ALL_MODELS_PATH, file)
                local_models["gpt4all"].append({
                    "name": file,
                    "display_name": file.replace('.gguf', '').replace('.bin', ''),
                    "path": model_path,
                    "size_gb": round(os.path.getsize(model_path) / (1024 ** 3), 2),
                    "enabled": True
                })

    # Scan SentenceTransformer models
    if os.path.exists(SENTENCE_TRANSFORMERS_MODELS_PATH):
        for dir_name in os.listdir(SENTENCE_TRANSFORMERS_MODELS_PATH):
            model_path = os.path.join(SENTENCE_TRANSFORMERS_MODELS_PATH, dir_name)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                local_models["sentence_transformer"].append({
                    "name": dir_name,
                    "display_name": dir_name.replace('-', ' ').title(),
                    "path": model_path,
                    "enabled": True
                })

    return local_models


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

# Configuration management functions
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


def get_tinyllama_config():
    """Get TinyLlama-specific configuration values."""
    try:
        model_path = ModelsConfig.get_config_value('ai', 'TINYLLAMA_MODEL_PATH',
                                                   r"C:\Users\10169062\PycharmProjects\MDEV_EMTAC\plugins\ai_modules\TinyLlama_1_1B")
        timeout = int(ModelsConfig.get_config_value('ai', 'TINYLLAMA_TIMEOUT', '120'))
        max_tokens = int(ModelsConfig.get_config_value('ai', 'TINYLLAMA_MAX_TOKENS', '256'))

        return {
            'model_path': model_path,
            'timeout': timeout,
            'max_tokens': max_tokens
        }
    except Exception as e:
        logger.error(f"Error getting TinyLlama config: {e}")
        return {
            'model_path': r"C:\Users\10169062\PycharmProjects\MDEV_EMTAC\plugins\ai_modules\TinyLlama_1_1B",
            'timeout': 120,
            'max_tokens': 256
        }


def update_tinyllama_config(model_path=None, timeout=None, max_tokens=None):
    """Update TinyLlama-specific configuration values."""
    try:
        updated = False

        if model_path is not None:
            success = ModelsConfig.set_config_value('ai', 'TINYLLAMA_MODEL_PATH', model_path)
            if success:
                logger.info(f"Updated TinyLlama model path: {model_path}")
                updated = True

        if timeout is not None:
            success = ModelsConfig.set_config_value('ai', 'TINYLLAMA_TIMEOUT', str(timeout))
            if success:
                logger.info(f"Updated TinyLlama timeout: {timeout}s")
                updated = True

        if max_tokens is not None:
            success = ModelsConfig.set_config_value('ai', 'TINYLLAMA_MAX_TOKENS', str(max_tokens))
            if success:
                logger.info(f"Updated TinyLlama max tokens: {max_tokens}")
                updated = True

        return updated
    except Exception as e:
        logger.error(f"Error updating TinyLlama config: {e}")
        return False


import os

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


# Enhanced embedding generation and storage functions
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


# Utility functions for model management
def get_current_models():
    """Get information about all currently active models."""
    try:
        ai_model = ModelsConfig.get_config_value('ai', 'CURRENT_MODEL', 'NoAIModel')
        embedding_model = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')
        image_model = ModelsConfig.get_config_value('image', 'CURRENT_MODEL', 'NoImageModel')

        return {
            'ai': ai_model,
            'embedding': embedding_model,
            'image': image_model
        }
    except Exception as e:
        logger.error(f"Error getting current models: {e}")
        return {
            'ai': 'NoAIModel',
            'embedding': 'NoEmbeddingModel',
            'image': 'NoImageModel'
        }


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


# Legacy function names for backward compatibility
def load_ai_model(model_name=None):
    """Legacy function - use ModelsConfig.load_ai_model instead."""
    return ModelsConfig.load_ai_model(model_name)


def load_embedding_model(model_name=None):
    """Legacy function - use ModelsConfig.load_embedding_model instead."""
    return ModelsConfig.load_embedding_model(model_name)


def load_image_model(model_name=None):
    """Legacy function - use ModelsConfig.load_image_model instead."""
    return ModelsConfig.load_image_model(model_name)


# Initialize models config on import
try:
    initialize_models_config()
    logger.info("AI models module initialized successfully")
except Exception as e:
    logger.error(f"Error during AI models module initialization: {e}")


# ==========================================
# FRAMEWORK INTEGRATION EXAMPLES
# ==========================================

def example_completeDocument_integration():
    """
    Example showing proper integration with CompleteDocument class.
    This demonstrates the correct usage patterns for the framework.
    """
    from modules.configuration.config_env import DatabaseConfig
    from modules.emtacdb.emtacdb_fts import DocumentEmbedding

    db_config = DatabaseConfig()

    with db_config.main_session() as session:
        # Example 1: Generate and store embedding for a document chunk
        document_id = 123
        content = "This is sample document content for embedding generation."

        success = generate_and_store_embedding(session, document_id, content)
        if success:
            print("Embedding generated and stored successfully")

        # Example 2: Store pre-generated embeddings
        embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example embedding vector
        success = store_embedding_enhanced(session, document_id, embeddings, "OpenAIEmbeddingModel")
        if success:
            print("Pre-generated embedding stored successfully")

        # Example 3: Query stored embeddings
        embedding_record = session.query(DocumentEmbedding).filter_by(
            document_id=document_id,
            model_name="OpenAIEmbeddingModel"
        ).first()

        if embedding_record:
            stored_embeddings = embedding_record.embedding_vector  # Uses property method
            print(f"Retrieved {len(stored_embeddings)} dimension embedding")


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


# Update the integration example too:
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


def get_tinyllama_embedding_config():
    """Get TinyLlama embedding-specific configuration values."""
    try:
        embedding_model = ModelsConfig.get_config_value('embedding', 'TINYLLAMA_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        cache_path = ModelsConfig.get_config_value('embedding', 'TINYLLAMA_EMBEDDING_CACHE_PATH',
                                                   os.path.join(SENTENCE_TRANSFORMERS_MODELS_PATH,
                                                                "tinyllama_embeddings"))
        return {
            'embedding_model': embedding_model,
            'cache_path': cache_path
        }
    except Exception as e:
        logger.error(f"Error getting TinyLlama embedding config: {e}")
        return {
            'embedding_model': 'all-MiniLM-L6-v2',
            'cache_path': os.path.join(SENTENCE_TRANSFORMERS_MODELS_PATH, "tinyllama_embeddings")
        }


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