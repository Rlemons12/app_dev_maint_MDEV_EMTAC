"""Model download and local model discovery helpers."""


from __future__ import annotations


import os

from modules.configuration.config import (
    OPENAI_API_KEY, OPENAI_MODEL_NAME, DATABASE_URL, ANTHROPIC_API_KEY,
    MODELS_DIR, MODELS_LLM_DIR, MODELS_IMAGE_DIR, MODEL_CLIP_DIR,
    MODELS_TINY_LLAMA_DIR, MODELS_QWEN_DIR,
    HF_HOME, HF_HUB_CACHE, SENTENCE_TRANSFORMERS_MODELS_PATH
)
from modules.configuration.log_config import logger, with_request_id

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
