#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Global configuration for OPTION C Q&A SYSTEM.
This file defines ALL directory paths used by the pipeline, tools,
and model loaders.

Every other script should import from here:
    from option_c_qna.configuration.config import cfg
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:

    def __init__(self):
        # ---------------------------------------------------------
        # Load environment variables
        # ---------------------------------------------------------
        ENV_PATH = Path(r"E:\emtac\dev_env\.env")
        if not ENV_PATH.exists():
            raise FileNotFoundError(f".env file not found at: {ENV_PATH}")

        load_dotenv(ENV_PATH)
        self.ENV_PATH = ENV_PATH

        # ---------------------------------------------------------
        # BASE ROOT DIR for DATASET_GEN
        # ---------------------------------------------------------
        self.PROJECT_ROOT = Path(__file__).resolve().parents[2]

        # ---------------------------------------------------------
        # MODEL DIRECTORIES (from .env)
        # ---------------------------------------------------------
        self.MODELS_LLM_DIR = Path(os.getenv("MODELS_LLM_DIR"))
        self.MODELS_QWEN_DIR = Path(os.getenv("MODELS_QWEN_DIR"))
        self.MODELS_TINY_LLAMA_DIR = Path(os.getenv("MODELS_TINY_LLAMA_DIR"))
        self.MODELS_APPLE_ELM_DIR = Path(os.getenv("MODELS_APPLE_ELM_DIR"))
        self.MODELS_GEMMA_DIR = Path(os.getenv("MODELS_GEMMA_DIR"))
        self.MODELS_FLAN_DIR = Path(os.getenv("MODEL_FLAN_DIR"))
        self.MODEL_MINILM_DIR = Path(os.getenv("MODEL_MINILM_DIR"))
        self.MODEL_mrm8488_DIR = Path(os.getenv("MODEL_mrm8488_DIR"))
        self.MODELS_MISTRAL_7B_DIR = Path(os.getenv("MODELS_MISTRAL_7B_DIR"))

        # ---------------------------------------------------------
        # PIPELINE OUTPUT DIRECTORIES
        # option_c_qna/qna/structure
        # option_c_qna/qna/clean_chunks
        # etc.
        # ---------------------------------------------------------
        qna_base = self.PROJECT_ROOT / "option_c_qna" / "qna"

        self.STRUCTURE_DIR = qna_base / "structure"
        self.CLEAN_DIR = qna_base / "clean_chunks"
        self.QUESTIONS_DIR = qna_base / "questions"
        self.ANSWERS_DIR = qna_base / "answers"

        # ---------------------------------------------------------
        # LOGS DIR
        # ---------------------------------------------------------
        self.LOG_DIR = self.PROJECT_ROOT / "option_c_qna" / "logs"

        # ---------------------------------------------------------
        # DB TOOLS DIRECTORIES (your request)
        # ---------------------------------------------------------
        self.DB_TOOLS_DIR = self.PROJECT_ROOT / "option_c_qna" / "tools" / "db_tools"
        self.DB_TOOLS_OUTPUT_DIR = self.DB_TOOLS_DIR / "output"

        # EXAMPLE:
        #   E:\emtac\projects\llm\DATASET_GEN\option_c_qna\tools\db_tools\output

        # ---------------------------------------------------------
        # Create all directories
        # ---------------------------------------------------------
        dirs = [
            self.STRUCTURE_DIR,
            self.CLEAN_DIR,
            self.QUESTIONS_DIR,
            self.ANSWERS_DIR,
            self.LOG_DIR,
            self.DB_TOOLS_DIR,
            self.DB_TOOLS_OUTPUT_DIR,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Singleton
cfg = Config()
