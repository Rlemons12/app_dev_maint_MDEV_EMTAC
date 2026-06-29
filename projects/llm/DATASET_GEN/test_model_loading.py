#!/usr/bin/env python
# -*- coding: utf-8 -*-

from option_c_qna.qanda_db.qa_db import LLMModel
from option_c_qna.pipeline.qanda_main_pipeline import _init_answer_models
from option_c_qna.models.flan_qa import FLAN_QA_Model
from option_c_qna.configuration.pg_db_config import get_qna_session
from option_c_qna.configuration.logging_config import get_qna_logger

log = get_qna_logger("test_models")


def main():
    session = get_qna_session()

    print("\n=== EXISTING MODELS IN DATABASE ===")
    models = session.query(LLMModel).all()
    for m in models:
        print(f" - {m.name} (type={m.model_type}, path={m.model_path}, enabled={m.enabled})")

    print("\n=== LOAD ALL ENABLED MODELS ===")
    all_enabled = _init_answer_models(None)
    print("Loaded:", list(all_enabled.keys()))

    print("\n=== LOAD ONLY 'flan,qwen' ===")
    filtered = _init_answer_models(["flan", "qwen"])
    print("Loaded:", list(filtered.keys()))

    print("\n=== TEST UNKNOWN MODEL ===")
    try:
        _ = _init_answer_models(["bozo"])
    except Exception as e:
        print("Caught exception (expected):", e)


if __name__ == "__main__":
    main()
