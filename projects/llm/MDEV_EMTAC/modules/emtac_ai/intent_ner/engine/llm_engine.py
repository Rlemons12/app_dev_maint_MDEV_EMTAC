"""
modules/emtac_ai/intent_ner/engine/llm_engine.py

LLM Engine using ner_config-resolved HF paths.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
)

from modules.emtac_ai.intent_ner.configuration.ner_config import (
    get_intent_model,
    get_entity_model,
)


class LLMENTityEngine:

    def __init__(self, mode="intent", device=None):
        """
        mode: "intent" or "ner"
        """

        if mode == "intent":
            self.model_path = get_intent_model()
        else:
            self.model_path = get_entity_model()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[LLM ENGINE] Loading {mode} model:")
        print(f"  Model path: {self.model_path}")
        print(f"  Device:     {self.device}")

        # Always load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Try causal LM first; if it fails, fall back to encoder-only
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)

            self.model_type = "causal_lm"

        except Exception:
            print("[LLM ENGINE] This model is NOT a Causal LM. Falling back to AutoModel (encoder-only).")

            self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
            self.model_type = "encoder"

    # ----------------------------------------------------------------------
    def _generate(self, prompt, max_tokens=256):
        """
        Only works if the model is a causal language model.
        """
        if self.model_type != "causal_lm":
            raise RuntimeError("This engine cannot generate text using an encoder-only model.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
