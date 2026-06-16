# modules/emtac_ai/query_expansion/synonym_loader.py
from __future__ import annotations

import json
import re
from typing import Dict, Optional, Any
from modules.configuration.log_config import debug_id, warning_id, get_request_id

def _safe_json_object(s: str) -> Optional[dict]:
    try:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

class SynonymLoader:
    """
    Central access point for synonyms/acronyms/rules.

    - If `ai_model` is provided (must implement .get_response(prompt)->str),
      we TRY to generate an intent-aware synonym MAP *directly* from the model.
      If parsing fails or the model is unavailable, we gracefully fall back to defaults.
    - No dependency on gen_syn_db or any DB layer.
    """

    def __init__(self, ai_model: Optional[Any] = None, domain: str = "maintenance"):
        self.domain = domain
        self.ai_model = ai_model

        # Minimal safe defaults to ensure functionality without AI/DB
        self._defaults = {
            "intent_synonyms": {
                "parts": {
                    "pump": ["pump", "centrifugal pump", "booster pump"],
                    "motor": ["motor", "electric motor", "drive"],
                    "valve": ["valve", "gate valve", "control valve", "ball valve"],
                },
                "troubleshooting": {
                    "noise": ["noise", "sound", "rattle"],
                    "vibration": ["vibration", "shake", "oscillation"],
                    "leak": ["leak", "seep", "drip"],
                },
                "documents": {
                    "manual": ["manual", "guide", "handbook", "documentation"],
                    "specification": ["specification", "spec", "standard"],
                },
                "prints": {
                    "schematic": ["schematic", "diagram", "drawing", "blueprint"],
                },
                "images": {
                    "photo": ["photo", "image", "picture"],
                },
                "tools": {
                    "wrench": ["wrench", "spanner", "socket"],
                },
                "general": {
                    "installation": ["installation", "setup", "assembly"],
                    "maintenance": ["maintenance", "service", "repair", "upkeep"],
                },
            },
            "acronyms": {
                "HVAC": "Heating Ventilation Air Conditioning",
                "VFD": "Variable Frequency Drive",
                "PLC": "Programmable Logic Controller",
                "HMI": "Human Machine Interface",
                "SCADA": "Supervisory Control and Data Acquisition",
            },
        }

    def _ai_generate_intent_map(self, intent: str) -> Optional[Dict[str, list]]:
        """
        Ask the AI model for a JSON OBJECT mapping:
            { "pump": ["centrifugal pump", ...], "manual": ["guide", ...], ... }
        Returns None if no model / parse fails / bad schema.
        """
        if self.ai_model is None:
            return None

        rid = get_request_id()
        try:
            prompt = (
                "You expand short technical maintenance search queries. "
                "Return a SMALL JSON object mapping base terms to concise synonyms/aliases "
                f"(domain: {self.domain}, intent: {intent}). "
                "Respond with JSON only, no commentary. Example:\n"
                "{\n"
                '  "pump": ["centrifugal pump", "booster pump"],\n'
                '  "manual": ["guide", "handbook"],\n'
                '  "valve": ["control valve", "ball valve"]\n'
                "}"
            )
            raw = self.ai_model.get_response(prompt)
            if not isinstance(raw, str):
                return None
            obj = _safe_json_object(raw)
            if not isinstance(obj, dict) or not obj:
                return None

            # Normalize to {str: [str,...]}
            out: Dict[str, list] = {}
            for k, v in obj.items():
                if not isinstance(k, str):
                    continue
                key = k.strip().lower()
                if not key:
                    continue
                if isinstance(v, list):
                    vals = [str(x).strip() for x in v if isinstance(x, (str, int, float))]
                    vals = [x for x in vals if x]
                    if vals:
                        out[key] = vals
                elif isinstance(v, str):
                    vs = v.strip()
                    if vs:
                        out[key] = [vs]
            # Empty or malformed → None to trigger fallback
            return out or None
        except Exception as e:
            warning_id(f"SynonymLoader: AI generation error: {e}", rid)
            return None

    def get_intent_synonyms(self, intent: Optional[str]) -> Dict[str, list]:
        """
        Returns dict: base_term -> [synonym, ...] for the given intent.
        Tries AI-first (if ai_model provided), then falls back to defaults.
        """
        req = get_request_id()
        intent = (intent or "general").strip().lower()

        # Try AI generation
        ai_map = self._ai_generate_intent_map(intent)
        if isinstance(ai_map, dict) and ai_map:
            debug_id(f"SynonymLoader: using AI synonyms for intent='{intent}' ({len(ai_map)})", req)
            return ai_map

        # Fallback to defaults
        fallback = self._defaults["intent_synonyms"].get(intent) or self._defaults["intent_synonyms"]["general"]
        debug_id(f"SynonymLoader: using fallback synonyms for intent='{intent}' ({len(fallback)})", req)
        return fallback

    def get_acronyms(self) -> Dict[str, str]:
        # Defaults only (no DB)
        return dict(self._defaults["acronyms"])
