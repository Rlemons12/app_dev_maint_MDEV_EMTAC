
# modules/emtac_ai/query_expansion/query_expansion_core.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
from modules.configuration.log_config import debug_id, info_id, warning_id, error_id, get_request_id
from .synonym_loader import SynonymLoader
from .query_utils import dedup_preserve_order, replace_word_boundary, tokenize_words_lower

class QueryExpansionRAG:
    """
    Core query expansion engine.
    - Intent-aware synonym expansion
    - Entity-based tweaks
    - Optional AI-based generation (if an ai_model is provided)
    - Optional HyDE/PRF helpers (lightweight, backend-agnostic)
    """
    def __init__(self, ai_model: Optional[Any] = None, domain: str = "maintenance"):
        self.ai_model = ai_model
        self.domain = domain
        self.syn_loader = SynonymLoader(ai_model=ai_model, domain=domain)

    # ---------- Base expansions ----------

    def expand_by_intent(self, query: str, intent: Optional[str]) -> List[str]:
        """
        Replace base terms with intent-specific synonyms (boundary-safe).
        """
        req = get_request_id()
        syn_map = self.syn_loader.get_intent_synonyms(intent)
        expansions: List[str] = []
        for base_term, synonyms in syn_map.items():
            if not isinstance(synonyms, (list, tuple)):
                continue
            if base_term.lower() in (query or "").lower():
                for syn in synonyms:
                    expansions.append(replace_word_boundary(query, base_term, syn))
        out = dedup_preserve_order(expansions)
        debug_id(f"[RAG] expand_by_intent produced {len(out)} variants", req)
        return out

    def expand_by_entities(self, query: str, entities: Iterable[Dict[str, Any]]) -> List[str]:
        """
        Lightweight entity-driven expansions. Example: add 'spec', 'datasheet' around part numbers.
        Entities from IntentEntityPlugin carry keys like: word, entity_group, score, start, end.
        """
        req = get_request_id()
        variants: List[str] = []
        for ent in (entities or []):
            word = ent.get("word")
            label = ent.get("entity_group") or ent.get("entity")
            if not word:
                continue
            # Simple policy: for PARTNUM/part numbers, expand with data-centric contexts
            if label and ("PARTNUM" in label or "PART" in label):
                variants += [
                    f"{word} datasheet",
                    f"{word} specification",
                    f"{word} manual",
                    f"{word} replacement",
                ]
            # For descriptions, create adjective-like expansions
            if label and ("DESC" in label or "DESCRIPTION" in label):
                variants += [
                    f"{word} troubleshooting",
                    f"{word} installation",
                    f"{word} maintenance",
                ]
        out = dedup_preserve_order(variants)
        debug_id(f"[RAG] expand_by_entities produced {len(out)} variants", req)
        return out

    # ---------- Rule-based & AI-based expansion helpers (compatibility layer) ----------

    def multi_query_expansion_rules(self, query: str, intent: Optional[str]) -> List[str]:
        """
        Compatibility wrapper to provide a 'rule-based' expansion list that mirrors prior API.
        """
        req = get_request_id()
        variants: List[str] = []
        # Intent-driven replacements:
        variants += self.expand_by_intent(query, intent)
        # Acronym expansions (both directions)
        acr = self.syn_loader.get_acronyms()
        for short, long in acr.items():
            if short.lower() in query.lower():
                variants.append(query.replace(short, long))
            if long.lower() in query.lower():
                variants.append(query.replace(long, short))
        out = dedup_preserve_order(variants)
        debug_id(f"[RAG] multi_query_expansion_rules -> {len(out)}", req)
        return out

    def multi_query_expansion_ai(self, query: str, intent: Optional[str]) -> List[str]:
        """
        AI-based paraphrase/intent templating (if ai_model supports it). Safe no-op fallback.
        Expected ai_model to implement get_response(prompt: str) -> str.
        """
        req = get_request_id()
        if self.ai_model is None:
            return []
        try:
            prompt = (
                "Paraphrase the maintenance search query in 5 distinct and concise ways, "
                "preserving technical meaning. Return as JSON array only.\n"
                f"INTENT: {intent or 'general'}\nQUERY: {query}"
            )
            raw = self.ai_model.get_response(prompt)
            # minimal JSON parsing without strict deps
            import json, re
            match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
            if not match:
                return []
            arr = json.loads(match.group(0))
            strings = [str(x) for x in arr if isinstance(x, str)]
            out = dedup_preserve_order(strings)
            debug_id(f"[RAG] AI expansion produced {len(out)} variants", req)
            return out
        except Exception as e:
            warning_id(f"[RAG] AI expansion failed: {e}", req)
            return []

    # ---------- HyDE / PRF (lightweight, optional) ----------

    def hyde(self, query: str) -> Tuple[Optional[str], Optional[List[float]]]:
        """
        HyDE: draft a hypothetical passage about the query, then embed.
        If no ai_model/embedding available, returns (None, None) safely.
        """
        req = get_request_id()
        hypo = None
        emb = None
        try:
            if self.ai_model is None:
                return None, None
            prompt = (
                "Write a short technical paragraph (120-180 words) that would answer the following "
                "maintenance-related search query. Keep it factual and concise:\n"
                f"QUERY: {query}"
            )
            hypo = self.ai_model.get_response(prompt)
            # If your environment provides an embedding model, you could hook it here.
            # We'll stay backend-agnostic and not embed by default.
            debug_id("[RAG] HyDE generated hypothetical content", req)
            return hypo, emb
        except Exception as e:
            warning_id(f"[RAG] HyDE failed: {e}", req)
            return None, None

    def prf(self, top_docs: Iterable[str], max_terms: int = 10) -> List[str]:
        """
        Pseudo-Relevance Feedback over raw text snippets.
        Extracts frequent tokens (naive), returns as additive terms.
        """
        req = get_request_id()
        tokens: List[str] = []
        for doc in (top_docs or []):
            tokens += tokenize_words_lower(doc)
        # naive term frequency
        from collections import Counter
        ctr = Counter(t for t in tokens if len(t) > 2)
        common = [w for (w, _) in ctr.most_common(max_terms)]
        debug_id(f"[RAG] PRF extracted {len(common)} common terms", req)
        return common

    # ---------- Comprehensive driver ----------

    def comprehensive_expand(
        self,
        query: str,
        intent: Optional[str],
        entities: Optional[Iterable[Dict[str, Any]]] = None,
        enable_ai: bool = True
    ) -> Dict[str, Any]:
        """
        End-to-end expansion that merges rule/AI/entity expansions with dedup.
        """
        req = get_request_id()
        rules = self.multi_query_expansion_rules(query, intent)
        ents = self.expand_by_entities(query, entities or [])
        ai = self.multi_query_expansion_ai(query, intent) if enable_ai else []
        combined = dedup_preserve_order([query] + rules + ents + ai)
        info_id(f"[RAG] comprehensive_expand -> {len(combined)} queries", req)
        return {
            "query": query,
            "intent": intent,
            "rules": rules,
            "entity_expansions": ents,
            "ai_expansions": ai,
            "final_expanded_queries": combined,
        }
