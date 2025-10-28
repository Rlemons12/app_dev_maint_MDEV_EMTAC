import os
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import json

# Import your existing modules
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    get_request_id, with_request_id, log_timed_operation
)

# Import your existing query expansion system
from query_expansion_techniques import QueryExpansionRAG

# Import your Intent/NER plugin
from emtac_intent_entity import IntentEntityPlugin


class EMTACQueryExpansionOrchestrator:
    """
    Complete orchestrator that integrates:
    1. Your IntentEntityPlugin for intent classification and NER
    2. Your QueryExpansionRAG for base query expansion
    3. Enhanced intent-aware query expansion using synonym dictionaries
    """

    def __init__(self,
                 intent_model_dir=None,
                 ner_model_dir=None,
                 query_expander=None):
        """
        Initialize the complete pipeline

        Args:
            intent_model_dir: Path to your trained intent model
            ner_model_dir: Path to your trained NER model
            query_expander: Optional pre-initialized QueryExpansionRAG instance
        """
        request_id = get_request_id()
        info_id("Initializing EMTAC Query Expansion Orchestrator", request_id)

        # Initialize Intent/NER plugin with your trained models
        self.intent_ner_plugin = IntentEntityPlugin(
            intent_model_dir=intent_model_dir,
            ner_model_dir=ner_model_dir,
            intent_labels=["parts", "images", "documents", "prints", "tools", "troubleshooting"],
            ner_labels=["O", "B-PARTDESC", "B-PARTNUM"]
        )

        # Initialize base query expander
        if query_expander is None:
            self.base_expander = QueryExpansionRAG()
        else:
            self.base_expander = query_expander

        # Load enhanced synonym dictionaries based on your intent labels
        self.intent_specific_synonyms = self._load_emtac_intent_synonyms()
        self.entity_expansion_rules = self._load_emtac_entity_rules()

        info_id("EMTAC Query Expansion Orchestrator initialized successfully", request_id)

    def _load_emtac_intent_synonyms(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load synonym dictionaries specific to your EMTAC intent labels:
        ["parts", "images", "documents", "prints", "tools", "troubleshooting"]
        """
        return {
            "parts": {
                # Equipment synonyms for parts search
                "pump": ["pump", "pumping station", "water pump", "centrifugal pump",
                         "booster pump", "circulation pump", "transfer pump"],
                "motor": ["motor", "drive", "actuator", "electric motor", "servo motor",
                          "variable speed drive", "VSD", "VFD"],
                "valve": ["valve", "gate valve", "control valve", "shutoff valve",
                          "ball valve", "butterfly valve", "check valve", "relief valve"],
                "bearing": ["bearing", "ball bearing", "roller bearing", "thrust bearing",
                            "journal bearing", "pillow block"],
                "seal": ["seal", "mechanical seal", "o-ring", "gasket", "packing", "lip seal"],
                "coupling": ["coupling", "flexible coupling", "rigid coupling", "jaw coupling"],
                "impeller": ["impeller", "rotor", "wheel", "fan blade"],
                "housing": ["housing", "casing", "enclosure", "shell", "body"],

                # Part identification terms
                "part": ["part", "component", "element", "piece", "item", "spare"],
                "number": ["number", "P/N", "part no", "item no", "component number"],
                "replacement": ["replacement", "spare", "substitute", "alternative", "backup"],
                "compatible": ["compatible", "equivalent", "interchangeable", "substitute"]
            },

            "troubleshooting": {
                # Problem/symptom synonyms
                "failure": ["failure", "fault", "malfunction", "breakdown", "defect", "issue"],
                "noise": ["noise", "sound", "grinding", "squealing", "rattling", "vibration"],
                "leak": ["leak", "leakage", "seepage", "drip", "spillage", "weeping"],
                "overheating": ["overheating", "hot", "thermal", "temperature", "heat"],
                "vibration": ["vibration", "oscillation", "shake", "tremor", "resonance"],
                "wear": ["wear", "erosion", "corrosion", "deterioration", "degradation"],

                # Diagnostic terms
                "problem": ["problem", "issue", "trouble", "difficulty", "concern"],
                "diagnosis": ["diagnosis", "troubleshooting", "fault finding", "root cause"],
                "repair": ["repair", "fix", "correct", "remedy", "resolve"],
                "maintenance": ["maintenance", "service", "upkeep", "care", "preservation"]
            },

            "documents": {
                # Document type synonyms
                "manual": ["manual", "guide", "handbook", "documentation", "instructions"],
                "procedure": ["procedure", "SOP", "work instruction", "protocol", "method"],
                "specification": ["specification", "spec", "requirement", "standard", "criteria"],
                "report": ["report", "analysis", "study", "assessment", "evaluation"],
                "certificate": ["certificate", "certification", "approval", "compliance"],
                "datasheet": ["datasheet", "data sheet", "technical data", "specs"],

                # Content type synonyms
                "installation": ["installation", "setup", "mounting", "assembly", "commissioning"],
                "operation": ["operation", "operating", "running", "functional", "performance"],
                "safety": ["safety", "hazard", "risk", "protection", "precaution"]
            },

            "prints": {
                # Drawing type synonyms
                "drawing": ["drawing", "print", "blueprint", "schematic", "diagram"],
                "P&ID": ["P&ID", "piping diagram", "process diagram", "flow diagram"],
                "electrical": ["electrical", "wiring", "circuit", "schematic", "panel"],
                "mechanical": ["mechanical", "assembly", "detail", "section", "view"],
                "layout": ["layout", "arrangement", "plan", "configuration", "setup"],

                # Drawing elements
                "dimension": ["dimension", "measurement", "size", "length", "width"],
                "tolerance": ["tolerance", "allowance", "clearance", "fit", "deviation"],
                "material": ["material", "grade", "specification", "type", "class"]
            },

            "images": {
                # Image type synonyms
                "photo": ["photo", "photograph", "picture", "image", "snapshot"],
                "diagram": ["diagram", "illustration", "figure", "chart", "graphic"],
                "screenshot": ["screenshot", "screen capture", "display", "interface"],

                # Image content synonyms
                "equipment": ["equipment", "machinery", "apparatus", "device", "unit"],
                "installation": ["installation", "site", "location", "setup", "configuration"],
                "condition": ["condition", "state", "status", "appearance", "wear"]
            },

            "tools": {
                # Tool type synonyms
                "wrench": ["wrench", "spanner", "key", "socket", "ratchet"],
                "screwdriver": ["screwdriver", "driver", "bit", "phillips", "flathead"],
                "meter": ["meter", "gauge", "instrument", "tester", "analyzer"],
                "multimeter": ["multimeter", "DMM", "volt meter", "ohm meter"],

                # Tool usage synonyms
                "measurement": ["measurement", "testing", "checking", "verification"],
                "calibration": ["calibration", "adjustment", "setting", "tuning"],
                "torque": ["torque", "tightening", "fastening", "clamping"]
            }
        }

    def _load_emtac_entity_rules(self) -> Dict[str, List[str]]:
        """
        Load expansion rules based on your NER entity types:
        ["O", "B-PARTDESC", "B-PARTNUM"]
        """
        return {
            "B-PARTDESC": [
                "description", "specification", "details", "characteristics",
                "features", "properties", "attributes", "parameters"
            ],
            "B-PARTNUM": [
                "part number", "P/N", "part no", "component number", "item number",
                "model number", "serial number", "catalog number", "order code"
            ]
        }

    @with_request_id
    def process_query_complete_pipeline(self, user_query: str,
                                        max_expansions: int = 10,
                                        confidence_threshold: float = 0.3) -> Dict:
        """
        Complete pipeline: User Query → Intent → NER → Enhanced Query Expansion

        Args:
            user_query: Raw user input
            max_expansions: Maximum number of expanded queries to generate
            confidence_threshold: Minimum confidence for intent classification

        Returns:
            Dict containing all pipeline results and expanded queries
        """
        request_id = get_request_id()
        info_id(f"Processing complete pipeline for: '{user_query}'", request_id)

        results = {
            "original_query": user_query,
            "intent_classification": None,
            "ner_extraction": None,
            "base_expansion": [],
            "intent_aware_expansion": [],
            "entity_enhanced_expansion": [],
            "final_expanded_queries": [],
            "pipeline_metadata": {
                "intent_confidence": 0.0,
                "entities_found": 0,
                "expansion_methods_used": []
            }
        }

        try:
            # Step 1: Intent Classification
            with log_timed_operation("Intent classification", request_id):
                intent_label, intent_confidence = self.intent_ner_plugin.classify_intent(user_query)
                results["intent_classification"] = {
                    "intent": intent_label,
                    "confidence": intent_confidence
                }
                results["pipeline_metadata"]["intent_confidence"] = intent_confidence

                if intent_label and intent_confidence > confidence_threshold:
                    info_id(f"Intent detected: {intent_label} (confidence: {intent_confidence:.3f})", request_id)
                else:
                    warning_id(f"Low confidence intent or no intent detected: {intent_label} ({intent_confidence:.3f})",
                               request_id)

            # Step 2: NER Entity Extraction
            with log_timed_operation("NER entity extraction", request_id):
                ner_entities = self.intent_ner_plugin.extract_entities(user_query)
                results["ner_extraction"] = ner_entities
                results["pipeline_metadata"]["entities_found"] = len(ner_entities)

                if ner_entities:
                    entity_summary = ", ".join([f"{e.get('word', '')}({e.get('entity_group', '')})"
                                                for e in ner_entities])
                    debug_id(f"Entities extracted: {entity_summary}", request_id)
                else:
                    debug_id("No entities extracted", request_id)

            # Step 3: Base Query Expansion (your existing system)
            with log_timed_operation("Base query expansion", request_id):
                try:
                    base_queries = self.base_expander.multi_query_expansion_rules(user_query)
                    results["base_expansion"] = base_queries[:max_expansions]
                    results["pipeline_metadata"]["expansion_methods_used"].append("base_rules")
                    debug_id(f"Base expansion generated {len(base_queries)} queries", request_id)
                except Exception as e:
                    warning_id(f"Base expansion failed: {e}", request_id)
                    results["base_expansion"] = [user_query]

            # Step 4: Intent-Aware Expansion
            if intent_label and intent_confidence > confidence_threshold:
                with log_timed_operation("Intent-aware expansion", request_id):
                    intent_queries = self._expand_by_intent(user_query, intent_label, ner_entities)
                    results["intent_aware_expansion"] = intent_queries[:max_expansions]
                    results["pipeline_metadata"]["expansion_methods_used"].append("intent_aware")
                    debug_id(f"Intent-aware expansion generated {len(intent_queries)} queries", request_id)

            # Step 5: Entity-Enhanced Expansion
            if ner_entities:
                with log_timed_operation("Entity-enhanced expansion", request_id):
                    entity_queries = self._expand_by_entities(user_query, ner_entities)
                    results["entity_enhanced_expansion"] = entity_queries[:max_expansions]
                    results["pipeline_metadata"]["expansion_methods_used"].append("entity_enhanced")
                    debug_id(f"Entity-enhanced expansion generated {len(entity_queries)} queries", request_id)

            # Step 6: Combine and Deduplicate All Expansions
            with log_timed_operation("Final query combination", request_id):
                all_queries = []

                # Start with original query
                all_queries.append(user_query)

                # Add base expansions
                all_queries.extend(results["base_expansion"])

                # Add intent-aware expansions
                all_queries.extend(results["intent_aware_expansion"])

                # Add entity-enhanced expansions
                all_queries.extend(results["entity_enhanced_expansion"])

                # Remove duplicates while preserving order
                seen = set()
                unique_queries = []
                for query in all_queries:
                    if query.lower() not in seen:
                        seen.add(query.lower())
                        unique_queries.append(query)

                # Limit to max_expansions
                results["final_expanded_queries"] = unique_queries[:max_expansions]

                info_id(f"Final expansion: {len(results['final_expanded_queries'])} unique queries", request_id)

        except Exception as e:
            error_id(f"Pipeline error: {e}", request_id)
            results["final_expanded_queries"] = [user_query]  # Fallback

        return results

    @with_request_id
    def _expand_by_intent(self, query: str, intent: str, entities: List[Dict]) -> List[str]:
        """
        Expand query based on detected intent using intent-specific synonyms
        """
        request_id = get_request_id()
        debug_id(f"Expanding by intent: {intent}", request_id)

        expanded_queries = [query]

        # Get intent-specific synonyms
        intent_synonyms = self.intent_specific_synonyms.get(intent, {})

        if not intent_synonyms:
            debug_id(f"No specific synonyms found for intent: {intent}", request_id)
            return expanded_queries

        # Tokenize query for word-level replacement
        words = re.findall(r'\b\w+\b', query.lower())

        # Apply intent-specific synonyms
        for word in words:
            if word in intent_synonyms:
                for synonym in intent_synonyms[word]:
                    if synonym.lower() != word:
                        # Create new query with synonym replacement
                        new_query = re.sub(r'\b' + re.escape(word) + r'\b',
                                           synonym, query, flags=re.IGNORECASE)
                        expanded_queries.append(new_query)

        # Add intent-specific query patterns
        expanded_queries.extend(self._generate_intent_patterns(query, intent, entities))

        # Remove duplicates
        return list(dict.fromkeys(expanded_queries))

    def _expand_by_entities(self, query: str, entities: List[Dict]) -> List[str]:
        """
        Expand query based on detected entities
        """
        request_id = get_request_id()
        debug_id(f"Expanding by entities: {len(entities)} entities", request_id)

        expanded_queries = [query]

        for entity in entities:
            entity_text = entity.get('word', '')
            entity_type = entity.get('entity_group', '')

            # Apply entity-type specific expansions
            if entity_type in self.entity_expansion_rules:
                for expansion_term in self.entity_expansion_rules[entity_type]:
                    # Add query variations with entity expansion terms
                    expanded_queries.extend([
                        f"{query} {expansion_term}",
                        f"{expansion_term} {entity_text}",
                        f"{entity_text} {expansion_term}"
                    ])

            # Entity-specific patterns
            if entity_type == "B-PARTNUM":
                expanded_queries.extend([
                    f"part {entity_text}",
                    f"P/N {entity_text}",
                    f"component {entity_text}",
                    f"order {entity_text}",
                    f"buy {entity_text}"
                ])

            elif entity_type == "B-PARTDESC":
                expanded_queries.extend([
                    f"{entity_text} specifications",
                    f"{entity_text} manual",
                    f"{entity_text} parts",
                    f"{entity_text} maintenance",
                    f"replace {entity_text}"
                ])

        return list(dict.fromkeys(expanded_queries))

    def _generate_intent_patterns(self, query: str, intent: str, entities: List[Dict]) -> List[str]:
        """
        Generate intent-specific query patterns
        """
        patterns = []

        if intent == "parts":
            patterns.extend([
                f"{query} part number",
                f"{query} spare parts",
                f"{query} replacement",
                f"order {query}",
                f"buy {query}",
                f"purchase {query}"
            ])

        elif intent == "troubleshooting":
            patterns.extend([
                f"{query} problems",
                f"{query} issues",
                f"{query} troubleshooting",
                f"{query} fault",
                f"fix {query}",
                f"repair {query}",
                f"{query} not working"
            ])

        elif intent == "documents":
            patterns.extend([
                f"{query} manual",
                f"{query} documentation",
                f"{query} guide",
                f"{query} procedure",
                f"{query} instructions"
            ])

        elif intent == "prints":
            patterns.extend([
                f"{query} drawing",
                f"{query} schematic",
                f"{query} blueprint",
                f"{query} diagram",
                f"{query} P&ID"
            ])

        elif intent == "images":
            patterns.extend([
                f"{query} photo",
                f"{query} picture",
                f"{query} image",
                f"{query} diagram"
            ])

        elif intent == "tools":
            patterns.extend([
                f"{query} tool",
                f"{query} equipment",
                f"use {query}",
                f"{query} procedure"
            ])

        return patterns

    def get_pipeline_status(self) -> Dict:
        """
        Get status of all pipeline components
        """
        return {
            "intent_classifier_loaded": self.intent_ner_plugin.intent_classifier is not None,
            "ner_model_loaded": self.intent_ner_plugin.ner is not None,
            "base_expander_available": self.base_expander is not None,
            "intent_labels": self.intent_ner_plugin.intent_labels,
            "ner_labels": self.intent_ner_plugin.ner_labels,
            "synonym_intents_loaded": list(self.intent_specific_synonyms.keys()),
            "entity_rules_loaded": list(self.entity_expansion_rules.keys())
        }


# Usage example and integration
def create_emtac_orchestrator(intent_model_path, ner_model_path):
    """
    Factory function to create the complete EMTAC query expansion system
    """
    try:
        # Initialize the orchestrator with your trained models
        orchestrator = EMTACQueryExpansionOrchestrator(
            intent_model_dir=intent_model_path,
            ner_model_dir=ner_model_path
        )

        # Check pipeline status
        status = orchestrator.get_pipeline_status()
        print("Pipeline Status:")
        for component, loaded in status.items():
            print(f"  {component}: {'✓' if loaded else '✗'}")

        return orchestrator

    except Exception as e:
        print(f"Error creating orchestrator: {e}")
        return None


# Example usage with your models
def example_usage():
    """
    Example of how to use the complete pipeline
    """
    # Paths to your trained models
    intent_model_path = "/path/to/your/trained/intent/model"
    ner_model_path = "/path/to/your/trained/ner/model"

    # Create orchestrator
    orchestrator = create_emtac_orchestrator(intent_model_path, ner_model_path)

    if orchestrator:
        # Test queries
        test_queries = [
            "pump bearing replacement",
            "centrifugal pump P&ID drawing",
            "motor vibration troubleshooting",
            "valve installation manual",
            "pump photos"
        ]

        for query in test_queries:
            print(f"\n{'=' * 50}")
            print(f"Testing query: '{query}'")
            print('=' * 50)

            result = orchestrator.process_query_complete_pipeline(query)

            print(f"Intent: {result['intent_classification']['intent']} "
                  f"(confidence: {result['intent_classification']['confidence']:.3f})")

            if result['ner_extraction']:
                print("Entities:", [f"{e['word']}({e['entity_group']})"
                                    for e in result['ner_extraction']])

            print(f"Expanded queries ({len(result['final_expanded_queries'])}):")
            for i, expanded_query in enumerate(result['final_expanded_queries'][:8], 1):
                print(f"  {i}. {expanded_query}")


if __name__ == "__main__":
    example_usage()