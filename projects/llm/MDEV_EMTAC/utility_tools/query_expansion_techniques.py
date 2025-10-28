import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import time

# Import your logging system
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    get_request_id, set_request_id, with_request_id,
    log_timed_operation
)

# Import your AI models system
from plugins.ai_modules import (
    ModelsConfig,
    NoAIModel, OpenAIModel, AnthropicModel, GPT4AllModel, TinyLlamaModel,
    NoEmbeddingModel, OpenAIEmbeddingModel, GPT4AllEmbeddingModel, TinyLlamaEmbeddingModel
)

# Download required NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

class QueryExpansionRAG:
    """
    Comprehensive Query Expansion for RAG systems with multiple AI model support:
    1. Multi-Query Expansion (AI-based and rule-based)
    2. HyDE (Hypothetical Document Embeddings) - AI only
    3. Pseudo-Relevance Feedback (Classic and AI-augmented)

    Supports multiple AI models: OpenAI, Anthropic, GPT4All, TinyLlama
    """

    def __init__(self,
                 ai_model_name: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 use_spacy: bool = True):
        """
        Initialize the Query Expansion RAG system with your AI framework

        Args:
            ai_model_name: Name of AI model to use (None = use current from config)
            embedding_model_name: Name of embedding model to use (None = use current from config)
            use_spacy: Whether to use spaCy for NER (default True)
        """
        request_id = get_request_id()
        info_id("Initializing QueryExpansionRAG system with AI framework integration", request_id)

        # Load AI models using your framework
        with log_timed_operation("Loading AI model", request_id):
            try:
                self.ai_model = ModelsConfig.load_ai_model(ai_model_name)
                ai_name = ai_model_name or ModelsConfig.get_config_value('ai', 'CURRENT_MODEL', 'NoAIModel')
                info_id(f"Loaded AI model: {ai_name}", request_id)

                # Check if AI model supports text generation
                self.llm_available = not isinstance(self.ai_model, NoAIModel)
                if self.llm_available:
                    info_id("LLM-based query expansion methods available", request_id)
                else:
                    warning_id("AI model disabled - only rule-based methods available", request_id)

            except Exception as e:
                error_id(f"Failed to load AI model: {e}", request_id)
                self.ai_model = NoAIModel()
                self.llm_available = False

        # Load embedding model using your framework
        with log_timed_operation("Loading embedding model", request_id):
            try:
                self.embedding_model = ModelsConfig.load_embedding_model(embedding_model_name)
                embedding_name = embedding_model_name or ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')
                info_id(f"Loaded embedding model: {embedding_name}", request_id)

                self.embeddings_available = not isinstance(self.embedding_model, NoEmbeddingModel)
                if self.embeddings_available:
                    info_id("Embedding generation available", request_id)
                else:
                    warning_id("Embedding model disabled - HyDE embeddings not available", request_id)

            except Exception as e:
                error_id(f"Failed to load embedding model: {e}", request_id)
                self.embedding_model = NoEmbeddingModel()
                self.embeddings_available = False

        # Load spaCy model for NER with error handling
        if use_spacy:
            try:
                with log_timed_operation("Loading spaCy model", request_id):
                    self.nlp = spacy.load("en_core_web_sm")
                    info_id("Loaded spaCy model for NER", request_id)
            except OSError:
                warning_id("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm", request_id)
                self.nlp = None
        else:
            self.nlp = None
            debug_id("spaCy disabled by user", request_id)

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        # Load domain-specific synonym dictionary (customize for your domain)
        self.synonym_dict = {
            "pump": ["pump", "pumping station", "water pump", "centrifugal pump"],
            "schematic": ["schematic", "diagram", "drawing", "blueprint", "layout"],
            "wiring": ["wiring", "electrical", "circuit", "connection"],
            "valve": ["valve", "gate valve", "control valve", "shutoff valve"],
            "motor": ["motor", "drive", "actuator", "engine"],
            "maintenance": ["maintenance", "service", "repair", "upkeep"],
            "troubleshooting": ["troubleshooting", "diagnostics", "fault finding", "problem solving"],
            "manual": ["manual", "guide", "handbook", "documentation"],
            "specification": ["specification", "spec", "requirement", "standard"],
            "installation": ["installation", "setup", "mounting", "assembly"],
        }

        # Domain-specific acronym expansion
        self.acronym_dict = {
            "HVAC": "Heating Ventilation Air Conditioning",
            "P&ID": "Piping and Instrumentation Diagram",
            "VFD": "Variable Frequency Drive",
            "PLC": "Programmable Logic Controller",
            "HMI": "Human Machine Interface",
            "SCADA": "Supervisory Control and Data Acquisition",
            "I&C": "Instrumentation and Control",
            "O&M": "Operations and Maintenance",
            "SOP": "Standard Operating Procedure",
            "ISO": "International Organization for Standardization",
            "API": "American Petroleum Institute",
            "ANSI": "American National Standards Institute",
            "IEC": "International Electrotechnical Commission",
            "IEEE": "Institute of Electrical and Electronics Engineers",
        }

        info_id(f"QueryExpansionRAG initialized with {len(self.synonym_dict)} synonym groups and {len(self.acronym_dict)} acronyms", request_id)
        info_id(f"Available methods: LLM={self.llm_available}, Embeddings={self.embeddings_available}, NER={self.nlp is not None}", request_id)

    # =============================================================================
    # 1. MULTI-QUERY EXPANSION
    # =============================================================================

    @with_request_id
    def multi_query_expansion_ai(self, query: str, num_variants: int = 4) -> List[str]:
        """
        AI-based multi-query expansion using your AI framework

        Args:
            query: Original user query
            num_variants: Number of query variants to generate

        Returns:
            List of expanded queries including the original
        """
        request_id = get_request_id()

        if not self.llm_available:
            error_id("AI model not available for query expansion", request_id)
            warning_id("Falling back to rule-based expansion", request_id)
            return self.multi_query_expansion_rules(query)

        info_id(f"Starting AI multi-query expansion for: '{query}' (generating {num_variants} variants)", request_id)

        # Get the AI model type for optimized prompts
        ai_model_type = type(self.ai_model).__name__
        debug_id(f"Using AI model: {ai_model_type}", request_id)

        # Customize prompt based on AI model capabilities
        if isinstance(self.ai_model, (OpenAIModel, AnthropicModel)):
            # Advanced prompt for high-capability models
            prompt = f"""Given the following search query, generate {num_variants} alternative ways to express the same information need.

Original query: "{query}"

Please generate {num_variants} alternative queries that:
- Use synonyms and technical terminology variations
- Include different levels of specificity
- Consider alternate phrasings and industry terminology
- Maintain the same intent and information need

Format: Return each alternative query on a separate line, numbered 1-{num_variants}."""

        else:
            # Simpler prompt for local models (GPT4All, TinyLlama)
            prompt = f"""Rewrite this search query in {num_variants} different ways using synonyms and different words:

Query: "{query}"

Write {num_variants} alternatives:"""

        try:
            with log_timed_operation("AI query expansion generation", request_id):
                response = self.ai_model.get_response(prompt)

            if not response or response.startswith("Error") or "error occurred" in response.lower():
                error_id(f"AI model returned error response: {response}", request_id)
                warning_id("Falling back to rule-based expansion", request_id)
                return self.multi_query_expansion_rules(query)

            # Parse the response to extract variants
            variants = self._parse_ai_response(response, num_variants)

            # Add original query and remove duplicates
            all_queries = [query] + variants
            final_queries = list(dict.fromkeys(all_queries))  # Remove duplicates while preserving order

            info_id(f"Generated {len(final_queries)} unique queries (including original)", request_id)
            debug_id(f"Expanded queries: {final_queries}", request_id)

            return final_queries

        except Exception as e:
            error_id(f"Error in AI query expansion: {e}", request_id)
            warning_id("Falling back to rule-based expansion", request_id)
            return self.multi_query_expansion_rules(query)

    def _parse_ai_response(self, response: str, expected_count: int) -> List[str]:
        """Parse AI response to extract query variants"""
        request_id = get_request_id()

        # Split response into lines and clean up
        lines = response.strip().split('\n')
        variants = []

        for line in lines:
            # Clean up the line
            line = line.strip()
            if not line:
                continue

            # Remove numbering, bullets, etc.
            line = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove "1. " or "1) "
            line = re.sub(r'^[-•*]\s*', '', line)      # Remove bullets
            line = re.sub(r'^["\'""]|["\'""]$', '', line)  # Remove quotes
            line = line.strip()

            if line and len(line) > 3:  # Avoid very short responses
                variants.append(line)

            # Stop if we have enough variants
            if len(variants) >= expected_count:
                break

        debug_id(f"Parsed {len(variants)} variants from AI response", request_id)
        return variants

    @with_request_id
    def multi_query_expansion_rules(self, query: str) -> List[str]:
        """
        Rule-based multi-query expansion using synonym and acronym dictionaries

        Args:
            query: Original user query

        Returns:
            List of expanded queries
        """
        request_id = get_request_id()
        info_id(f"Starting rule-based multi-query expansion for: '{query}'", request_id)

        expanded_queries = [query]
        query_lower = query.lower()

        # Expand acronyms
        with log_timed_operation("Acronym expansion", request_id):
            acronym_expansions = 0
            for acronym, expansion in self.acronym_dict.items():
                if acronym.lower() in query_lower:
                    new_query = query.replace(acronym, expansion)
                    expanded_queries.append(new_query)
                    acronym_expansions += 1
                    debug_id(f"Expanded acronym '{acronym}' to '{expansion}'", request_id)

            if acronym_expansions > 0:
                info_id(f"Applied {acronym_expansions} acronym expansions", request_id)

        # Synonym expansion
        with log_timed_operation("Synonym expansion", request_id):
            words = word_tokenize(query_lower)
            synonym_expansions = 0

            for word in words:
                if word in self.synonym_dict:
                    for synonym in self.synonym_dict[word]:
                        if synonym != word:  # Don't replace with the same word
                            new_query = re.sub(r'\b' + word + r'\b', synonym, query, flags=re.IGNORECASE)
                            if new_query != query:
                                expanded_queries.append(new_query)
                                synonym_expansions += 1
                                debug_id(f"Replaced '{word}' with '{synonym}'", request_id)

            if synonym_expansions > 0:
                info_id(f"Applied {synonym_expansions} synonym expansions", request_id)

        # Generate combinations of synonyms (limited to avoid explosion)
        if len(words) <= 3:  # Only for short queries
            with log_timed_operation("Synonym combinations", request_id):
                combinations = []
                combination_count = 0

                for word in words:
                    if word in self.synonym_dict and len(self.synonym_dict[word]) > 1:
                        for synonym in self.synonym_dict[word][:2]:  # Limit to 2 synonyms
                            combo_query = re.sub(r'\b' + word + r'\b', synonym, query, flags=re.IGNORECASE)
                            combinations.append(combo_query)
                            combination_count += 1

                expanded_queries.extend(combinations)
                if combination_count > 0:
                    debug_id(f"Generated {combination_count} synonym combinations", request_id)

        final_queries = list(dict.fromkeys(expanded_queries))
        info_id(f"Rule-based expansion generated {len(final_queries)} unique queries (including original)", request_id)
        debug_id(f"Final expanded queries: {final_queries}", request_id)

        return final_queries

    # =============================================================================
    # 2. HYDE (HYPOTHETICAL DOCUMENT EMBEDDINGS)
    # =============================================================================

    @with_request_id
    def hyde_generate_hypothetical_doc(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query using your AI framework

        Args:
            query: User query

        Returns:
            Hypothetical document text
        """
        request_id = get_request_id()

        if not self.llm_available:
            error_id("AI model not available for HyDE", request_id)
            fallback_doc = f"Technical document about {query}"
            warning_id(f"Using fallback document: '{fallback_doc}'", request_id)
            return fallback_doc

        info_id(f"Generating hypothetical document for query: '{query}'", request_id)

        # Get the AI model type for optimized prompts
        ai_model_type = type(self.ai_model).__name__
        debug_id(f"Using AI model for HyDE: {ai_model_type}", request_id)

        # Customize prompt based on AI model capabilities
        if isinstance(self.ai_model, (OpenAIModel, AnthropicModel)):
            # Detailed prompt for high-capability models
            prompt = f"""Write a comprehensive technical document that would perfectly answer this query: "{query}"

The document should be:
- Factual and technically accurate
- Include specific details, measurements, and procedures where relevant
- Use appropriate technical terminology for the domain
- Be 200-400 words in length
- Written in the style of a technical manual or engineering document
- Include practical information that someone would actually search for

Document:"""

        else:
            # Simpler prompt for local models
            prompt = f"""Write a technical document about: {query}

Include details about procedures, specifications, and technical information. Write 150-250 words.

Document:"""

        try:
            with log_timed_operation("HyDE document generation", request_id):
                response = self.ai_model.get_response(prompt)

            if not response or response.startswith("Error") or "error occurred" in response.lower():
                error_id(f"AI model returned error for HyDE: {response}", request_id)
                fallback_doc = f"Technical document about {query}"
                warning_id(f"Using fallback document: '{fallback_doc}'", request_id)
                return fallback_doc

            # Clean up the response
            hypothetical_doc = response.strip()

            # Remove any introductory phrases that AI models might add
            cleanup_patterns = [
                r'^(Here is|Here\'s|This is|Below is).*?:\s*',
                r'^Document:\s*',
                r'^Technical Document:\s*',
            ]

            for pattern in cleanup_patterns:
                hypothetical_doc = re.sub(pattern, '', hypothetical_doc, flags=re.IGNORECASE)

            hypothetical_doc = hypothetical_doc.strip()

            info_id(f"Generated hypothetical document ({len(hypothetical_doc)} characters)", request_id)
            debug_id(f"Hypothetical document preview: {hypothetical_doc[:200]}...", request_id)

            return hypothetical_doc

        except Exception as e:
            error_id(f"Error in HyDE generation: {e}", request_id)
            fallback_doc = f"Technical document about {query}"
            warning_id(f"Using fallback document: '{fallback_doc}'", request_id)
            return fallback_doc

    @with_request_id
    def hyde_embed_hypothetical(self, hypothetical_doc: str) -> np.ndarray:
        """
        Create embedding for the hypothetical document using your embedding framework

        Args:
            hypothetical_doc: Generated hypothetical document

        Returns:
            Document embedding vector
        """
        request_id = get_request_id()

        if not self.embeddings_available:
            error_id("Embedding model not available for HyDE", request_id)
            return np.array([])

        with log_timed_operation("HyDE embedding generation", request_id):
            try:
                embeddings = self.embedding_model.get_embeddings(hypothetical_doc)

                if embeddings and len(embeddings) > 0:
                    embedding_array = np.array(embeddings)
                    info_id(f"Generated HyDE embedding vector of shape {embedding_array.shape}", request_id)
                    return embedding_array
                else:
                    error_id("Embedding model returned empty embeddings", request_id)
                    return np.array([])

            except Exception as e:
                error_id(f"Error generating HyDE embedding: {e}", request_id)
                return np.array([])

    # =============================================================================
    # 3. PSEUDO-RELEVANCE FEEDBACK (PRF)
    # =============================================================================

    @with_request_id
    def prf_classic(self, query: str, top_docs: List[str], top_k_terms: int = 5) -> List[str]:
        """
        Classic Pseudo-Relevance Feedback without AI
        Extract important terms from top retrieved documents

        Args:
            query: Original query
            top_docs: List of top retrieved document texts
            top_k_terms: Number of terms to extract

        Returns:
            List of expanded queries with additional terms
        """
        request_id = get_request_id()

        if not top_docs:
            warning_id("No top documents provided for PRF - returning original query", request_id)
            return [query]

        info_id(f"Starting classic PRF for query: '{query}' with {len(top_docs)} documents", request_id)

        # Combine top documents
        combined_docs = " ".join(top_docs)
        debug_id(f"Combined documents length: {len(combined_docs)} characters", request_id)

        # Extract entities using spaCy if available
        additional_terms = []

        if self.nlp:
            with log_timed_operation("NER entity extraction", request_id):
                try:
                    doc = self.nlp(combined_docs)
                    entities = [ent.text.lower() for ent in doc.ents
                                if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'GPE', 'PERSON']]
                    entities = list(dict.fromkeys(entities))  # Remove duplicates
                    additional_terms.extend(entities[:top_k_terms // 2])
                    info_id(f"Extracted {len(entities)} unique entities via NER", request_id)
                    debug_id(f"Top entities: {entities[:5]}", request_id)
                except Exception as e:
                    warning_id(f"NER processing failed: {e}", request_id)
        else:
            debug_id("spaCy not available - skipping NER extraction", request_id)

        # Extract important terms using TF-IDF
        with log_timed_operation("TF-IDF term extraction", request_id):
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )

            try:
                tfidf_matrix = vectorizer.fit_transform([combined_docs])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]

                # Get top terms by TF-IDF score
                top_indices = np.argsort(tfidf_scores)[-top_k_terms:]
                top_tfidf_terms = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
                additional_terms.extend(top_tfidf_terms)

                info_id(f"Extracted {len(top_tfidf_terms)} terms via TF-IDF", request_id)
                debug_id(f"Top TF-IDF terms: {top_tfidf_terms}", request_id)

            except Exception as e:
                error_id(f"TF-IDF extraction error: {e}", request_id)

        # Create expanded queries
        expanded_queries = [query]
        query_terms = set(word_tokenize(query.lower()))

        added_terms = 0
        for term in additional_terms:
            if term and len(term) > 2 and term not in query_terms:
                expanded_query = f"{query} {term}"
                expanded_queries.append(expanded_query)
                added_terms += 1
                debug_id(f"Added term '{term}' to create: '{expanded_query}'", request_id)

        final_queries = list(dict.fromkeys(expanded_queries))
        info_id(f"Classic PRF generated {len(final_queries)} queries with {added_terms} additional terms", request_id)

        return final_queries

    @with_request_id
    def prf_ai_augmented(self, query: str, top_docs: List[str]) -> List[str]:
        """
        AI-augmented Pseudo-Relevance Feedback using your AI framework
        Use AI to intelligently extract relevant terms from top documents

        Args:
            query: Original query
            top_docs: List of top retrieved document texts

        Returns:
            List of expanded queries
        """
        request_id = get_request_id()

        if not self.llm_available:
            warning_id("AI model not available for AI-augmented PRF, falling back to classic PRF", request_id)
            return self.prf_classic(query, top_docs)

        if not top_docs:
            warning_id("No top documents provided for PRF", request_id)
            return [query]

        info_id(f"Starting AI-augmented PRF for query: '{query}' with {len(top_docs)} documents", request_id)

        # Truncate documents to avoid token limits
        combined_docs = " ".join(top_docs)[:3000]
        debug_id(f"Using {len(combined_docs)} characters from combined documents", request_id)

        # Get the AI model type for optimized prompts
        ai_model_type = type(self.ai_model).__name__
        debug_id(f"Using AI model for PRF: {ai_model_type}", request_id)

        # Customize prompt based on AI model capabilities
        if isinstance(self.ai_model, (OpenAIModel, AnthropicModel)):
            # Advanced prompt for high-capability models
            prompt = f"""Original search query: "{query}"

Retrieved documents:
{combined_docs}

Based on these retrieved documents, identify 3-5 additional keywords, technical terms, or phrases that would help improve the search for information related to the original query.

Focus on:
- Technical terminology and jargon
- Specific components, parts, or equipment names
- Related concepts and processes
- Alternative naming conventions or synonyms
- Industry-specific terms

Please provide only the additional search terms, one per line, without explanations:"""

        else:
            # Simpler prompt for local models
            prompt = f"""Query: "{query}"

Documents: {combined_docs[:1500]}

Find 3-5 important keywords from these documents that relate to the query. List them:"""

        try:
            with log_timed_operation("AI-augmented PRF analysis", request_id):
                response = self.ai_model.get_response(prompt)

            if not response or response.startswith("Error") or "error occurred" in response.lower():
                error_id(f"AI model returned error for PRF: {response}", request_id)
                warning_id("Falling back to classic PRF", request_id)
                return self.prf_classic(query, top_docs)

            # Parse the response to extract terms
            additional_terms = self._parse_prf_terms(response)

            # Create expanded queries
            expanded_queries = [query]
            added_terms = 0

            for term in additional_terms:
                if term and len(term) > 2:
                    expanded_query = f"{query} {term}"
                    expanded_queries.append(expanded_query)
                    added_terms += 1
                    debug_id(f"AI suggested term '{term}' creating query: '{expanded_query}'", request_id)

            final_queries = list(dict.fromkeys(expanded_queries))
            info_id(f"AI-augmented PRF generated {len(final_queries)} queries with {added_terms} AI-suggested terms", request_id)
            debug_id(f"AI suggested terms: {additional_terms}", request_id)

            return final_queries

        except Exception as e:
            error_id(f"Error in AI-augmented PRF: {e}", request_id)
            warning_id("Falling back to classic PRF", request_id)
            return self.prf_classic(query, top_docs)

    def _parse_prf_terms(self, response: str) -> List[str]:
        """Parse AI response to extract PRF terms"""
        request_id = get_request_id()

        lines = response.strip().split('\n')
        terms = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering, bullets, etc.
            line = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove "1. " or "1) "
            line = re.sub(r'^[-•*]\s*', '', line)      # Remove bullets
            line = re.sub(r'^["\'""]|["\'""]$', '', line)  # Remove quotes
            line = line.strip()

            # Skip lines that look like explanations
            if any(phrase in line.lower() for phrase in ['based on', 'these terms', 'keywords', 'search terms']):
                continue

            if line and len(line) > 2 and len(line) < 50:  # Reasonable term length
                terms.append(line)

        debug_id(f"Parsed {len(terms)} terms from AI PRF response", request_id)
        return terms

    # =============================================================================
    # MAIN INTERFACE METHODS
    # =============================================================================

    @with_request_id
    def expand_query(self,
                     query: str,
                     method: str = "multi_query_ai",
                     top_docs: Optional[List[str]] = None,
                     **kwargs) -> List[str]:
        """
        Main method to expand queries using specified technique

        Args:
            query: Original user query
            method: Expansion method ('multi_query_ai', 'multi_query_rules', 'hyde', 'prf_classic', 'prf_ai')
            top_docs: Required for PRF methods
            **kwargs: Additional method-specific parameters

        Returns:
            List of expanded queries
        """
        request_id = get_request_id()
        info_id(f"Expanding query '{query}' using method '{method}'", request_id)

        try:
            if method == "multi_query_ai":
                return self.multi_query_expansion_ai(query, kwargs.get('num_variants', 4))
            elif method == "multi_query_rules":
                return self.multi_query_expansion_rules(query)
            elif method == "hyde":
                # For HyDE, return the hypothetical document as a "query"
                hyp_doc = self.hyde_generate_hypothetical_doc(query)
                return [query, hyp_doc]
            elif method == "prf_classic":
                if not top_docs:
                    error_id("top_docs required for PRF methods", request_id)
                    raise ValueError("top_docs required for PRF methods")
                return self.prf_classic(query, top_docs, kwargs.get('top_k_terms', 5))
            elif method == "prf_ai":
                if not top_docs:
                    error_id("top_docs required for PRF methods", request_id)
                    raise ValueError("top_docs required for PRF methods")
                return self.prf_ai_augmented(query, top_docs)
            else:
                error_id(f"Unknown expansion method: {method}", request_id)
                raise ValueError(f"Unknown method: {method}")

        except Exception as e:
            error_id(f"Query expansion failed for method '{method}': {e}", request_id)
            warning_id("Returning original query as fallback", request_id)
            return [query]

    @with_request_id
    def comprehensive_expansion(self,
                                query: str,
                                top_docs: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Apply all available expansion techniques based on loaded models

        Args:
            query: Original user query
            top_docs: Top retrieved documents for PRF

        Returns:
            Dictionary with results from all available methods
        """
        request_id = get_request_id()
        info_id(f"Starting comprehensive expansion for query: '{query}'", request_id)

        results = {
            "original_query": [query]
        }

        # Rule-based methods (always available)
        with log_timed_operation("Rule-based multi-query expansion", request_id):
            try:
                results["rule_based"] = self.multi_query_expansion_rules(query)
                info_id(f"Rule-based expansion: {len(results['rule_based'])} queries", request_id)
            except Exception as e:
                error_id(f"Rule-based expansion failed: {e}", request_id)
                results["rule_based"] = [query]

        # AI-based methods (if AI model available)
        if self.llm_available:
            ai_model_name = type(self.ai_model).__name__
            info_id(f"AI model available ({ai_model_name}) - running AI-based expansions", request_id)

            # AI Multi-Query
            try:
                with log_timed_operation("AI multi-query expansion", request_id):
                    results["ai_multi_query"] = self.multi_query_expansion_ai(query)
                    info_id(f"AI multi-query: {len(results['ai_multi_query'])} queries", request_id)
            except Exception as e:
                error_id(f"AI multi-query expansion failed: {e}", request_id)
                results["ai_multi_query"] = [query]

            # HyDE (requires both AI and embedding models)
            if self.embeddings_available:
                try:
                    with log_timed_operation("HyDE expansion", request_id):
                        hyp_doc = self.hyde_generate_hypothetical_doc(query)
                        results["hyde"] = [query, hyp_doc]
                        info_id("HyDE expansion completed", request_id)
                except Exception as e:
                    error_id(f"HyDE expansion failed: {e}", request_id)
                    results["hyde"] = [query]
            else:
                warning_id("Embedding model not available - skipping HyDE", request_id)
                results["hyde"] = [query]

            # AI-augmented PRF
            if top_docs:
                try:
                    with log_timed_operation("AI-augmented PRF", request_id):
                        results["prf_ai"] = self.prf_ai_augmented(query, top_docs)
                        info_id(f"AI PRF: {len(results['prf_ai'])} queries", request_id)
                except Exception as e:
                    error_id(f"AI-augmented PRF failed: {e}", request_id)
                    results["prf_ai"] = [query]
        else:
            warning_id("No AI model available - skipping AI-based methods", request_id)

        # Classic PRF (no AI required)
        if top_docs:
            try:
                with log_timed_operation("Classic PRF", request_id):
                    results["prf_classic"] = self.prf_classic(query, top_docs)
                    info_id(f"Classic PRF: {len(results['prf_classic'])} queries", request_id)
            except Exception as e:
                error_id(f"Classic PRF failed: {e}", request_id)
                results["prf_classic"] = [query]
        else:
            debug_id("No top documents provided - skipping PRF methods", request_id)

        # Summary
        total_methods = len([k for k in results.keys() if k != "original_query"])
        total_unique_queries = len(set([q for queries in results.values() for q in queries]))

        info_id(f"Comprehensive expansion completed: {total_methods} methods, {total_unique_queries} unique queries total", request_id)

        return results

    # =============================================================================
    # UTILITY AND DIAGNOSTIC METHODS
    # =============================================================================

    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive status of the query expansion system"""
        request_id = get_request_id()

        # Get AI model info
        ai_model_name = type(self.ai_model).__name__
        ai_status = {
            "name": ai_model_name,
            "available": self.llm_available,
            "supports_text_generation": self.llm_available
        }

        # Get embedding model info
        embedding_model_name = type(self.embedding_model).__name__
        embedding_status = {
            "name": embedding_model_name,
            "available": self.embeddings_available,
            "supports_embeddings": self.embeddings_available
        }

        # Get available methods based on loaded models
        available_methods = {
            "multi_query_rules": True,  # Always available
            "multi_query_ai": self.llm_available,
            "hyde": self.llm_available and self.embeddings_available,
            "prf_classic": True,  # Always available
            "prf_ai": self.llm_available
        }

        status = {
            "ai_model": ai_status,
            "embedding_model": embedding_status,
            "spacy_available": self.nlp is not None,
            "available_methods": available_methods,
            "synonym_groups": len(self.synonym_dict),
            "acronym_expansions": len(self.acronym_dict),
            "system_ready": self.llm_available or True  # System works with just rules
        }

        debug_id(f"System status: AI={self.llm_available}, Embeddings={self.embeddings_available}, NER={self.nlp is not None}", request_id)

        return status

    def test_system(self) -> Dict[str, any]:
        """Test all available system components"""
        request_id = get_request_id()
        info_id("Starting comprehensive system test", request_id)

        test_query = "pump maintenance procedure"
        test_docs = [
            "Regular maintenance of centrifugal pumps includes checking seals and bearings.",
            "Pump troubleshooting guide covers common issues like cavitation and vibration.",
            "Preventive maintenance schedules help avoid unexpected pump failures."
        ]

        test_results = {
            "test_query": test_query,
            "results": {},
            "overall_status": "unknown"
        }

        # Test rule-based expansion
        try:
            with log_timed_operation("Testing rule-based expansion", request_id):
                rule_result = self.multi_query_expansion_rules(test_query)
                test_results["results"]["rule_based"] = {
                    "success": True,
                    "query_count": len(rule_result),
                    "sample_queries": rule_result[:3]
                }
        except Exception as e:
            test_results["results"]["rule_based"] = {
                "success": False,
                "error": str(e)
            }

        # Test AI expansion if available
        if self.llm_available:
            try:
                with log_timed_operation("Testing AI expansion", request_id):
                    ai_result = self.multi_query_expansion_ai(test_query, num_variants=2)
                    test_results["results"]["ai_expansion"] = {
                        "success": True,
                        "query_count": len(ai_result),
                        "sample_queries": ai_result[:3]
                    }
            except Exception as e:
                test_results["results"]["ai_expansion"] = {
                    "success": False,
                    "error": str(e)
                }

        # Test HyDE if available
        if self.llm_available and self.embeddings_available:
            try:
                with log_timed_operation("Testing HyDE", request_id):
                    hyp_doc = self.hyde_generate_hypothetical_doc(test_query)
                    test_results["results"]["hyde"] = {
                        "success": True,
                        "doc_length": len(hyp_doc),
                        "doc_preview": hyp_doc[:100] + "..."
                    }
            except Exception as e:
                test_results["results"]["hyde"] = {
                    "success": False,
                    "error": str(e)
                }

        # Test PRF
        try:
            with log_timed_operation("Testing classic PRF", request_id):
                prf_result = self.prf_classic(test_query, test_docs)
                test_results["results"]["prf_classic"] = {
                    "success": True,
                    "query_count": len(prf_result),
                    "sample_queries": prf_result[:3]
                }
        except Exception as e:
            test_results["results"]["prf_classic"] = {
                "success": False,
                "error": str(e)
            }

        # Determine overall status
        successful_tests = sum(1 for result in test_results["results"].values() if result.get("success", False))
        total_tests = len(test_results["results"])

        if successful_tests == total_tests:
            test_results["overall_status"] = "all_passed"
        elif successful_tests > 0:
            test_results["overall_status"] = "partial_success"
        else:
            test_results["overall_status"] = "all_failed"

        info_id(f"System test completed: {successful_tests}/{total_tests} tests passed", request_id)

        return test_results

    def optimize_for_model_type(self) -> None:
        """Optimize settings based on the loaded AI model type"""
        request_id = get_request_id()

        if not self.llm_available:
            info_id("No AI model loaded - using default settings", request_id)
            return

        ai_model_name = type(self.ai_model).__name__

        if isinstance(self.ai_model, (GPT4AllModel, TinyLlamaModel)):
            # Optimize for local models
            info_id(f"Optimizing for local model: {ai_model_name}", request_id)
            # Local models might be slower, so we could adjust timeouts, reduce variants, etc.
            # This is where you could add model-specific optimizations

        elif isinstance(self.ai_model, (OpenAIModel, AnthropicModel)):
            # Optimize for cloud models
            info_id(f"Optimizing for cloud model: {ai_model_name}", request_id)
            # Cloud models are fast and capable, can use more complex prompts

        else:
            info_id(f"Using default optimization for model: {ai_model_name}", request_id)


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Set a request ID for this execution
    execution_request_id = set_request_id()
    info_id("=== Starting Query Expansion Demo with AI Framework ===", execution_request_id)

    # Initialize the system with your AI framework
    info_id("Initializing QueryExpansionRAG system with AI framework", execution_request_id)

    # You can specify which models to use, or let it use the current ones from config
    expander = QueryExpansionRAG(
        # ai_model_name="TinyLlamaModel",  # Force specific AI model
        # embedding_model_name="TinyLlamaEmbeddingModel",  # Force specific embedding model
        # Or let it use the current models from ModelsConfig:
        ai_model_name=None,  # Use current AI model from config
        embedding_model_name=None,  # Use current embedding model from config
        use_spacy=True
    )

    # Get system status
    status = expander.get_system_status()
    info_id(f"System loaded - AI: {status['ai_model']['name']}, Embeddings: {status['embedding_model']['name']}", execution_request_id)

    # Example query
    test_query = "pump schematic"
    info_id(f"Testing with query: '{test_query}'", execution_request_id)

    info_id("=== QUERY EXPANSION WITH AI FRAMEWORK ===", execution_request_id)
    info_id(f"Original Query: '{test_query}'", execution_request_id)
    info_id(f"AI Model: {status['ai_model']['name']} (Available: {status['ai_model']['available']})", execution_request_id)
    info_id(f"Embedding Model: {status['embedding_model']['name']} (Available: {status['embedding_model']['available']})", execution_request_id)
    info_id(f"spaCy NER: {'Available' if status['spacy_available'] else 'Not Available'}", execution_request_id)

    # 1. Rule-based multi-query expansion (always available)
    info_id("1. RULE-BASED MULTI-QUERY EXPANSION:", execution_request_id)
    with log_timed_operation("Rule-based expansion demo", execution_request_id):
        rule_queries = expander.expand_query(test_query, method="multi_query_rules")
    for i, q in enumerate(rule_queries, 1):
        info_id(f"   {i}. {q}", execution_request_id)

    # 2. AI-based expansion (if AI model available)
    if status['available_methods']['multi_query_ai']:
        info_id("2. AI-BASED MULTI-QUERY EXPANSION:", execution_request_id)
        with log_timed_operation("AI expansion demo", execution_request_id):
            ai_queries = expander.expand_query(test_query, method="multi_query_ai", num_variants=3)
        for i, q in enumerate(ai_queries, 1):
            info_id(f"   {i}. {q}", execution_request_id)
    else:
        warning_id("2. AI-BASED MULTI-QUERY EXPANSION: [SKIPPED - AI model not available]", execution_request_id)

    # Example top documents for PRF
    example_docs = [
        "Centrifugal pump installation requires proper wiring diagram and electrical connections to motor drive unit.",
        "P&ID diagrams show piping and instrumentation details for pump stations including control valves and sensors.",
        "Pump maintenance manual includes schematic drawings for troubleshooting electrical and mechanical components.",
        "HVAC system pumps require regular inspection of bearings, seals, and impellers according to manufacturer specifications."
    ]

    info_id(f"Using {len(example_docs)} example documents for PRF demonstrations", execution_request_id)

    # 3. Classic PRF (always available)
    info_id("3. CLASSIC PSEUDO-RELEVANCE FEEDBACK:", execution_request_id)
    with log_timed_operation("Classic PRF demo", execution_request_id):
        prf_queries = expander.expand_query(test_query, method="prf_classic", top_docs=example_docs)
    for i, q in enumerate(prf_queries, 1):
        info_id(f"   {i}. {q}", execution_request_id)

    # 4. AI-augmented PRF (if AI model available)
    if status['available_methods']['prf_ai']:
        info_id("4. AI-AUGMENTED PSEUDO-RELEVANCE FEEDBACK:", execution_request_id)
        with log_timed_operation("AI PRF demo", execution_request_id):
            prf_ai_queries = expander.expand_query(test_query, method="prf_ai", top_docs=example_docs)
        for i, q in enumerate(prf_ai_queries, 1):
            info_id(f"   {i}. {q}", execution_request_id)
    else:
        warning_id("4. AI-AUGMENTED PSEUDO-RELEVANCE FEEDBACK: [SKIPPED - AI model not available]", execution_request_id)

    # 5. HyDE (if both AI and embedding models available)
    if status['available_methods']['hyde']:
        info_id("5. HYDE (HYPOTHETICAL DOCUMENT EMBEDDINGS):", execution_request_id)
        with log_timed_operation("HyDE demo", execution_request_id):
            hyde_results = expander.expand_query(test_query, method="hyde")
        info_id(f"   Original: {hyde_results[0]}", execution_request_id)
        info_id(f"   Hypothetical Doc: {hyde_results[1][:200]}...", execution_request_id)
    else:
        warning_id("5. HYDE (HYPOTHETICAL DOCUMENT EMBEDDINGS): [SKIPPED - Requires AI and embedding models]", execution_request_id)

    # 6. Comprehensive expansion demonstration
    info_id("6. COMPREHENSIVE EXPANSION (ALL AVAILABLE METHODS):", execution_request_id)
    with log_timed_operation("Comprehensive expansion demo", execution_request_id):
        all_results = expander.comprehensive_expansion(test_query, top_docs=example_docs)

    for method, queries in all_results.items():
        info_id(f"   {method.upper().replace('_', ' ')}:", execution_request_id)
        for i, q in enumerate(queries, 1):
            if method == "hyde" and i == 2:  # Truncate hypothetical document for display
                info_id(f"      {i}. {q[:100]}...", execution_request_id)
            else:
                info_id(f"      {i}. {q}", execution_request_id)

    # 7. System test
    info_id("7. SYSTEM TEST:", execution_request_id)
    with log_timed_operation("System test demo", execution_request_id):
        test_results = expander.test_system()

    info_id(f"   Overall Status: {test_results['overall_status'].upper().replace('_', ' ')}", execution_request_id)
    for method, result in test_results['results'].items():
        status_str = "PASSED" if result['success'] else "FAILED"
        info_id(f"   {method.upper().replace('_', ' ')}: {status_str}", execution_request_id)
        if not result['success']:
            error_id(f"      Error: {result.get('error', 'Unknown error')}", execution_request_id)

    info_id("=== SUMMARY ===", execution_request_id)
    info_id("AI Framework Integration: Complete", execution_request_id)
    info_id(f"AI Model: {status['ai_model']['name']} ({'Available' if status['ai_model']['available'] else 'Disabled'})", execution_request_id)
    info_id(f"Embedding Model: {status['embedding_model']['name']} ({'Available' if status['embedding_model']['available'] else 'Disabled'})", execution_request_id)
    info_id(f"Available Methods: {len([m for m, available in status['available_methods'].items() if available])}/5", execution_request_id)
    info_id("Comprehensive logging with request ID tracking", execution_request_id)
    info_id("Performance timing for all operations", execution_request_id)
    info_id("Automatic fallbacks when models unavailable", execution_request_id)

    info_id("=== Query Expansion Demo with AI Framework Completed ===", execution_request_id)