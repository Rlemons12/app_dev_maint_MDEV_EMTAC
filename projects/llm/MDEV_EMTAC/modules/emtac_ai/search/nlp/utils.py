# modules/search/utils.py
"""
Utility functions for the search module.
Provides common functionality for AggregateSearch and SpaCyEnhancedAggregateSearch.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def normalize_part_number(part_number: str) -> str:
    """
    Normalize part number formats for consistent searching.

    Args:
        part_number: Raw part number string

    Returns:
        Normalized part number
    """
    if not part_number:
        return ""

    # Remove extra whitespace and convert to uppercase
    normalized = part_number.strip().upper()

    # Common normalizations
    normalizations = [
        (r'\s+', ''),  # Remove all whitespace
        (r'[_]+', '-'),  # Convert underscores to dashes
        (r'[-]{2,}', '-'),  # Multiple dashes to single dash
    ]

    for pattern, replacement in normalizations:
        normalized = re.sub(pattern, replacement, normalized)

    return normalized


def log_search_performance(search_type: str, execution_time_ms: int, result_count: int,
                           user_input: str = "", success: bool = True) -> None:
    """
    Log search performance metrics for monitoring and analytics.

    Args:
        search_type: Type of search performed
        execution_time_ms: Execution time in milliseconds
        result_count: Number of results returned
        user_input: Original user input (optional)
        success: Whether the search was successful
    """
    try:
        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "search_type": search_type,
            "execution_time_ms": execution_time_ms,
            "result_count": result_count,
            "success": success,
            "user_input_length": len(user_input) if user_input else 0
        }

        # Log at appropriate level based on performance
        if execution_time_ms > 5000:  # > 5 seconds
            logger.warning(f"Slow search performance: {performance_data}")
        elif execution_time_ms > 1000:  # > 1 second
            logger.info(f"Search performance: {performance_data}")
        else:
            logger.debug(f"Search performance: {performance_data}")

    except Exception as e:
        logger.error(f"Error logging search performance: {e}")


def fuzzy_match_score(query: str, target: str, case_sensitive: bool = False) -> float:
    """
    Calculate simple fuzzy match score between two strings.

    Args:
        query: Query string
        target: Target string to match against
        case_sensitive: Whether matching should be case sensitive

    Returns:
        Match score between 0.0 and 1.0
    """
    if not query or not target:
        return 0.0

    if not case_sensitive:
        query = query.lower()
        target = target.lower()

    # Exact match
    if query == target:
        return 1.0

    # Substring match
    if query in target:
        return len(query) / len(target)

    # Character overlap score
    query_chars = set(query)
    target_chars = set(target)
    overlap = len(query_chars & target_chars)
    total_chars = len(query_chars | target_chars)

    if total_chars == 0:
        return 0.0

    return overlap / total_chars


def validate_search_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean search parameters.

    Args:
        params: Raw search parameters

    Returns:
        Cleaned and validated parameters
    """
    cleaned = {}

    # Clean string parameters
    string_params = [
        'part_number', 'area', 'equipment', 'equipment_group', 'description',
        'search_text', 'raw_input', 'title', 'name', 'oem_mfg', 'model', 'location'
    ]
    for param in string_params:
        if param in params and params[param]:
            value = str(params[param]).strip()
            if value:
                cleaned[param] = value

    # Clean numeric parameters
    numeric_params = [
        'extracted_id', 'reference_image_id', 'limit', 'area_id', 'equipment_group_id',
        'model_id', 'location_id', 'position_id', 'part_id', 'image_id'
    ]
    for param in numeric_params:
        if param in params and params[param] is not None:
            try:
                cleaned[param] = int(params[param])
            except (ValueError, TypeError):
                logger.warning(f"Invalid numeric parameter {param}: {params[param]}")

    # Clean float parameters
    float_params = ['similarity_threshold', 'confidence_threshold']
    for param in float_params:
        if param in params and params[param] is not None:
            try:
                cleaned[param] = float(params[param])
            except (ValueError, TypeError):
                logger.warning(f"Invalid float parameter {param}: {params[param]}")

    # Set defaults
    if 'limit' not in cleaned:
        cleaned['limit'] = 10

    if 'similarity_threshold' not in cleaned:
        cleaned['similarity_threshold'] = 0.7

    # Normalize part numbers
    if 'part_number' in cleaned:
        cleaned['part_number'] = normalize_part_number(cleaned['part_number'])

    # Clean area identifiers
    if 'area' in cleaned:
        cleaned['area'] = cleaned['area'].upper()

    return cleaned


def format_search_results(results: List[Dict], search_type: str = "generic") -> Dict[str, Any]:
    """
    Format search results for consistent output.

    Args:
        results: Raw search results
        search_type: Type of search performed

    Returns:
        Formatted results dictionary
    """
    if not results:
        return {
            "status": "success",
            "count": 0,
            "results": [],
            "search_type": search_type,
            "message": "No results found"
        }

    formatted_results = []
    for result in results:
        if isinstance(result, dict):
            # Ensure consistent fields
            formatted_result = {
                "id": result.get("id"),
                "title": result.get("title", result.get("name", "Unknown")),
                "description": result.get("description", ""),
                **result  # Include all original fields
            }
            formatted_results.append(formatted_result)

    return {
        "status": "success",
        "count": len(formatted_results),
        "results": formatted_results,
        "search_type": search_type,
        "timestamp": datetime.utcnow().isoformat()
    }


def merge_search_results(*result_sets: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple search result sets into a comprehensive result.

    Args:
        *result_sets: Variable number of search result dictionaries

    Returns:
        Merged comprehensive search results
    """
    all_results = []
    total_count = 0
    search_types = []
    error_messages = []

    for result_set in result_sets:
        if not isinstance(result_set, dict):
            continue

        if result_set.get("status") == "success":
            results = result_set.get("results", [])
            all_results.extend(results)
            total_count += result_set.get("count", 0)

            search_type = result_set.get("search_type")
            if search_type and search_type not in search_types:
                search_types.append(search_type)

        elif result_set.get("status") == "error":
            error_msg = result_set.get("message", "Unknown error")
            error_messages.append(error_msg)

    # Remove duplicates based on ID
    unique_results = []
    seen_ids = set()

    for result in all_results:
        result_id = result.get("id")
        if result_id and result_id not in seen_ids:
            unique_results.append(result)
            seen_ids.add(result_id)
        elif not result_id:  # Include results without IDs
            unique_results.append(result)

    # Sort by relevance/score if available
    try:
        unique_results.sort(
            key=lambda x: x.get("score", x.get("relevance", 0)),
            reverse=True
        )
    except (TypeError, KeyError):
        pass  # Skip sorting if no comparable scores

    # Determine overall status
    if unique_results:
        status = "success"
        message = f"Found {len(unique_results)} results from {len(search_types)} search methods"
    elif error_messages:
        status = "error"
        message = f"Search failed: {'; '.join(error_messages)}"
    else:
        status = "success"
        message = "No results found"

    return {
        "status": status,
        "count": len(unique_results),
        "results": unique_results,
        "search_types": search_types,
        "message": message,
        "errors": error_messages if error_messages else None,
        "timestamp": datetime.utcnow().isoformat()
    }


def clean_text_for_search(text: str) -> str:
    """
    Clean and normalize text for search operations.

    Args:
        text: Raw text input

    Returns:
        Cleaned text suitable for searching
    """
    if not text:
        return ""

    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', text.strip())

    # Remove special characters except alphanumeric, spaces, hyphens, dots
    cleaned = re.sub(r'[^\w\s\-\.]', ' ', cleaned)

    # Normalize multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def extract_equipment_keywords(text: str) -> List[str]:
    """
    Extract equipment-related keywords from text.

    Args:
        text: Input text

    Returns:
        List of equipment keywords
    """
    equipment_keywords = {
        'pump', 'motor', 'valve', 'bearing', 'filter', 'sensor', 'compressor',
        'fan', 'blower', 'conveyor', 'gearbox', 'coupling', 'seal', 'gasket',
        'impeller', 'rotor', 'stator', 'shaft', 'housing', 'casing', 'frame',
        'belt', 'chain', 'gear', 'spring', 'bracket', 'mount', 'guard',
        'cylinder', 'piston', 'rod', 'tube', 'pipe', 'hose', 'fitting',
        'connector', 'switch', 'relay', 'controller', 'display', 'gauge'
    }

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    found_keywords = [word for word in words if word in equipment_keywords]

    # Remove duplicates while preserving order
    unique_keywords = []
    for keyword in found_keywords:
        if keyword not in unique_keywords:
            unique_keywords.append(keyword)

    return unique_keywords


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using simple word overlap.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0

    # Convert to lowercase and extract words
    words1 = set(re.findall(r'\b[a-zA-Z]+\b', text1.lower()))
    words2 = set(re.findall(r'\b[a-zA-Z]+\b', text2.lower()))

    if not words1 or not words2:
        return 0.0

    # Calculate Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def format_error_response(error_message: str, error_type: str = "general_error",
                          additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Format error response with consistent structure.

    Args:
        error_message: Error message
        error_type: Type of error
        additional_data: Additional data to include

    Returns:
        Formatted error response
    """
    response = {
        "status": "error",
        "error_type": error_type,
        "message": error_message,
        "timestamp": datetime.utcnow().isoformat(),
        "count": 0,
        "results": []
    }

    if additional_data:
        response.update(additional_data)

    return response


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with optional suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def validate_numeric_range(value: Any, min_val: float = None, max_val: float = None,
                           param_name: str = "value") -> Optional[float]:
    """
    Validate that a numeric value is within specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Parameter name for error messages

    Returns:
        Validated numeric value or None if invalid
    """
    try:
        numeric_value = float(value)

        if min_val is not None and numeric_value < min_val:
            logger.warning(f"{param_name} {numeric_value} below minimum {min_val}")
            return None

        if max_val is not None and numeric_value > max_val:
            logger.warning(f"{param_name} {numeric_value} above maximum {max_val}")
            return None

        return numeric_value

    except (ValueError, TypeError):
        logger.warning(f"Invalid numeric value for {param_name}: {value}")
        return None
