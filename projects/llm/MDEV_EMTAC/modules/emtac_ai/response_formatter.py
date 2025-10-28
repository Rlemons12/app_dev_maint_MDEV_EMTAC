import re
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from modules.configuration.log_config import error_id



class ResponseFormatter:
    """Utility class for formatting search responses."""

    @staticmethod
    def format_search_results(result):
        """Format search results into a user-friendly response."""
        try:
            if not result or not isinstance(result, dict):
                return "I couldn't find relevant information for your query."

            if result.get("status") != "success":
                error_msg = result.get("message", "Search failed")
                return f"Search error: {error_msg}"

            # ✅ Only attempt summarization if method is an AI synthesis type
            if "answer" in result and result.get("method") in [
                "ai_knowledge_synthesis_with_chunks",
                "ai_knowledge_synthesis_direct",
            ]:
                ai_answer = result["answer"]
                if "source_info" in result:
                    source_info = result["source_info"]
                    if source_info.get("document_source") and source_info.get(
                            "chunk_similarity"
                    ):
                        similarity = source_info["chunk_similarity"]
                        doc_source = source_info["document_source"]
                        ai_answer += f"\n\n*Source: {doc_source} (Similarity: {similarity:.1%})*"
                    elif source_info.get("source_type") == "ai_general_knowledge":
                        ai_answer += f"\n\n*Source: AI General Knowledge*"
                return ai_answer

            total_results = result.get("total_results", 0)
            if total_results == 0:
                return "No results found for your query."

            if "organized_results" in result:
                return ResponseFormatter._format_organized_results(
                    result["organized_results"], total_results
                )
            elif "results_by_type" in result:
                return ResponseFormatter._format_results_by_type(
                    result["results_by_type"], total_results
                )
            elif "results" in result and isinstance(result["results"], list):
                return ResponseFormatter._format_direct_results(
                    result["results"], total_results
                )
            elif total_results > 0:
                return ResponseFormatter._format_main_result_structure(
                    result, total_results
                )

            return f"Found {total_results} results for your query."

        except Exception as e:
            logger.error(f"Error formatting search results: {e}", exc_info=True)
            return "Found some results, but had trouble formatting them."

    @staticmethod
    def _format_organized_results(organized_results, total_results):
        """Format organized results structure."""
        parts = []

        if 'parts' in organized_results and organized_results['parts']:
            parts_list = organized_results['parts'][:10]
            parts.append(f"Found {len(parts_list)} Banner sensor{'s' if len(parts_list) != 1 else ''}:")
            for i, part in enumerate(parts_list, 1):
                part_info = f"{i}. {part.get('part_number', 'Unknown')}"
                if part.get('name'):
                    part_info += f" - {part.get('name')}"
                if part.get('oem_mfg'):
                    part_info += f" (Manufacturer: {part['oem_mfg']})"
                parts.append(part_info)

        if 'images' in organized_results and organized_results['images']:
            image_count = len(organized_results['images'])
            parts.append(f"\nFound {image_count} related image{'s' if image_count != 1 else ''}.")

        if 'positions' in organized_results and organized_results['positions']:
            position_count = len(organized_results['positions'])
            parts.append(f"\nFound {position_count} installation location{'s' if position_count != 1 else ''}.")

        return "\n".join(parts) if parts else f"Found {total_results} results for your query."

    @staticmethod
    def _format_results_by_type(results_by_type, total_results):
        """Format results_by_type structure."""
        response_parts = []

        # Handle parts
        if 'parts' in results_by_type and results_by_type['parts']:
            parts_list = results_by_type['parts'][:10]
            response_parts.append(f"Found {len(parts_list)} part{'s' if len(parts_list) != 1 else ''}:")
            for i, part in enumerate(parts_list, 1):
                part_info = f"{i}. {part.get('part_number', 'Unknown')}"
                if part.get('name'):
                    part_info += f" - {part.get('name')}"
                if part.get('oem_mfg'):
                    part_info += f" (Manufacturer: {part['oem_mfg']})"
                response_parts.append(part_info)

        # Handle other types
        for result_type, results in results_by_type.items():
            if result_type == 'parts' or not results:
                continue
            if response_parts:
                response_parts.append("")
            type_name = result_type.replace('_', ' ').title()
            response_parts.append(f"Found {len(results)} {type_name}:")
            for i, item in enumerate(results[:5], 1):
                item_info = f"{i}. {item.get('title', item.get('name', 'Unknown'))}"
                response_parts.append(item_info)

        return "\n".join(response_parts) if response_parts else f"Found {total_results} results."

    @staticmethod
    def _format_direct_results(results, total_results):
        """Format direct results list structure - handles both dicts and SQLAlchemy objects."""
        if not results or not isinstance(results, list):
            return f"Found {total_results} results for your query."
        if len(results) == 0:
            return "No results found for your query."

        response_parts = []
        results_list = results[:10]  # Limit to first 10 results
        response_parts.append(f"Found {len(results_list)} result{'s' if len(results_list) != 1 else ''}:")
        for i, item in enumerate(results_list, 1):
            # Handle SQLAlchemy Part-like objects
            if hasattr(item, 'part_number'):
                part_info = f"{i}. {item.part_number or 'Unknown Part'}"
                if getattr(item, 'name', None):
                    part_info += f" - {item.name}"
                if getattr(item, 'oem_mfg', None):
                    part_info += f" (Manufacturer: {item.oem_mfg})"
                if getattr(item, 'model', None) and item.model != getattr(item, 'name', None):
                    part_info += f" [Model: {item.model}]"
                response_parts.append(part_info)

            elif isinstance(item, dict):
                if 'part_number' in item:
                    part_info = f"{i}. {item.get('part_number', 'Unknown')}"
                    if item.get('name'):
                        part_info += f" - {item.get('name')}"
                    if item.get('oem_mfg'):
                        part_info += f" (Manufacturer: {item['oem_mfg']})"
                    response_parts.append(part_info)
                else:
                    name = (item.get('name') or item.get('title') or item.get('id') or 'Unknown')
                    description = (item.get('description') or item.get('notes') or item.get('model') or item.get('oem_mfg') or '')
                    if description:
                        response_parts.append(f"{i}. {name} - {description}")
                    else:
                        response_parts.append(f"{i}. {name}")
            else:
                response_parts.append(f"{i}. {str(item)}")

        return "\n".join(response_parts)

    @staticmethod
    def _format_main_result_structure(result, total_results):
        """Handle main result structure."""
        response_parts = []
        if 'summary' in result:
            response_parts.append(result['summary'])

        found_results = False
        for key in ['parts', 'results', 'data', 'items']:
            if key in result and isinstance(result[key], list) and result[key]:
                found_results = True
                items_list = result[key][:10]
                if not response_parts:
                    response_parts.append(f"Found {len(items_list)} result{'s' if len(items_list) != 1 else ''}:")
                for i, item in enumerate(items_list, 1):
                    if isinstance(item, dict):
                        if 'part_number' in item:
                            part_info = f"{i}. {item.get('part_number', 'Unknown')}"
                            if item.get('name'):
                                part_info += f" - {item.get('name')}"
                            response_parts.append(part_info)
                        else:
                            name = item.get('name', item.get('title', item.get('id', 'Unknown')))
                            response_parts.append(f"{i}. {name}")
                    else:
                        response_parts.append(f"{i}. {str(item)}")
                break

        if not found_results:
            response_parts.append(f"Found {total_results} results for your query.")
        return "\n".join(response_parts) if response_parts else f"Found {total_results} results."