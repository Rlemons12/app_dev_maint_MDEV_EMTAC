#!/usr/bin/env python3
"""
generate_ai_pipeline_diagram.py
--------------------------------
Generates architecture diagrams (PNG + SVG) for the AI pipeline
using Graphviz, with integrated logging.
"""

import os
from graphviz import Digraph

# Ensure Graphviz executables (dot.exe) are on PATH
os.environ["PATH"] += os.pathsep + r"C:\Users\10169062\PycharmProjects\windows_10_cmake_Release_Graphviz-14.0.0-win64\Graphviz-14.0.0-win64\bin"

# Import your logger
from modules.configuration.log_config import (
    info_id, debug_id, error_id, get_request_id
)


def build_ai_pipeline_diagram(output_dir: str = "docs/diagrams", filename: str = "ai_pipeline"):
    request_id = get_request_id()
    try:
        info_id("Starting AI pipeline diagram generation", request_id)

        os.makedirs(output_dir, exist_ok=True)

        dot = Digraph("AIPipeline", format="png")
        dot.attr(rankdir="LR")
        dot.attr("node", shape="box", style="filled", color="lightgray", fontname="Helvetica")

        # User
        dot.node("User", "User", shape="oval", color="lightblue", style="filled")

        # AistManager cluster
        with dot.subgraph(name="cluster_aist") as c:
            c.attr(label="AistManager", color="blue", style="rounded")
            c.node("AIST", "AistManager\n(answer_question)", shape="box3d", color="lightyellow")
            c.node("Formatter", "ResponseFormatter", shape="box")
            c.node("DB", "QandA / Analytics DB", shape="cylinder", color="lightgreen")

        # UnifiedSearch cluster
        with dot.subgraph(name="cluster_us") as c:
            c.attr(label="UnifiedSearch Hub", color="darkgreen", style="rounded")
            c.node("US", "UnifiedSearch\n(execute_unified_search)", shape="box")
            c.node("Tracker", "SearchTracker /\nQueryTracker", shape="box", color="lightpink")

        # Orchestrator cluster
        with dot.subgraph(name="cluster_orch") as c:
            c.attr(label="Query Expansion Orchestrator", color="darkorange", style="rounded")
            c.node("ORCH", "EMTACQueryExpansionOrchestrator", shape="box")
            c.node("IE", "IntentEntityPlugin\n(Intent + NER)", shape="box")
            c.node("RAG", "QueryExpansionRAG\n(Synonyms / AI Expansions)", shape="box")

        # Edges
        dot.edge("User", "AIST")
        dot.edge("AIST", "US")
        dot.edge("US", "ORCH")
        dot.edge("ORCH", "IE")
        dot.edge("ORCH", "RAG")
        dot.edge("US", "Tracker")
        dot.edge("US", "AIST")
        dot.edge("AIST", "Formatter")
        dot.edge("AIST", "DB")
        dot.edge("Formatter", "AIST")
        dot.edge("AIST", "User", label="Answer", color="blue")

        # Output files
        png_path = os.path.join(output_dir, f"{filename}.png")
        svg_path = os.path.join(output_dir, f"{filename}.svg")

        dot.render(os.path.join(output_dir, filename), format="png", cleanup=True)
        dot.render(os.path.join(output_dir, filename), format="svg", cleanup=True)

        info_id(f"AI pipeline diagram generated successfully", request_id)
        debug_id(f"PNG saved at {png_path}", request_id)
        debug_id(f"SVG saved at {svg_path}", request_id)

        return {"png": png_path, "svg": svg_path}

    except Exception as e:
        error_id(f"Diagram generation failed: {e}", request_id)
        raise


if __name__ == "__main__":
    build_ai_pipeline_diagram()
