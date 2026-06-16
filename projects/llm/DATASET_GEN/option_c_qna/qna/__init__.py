"""
Q&A utilities for the Option C pipeline.

This package is currently a placeholder for:
    • Q&A record helpers
    • dataset formatting
    • future ORPO / preference-learning tools
    • post-generation validation utilities

Modules can be added here as the Q&A tooling expands.
"""

# Nothing exported yet; ready for future expansion.
__all__ = []
"""
Option C – Q&A Output Package

This package contains:
    - Generated Q&A datasets (JSONL, Excel)
    - Multi-model comparison results
    - Debug logs / traces
    - Export utilities for Option C
    - Future ORPO-ready training files

Modules inside this package may:
    - Write outputs
    - Manage output directories
    - Bundle Q&A sets
    - Provide helper functions for pipeline steps

Nothing in this __init__ modifies behavior — it simply exposes
the key public symbols for convenience.
"""

from pathlib import Path

# Public exports (so you can import them as option_c_qna.qna.*)
__all__ = [
    "qna_output_dir",
    "ensure_qna_output_dir",
]

# ---------------------------------------------------------------------
# OUTPUT DIRECTORY HANDLING
# ---------------------------------------------------------------------

from option_c_qna.configuration.config import cfg


def qna_output_dir() -> Path:
    """
    Return the directory where all Q&A files should be written.

    This is tied to:
        cfg.OUTPUT_DIR  (from the master EMTAC .env)
    """
    return cfg.OUTPUT_DIR


def ensure_qna_output_dir() -> Path:
    """
    Creates the Q&A directory if missing.
    Safe to call from anywhere in the pipeline.
    """
    out = qna_output_dir()
    out.mkdir(parents=True, exist_ok=True)
    return out
