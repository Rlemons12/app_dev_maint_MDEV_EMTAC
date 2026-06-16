#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DATASET_GEN / data_gen
======================

This is the root package for the entire structure-aware Q&A dataset pipeline.

Layers:
    1. document_structure_extractor – structure analysis models + analyzer
    2. extraction         – guided chunk + image extraction
    3. qna                – structure-aware Q&A generation
    4. pipeline           – orchestrators / runners

This package is designed to be portable and works entirely offline.
"""

from .document_structure import (
    DocumentStructureAnalyzer,
    DocumentStructureMap,
    ImagePosition,
    ChunkBoundary,
)

__all__ = [
    "DocumentStructureAnalyzer",
    "DocumentStructureMap",
    "ImagePosition",
    "ChunkBoundary",
]
