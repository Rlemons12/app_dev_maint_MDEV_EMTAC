#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Document Structure Package
"""

from .doc_structure_mapping import (
    DocumentStructureMap,
    ImagePosition,
    ChunkBoundary,
)

from .structure_analyzer import DocumentStructureAnalyzer

__all__ = [
    "DocumentStructureMap",
    "ImagePosition",
    "ChunkBoundary",
    "DocumentStructureAnalyzer",
]
