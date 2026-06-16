from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel


class VisionMarkdownResponse(BaseModel):
    markdown: str
    model: str
    request_id: str
    timing: Dict[str, float]


class VisionDescribeResponse(BaseModel):
    description: str
    model: str
    request_id: str
    timing: Dict[str, float]


class VisionPDFChunk(BaseModel):
    chunk_id: str
    page_number: int
    markdown: str
    text: str
    images: List[str]


class VisionPDFResponse(BaseModel):
    source_path: str
    doc_type: str
    total_pages: int
    chunks: List[VisionPDFChunk]
    images: List[Dict[str, Any]]
    model: str
    request_id: str
    timing: Dict[str, float]