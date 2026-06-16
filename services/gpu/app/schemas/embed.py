from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    model: Optional[str] = Field(
        default=None,
        description="Model key to use (defaults to service default)",
    )
    normalize: bool = Field(default=True)
    batch_size: int = Field(default=32, ge=1, le=512)


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    device: str
    dims: int
    normalized: bool
    meta: Dict[str, Any] = Field(default_factory=dict)
