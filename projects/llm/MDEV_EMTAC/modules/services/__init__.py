"""
Service Layer Package

⚠ IMPORTANT ARCHITECTURAL RULES

- This package is foundational infrastructure
- DO NOT eagerly import services here
- DO NOT import AI / RAG / model code here
- Services must be imported explicitly by consumers

This prevents:
- Circular imports
- Accidental AI startup
- SQLAlchemy metadata duplication
"""

__all__ = []
