"""
Bootstrap helpers for AI package setup.

Keep this package export small. Import concrete modules directly inside
the package implementation if circular imports appear.
"""

from .initialize_models import (

    initialize_models_config,
    register_default_models,
    register_default_models_with_tinyllama,
    register_default_models_with_tinyllama_updated,
)

__all__ = [
    "initialize_models_config",
    "register_default_models",
    "register_default_models_with_tinyllama",
    "register_default_models_with_tinyllama_updated",
]
