"""
Utility package for AI helpers.

Keep this file intentionally light. Utility modules often depend on many
other modules, so broad re-exports here can increase circular-import risk.

Preferred imports:
    from modules.ai.utils.embedding_storage import generate_embedding
    from modules.ai.utils.model_diagnostics import diagnose_models
    from modules.ai.utils.model_downloads import download_recommended_models
    from modules.ai.utils.model_recommendations import get_recommended_model_setup
"""

__all__: list[str] = []
