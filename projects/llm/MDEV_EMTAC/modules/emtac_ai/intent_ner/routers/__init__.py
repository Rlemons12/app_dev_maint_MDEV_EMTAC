"""
Router registry for EMTAC intent_ner.

Maps intent labels (normalized by IntentOrchestrator)
to router callables.

Required:
    - Keys MUST match normalized intent names
    - MUST include a "default" router
"""

from .parts_router import parts_router
from .images_router import images_router
from .documents_router import documents_router
from .tools_router import tools_router
from .drawings_router import drawings_router
from .troubleshooting_router import troubleshooting_router
#from .general_router import general_chat_router   # used as fallback
from .default_router import default_router        # recommended fallback handler


ROUTER_MAP = {
    # ----------------------------------------
    # Primary domain intents
    # These keys must match normalized intent labels
    # ----------------------------------------
    "Parts": parts_router,
    "Images": images_router,
    "Documents": documents_router,
    "Tools": tools_router,
    "Drawings": drawings_router,
    "Troubleshooting": troubleshooting_router,

    # ----------------------------------------
    # Optional generic conversational intent
    # ----------------------------------------
    #"General_Chat": general_chat_router,

    # ----------------------------------------
    # REQUIRED fallback router
    # This MUST exist or orchestrator crashes
    # ----------------------------------------
    "default": default_router,
}
