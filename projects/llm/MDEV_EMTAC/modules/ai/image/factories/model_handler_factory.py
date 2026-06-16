"""Factory helpers for resolving image model handler instances."""


from __future__ import annotations


import sys

from modules.ai.image.base import BaseImageModelHandler
from modules.ai.image.models import NoImageModel
from modules.configuration.log_config import logger, with_request_id, debug_id, info_id, warning_id, error_id, get_request_id, log_timed_operation

def get_image_model_handler(model_name):
    module = sys.modules[__name__]
    try:
        model_class = getattr(module, model_name)
        if issubclass(model_class, BaseImageModelHandler):
            return model_class()
        else:
            raise ValueError(f"{model_name} is not a subclass of BaseImageModelHandler")
    except AttributeError:
        logger.error(f"{model_name} not found in {__name__}")
        return NoImageModel()
