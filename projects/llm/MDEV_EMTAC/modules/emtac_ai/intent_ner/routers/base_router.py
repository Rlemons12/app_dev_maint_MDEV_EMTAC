from typing import Any, Dict, List, Callable, Optional, Tuple


class BaseRouter:
    """
    Generic router framework for EMTAC intent routing.

    Handlers may return:
        - list[models]
        - OR tuple(list[models], matched_on)
        - OR tuple(list[models], matched_on, serializer_override)
    """

    def __init__(self, service, serializer: Callable):
        self.service = service
        self.serializer = serializer

    # ------------------------------------------------------------
    # PUBLIC ENTRY
    # ------------------------------------------------------------
    def route(
            self,
            *,
            text: str,
            intent: str,
            confidence: float,
            entities: Dict[str, Any],
            priority_handlers: List[Callable],
            fallback: Optional[Callable] = None
    ) -> Dict[str, Any]:

        # 1) Priority handlers
        for handler in priority_handlers:
            result = handler(entities)

            if result:
                models, matched_on, serializer, related_override = self._unpack(result, handler.__name__)

                # If handler returned its own related info, use it.
                related = related_override if related_override is not None else self._fetch_related(models)

                return self._format_success(
                    intent=intent,
                    confidence=confidence,
                    matched_on=matched_on,
                    models=models,
                    serializer=serializer,
                    related=related
                )

        # 2) Fallback handler
        if fallback:
            fb_result = fallback(text)

            if fb_result:
                models, matched_on, serializer, related_override = self._unpack(fb_result, "fallback")

                return self._format_success(
                    intent=intent,
                    confidence=confidence,
                    matched_on=matched_on,
                    models=models,
                    serializer=serializer,
                    related=related_override  # fallback usually has no related info
                )

        # 3) Nothing matched
        return self._format_empty(intent, confidence)

    # ------------------------------------------------------------
    # Result unpack helper
    # ------------------------------------------------------------
    def _unpack(self, result, default_match):
        """
        Normalize handler output into:
            (models, matched_on, serializer, related)
        """

        # (models, matched_on, serializer) or (models, matched_on, related_dict)
        if isinstance(result, tuple):

            if len(result) == 2:
                return result[0], result[1], self.serializer, None

            if len(result) == 3:
                models, matched_on, extra = result

                if callable(extra):
                    # Serializer override
                    return models, matched_on, extra, None
                else:
                    # Related info dictionary
                    return models, matched_on, self.serializer, extra

        # Handler returned just a list
        return result, default_match, self.serializer, None

    # ------------------------------------------------------------
    def _fetch_related(self, models: List[Any]):
        """
        If service supports find_related(), attempt it.
        """
        if not models:
            return None

        if hasattr(self.service, "find_related"):
            try:
                return self.service.find_related(models[0].id)
            except Exception:
                return None

        return None

    # ------------------------------------------------------------
    def _format_success(self, *, intent, confidence, matched_on, models, serializer, related):
        return {
            "router": intent,
            "intent": intent,
            "confidence": confidence,
            "matched_on": matched_on,
            "results": [serializer(m) for m in models],
            "related": related,
        }

    def _format_empty(self, intent, confidence):
        return {
            "router": intent,
            "intent": intent,
            "confidence": confidence,
            "matched_on": "no_results",
            "results": [],
            "related": None,
        }
