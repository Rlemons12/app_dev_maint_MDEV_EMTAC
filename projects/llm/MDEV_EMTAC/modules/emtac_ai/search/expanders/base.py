from abc import ABC, abstractmethod
from typing import Dict
from modules.emtac_ai.search.expanders.search_expansion_result import SearchExpansionResult


class BaseSearchExpander(ABC):
    """
    Base contract for all search expanders.
    """

    intent: str

    @abstractmethod
    def expand(self, query: str, entities: Dict) -> SearchExpansionResult:
        raise NotImplementedError
