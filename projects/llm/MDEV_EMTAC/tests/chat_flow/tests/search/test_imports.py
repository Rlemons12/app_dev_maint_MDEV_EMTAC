import importlib
import pytest

@pytest.mark.parametrize("module_path", [
    "modules.emtac_ai.search",
    "modules.emtac_ai.search.nlp",
])
def test_all_exports_importable(module_path):
    """
    Ensure every symbol in __all__ of the module can be imported
    and is not None.
    """
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "__all__"), f"{module_path} missing __all__"

    for symbol in mod.__all__:
        obj = getattr(mod, symbol, None)
        assert obj is not None, f"{symbol} in {module_path} is None"
