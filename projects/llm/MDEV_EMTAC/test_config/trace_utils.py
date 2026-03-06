import sys
from contextlib import contextmanager


@contextmanager
def trace_calls():
    """
    Capture full Python call trace during execution.
    """
    call_stack = []

    def tracer(frame, event, arg):
        if event == "call":
            func_name = frame.f_code.co_name
            module = frame.f_globals.get("__name__", "")
            call_stack.append(f"{module}.{func_name}")
        return tracer

    sys.settrace(tracer)
    try:
        yield call_stack
    finally:
        sys.settrace(None)
