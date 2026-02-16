import sys
from pathlib import Path

TRACE_ROOT = str(Path(__file__).resolve().parents[3])  # adjust if needed

call_depth = 0


def trace_calls(frame, event, arg):
    global call_depth

    if event not in ("call", "return"):
        return trace_calls

    code = frame.f_code
    filename = code.co_filename

    # Only trace your project files
    if "emtac" not in filename.lower():
        return trace_calls

    if event == "call":
        call_depth += 1
        print("  " * call_depth + f"→ {code.co_name}")

    elif event == "return":
        print("  " * call_depth + f"← {code.co_name}")
        call_depth -= 1

    return trace_calls
