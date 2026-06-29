import os
from pathlib import Path


def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def list_tree(root: Path, limit: int = 50):
    if not root.exists():
        return []
    out = []
    for i, p in enumerate(root.rglob("*")):
        if i >= limit:
            break
        out.append(str(p))
    return out
