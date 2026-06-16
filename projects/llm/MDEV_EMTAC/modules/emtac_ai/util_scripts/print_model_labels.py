# tools/print_model_labels.py
import json
import sys
from pathlib import Path

from modules.configuration.log_config import (
    info_id, warning_id, error_id, set_request_id, get_request_id
)

def _sorted_id2label_str(id2label: dict) -> str:
    try:
        pairs = sorted(((int(k), v) for k, v in id2label.items()))
    except Exception:
        # already int keys or mixed
        def keyfn(x):
            k = x[0]
            return int(k) if isinstance(k, str) and k.isdigit() else k
        pairs = sorted(id2label.items(), key=keyfn)
    return ", ".join(f"{k}:{v}" for k, v in pairs)

def print_labels(path: Path):
    rid = get_request_id()
    cfg = path / "config.json"
    if not cfg.exists():
        warning_id(f"[{path}] no config.json", rid)
        return
    try:
        j = json.loads(cfg.read_text(encoding="utf-8"))
        id2 = j.get("id2label", {})
        l2i = j.get("label2id", {})
        info_id(f"[{path}] model_type={j.get('model_type')} arch={j.get('architectures')}", rid)
        info_id(f"[{path}] num_labels={j.get('num_labels') or j.get('_num_labels')}", rid)
        info_id(f"[{path}] id2label={{ {_sorted_id2label_str(id2)} }}", rid)
        info_id(f"[{path}] label2id_keys={sorted(list(l2i.keys()))[:10]}{'...' if len(l2i)>10 else ''}", rid)
    except Exception as e:
        error_id(f"[{path}] failed to read config: {e}", rid)

def main():
    set_request_id()
    args = sys.argv[1:]
    if not args:
        print("Usage: python tools/print_model_labels.py <model_folder> [<model_folder_2> ...]")
        sys.exit(1)
    for p in args:
        print_labels(Path(p))

if __name__ == "__main__":
    main()

