#!/usr/bin/env python3
"""
Create or update an __init__.py in a given directory (or in the script's own dir if no --dir is provided).

Examples:
    # Run in the current directory
    python make_init.py

    # Generate __init__.py in a specific directory
    python make_init.py --dir modules/emtac_ai/training_scripts/dataset_gen
"""

import ast
import argparse
from pathlib import Path

# Import your custom logger
try:
    from modules.configuration.log_config import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("make_init")

MANAGED_START = "# === AUTO-IMPORTS: BEGIN (managed) ==="
MANAGED_END   = "# === AUTO-IMPORTS: END (managed) ==="

def get_public_symbols(py_file: Path):
    """Parse a .py file and return public functions, classes, and constants."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Could not parse {py_file}: {e}")
        return []
    names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            names.append(node.name)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    names.append(t.id)
    return sorted(names)

def build_imports_block(pkg_dir: Path) -> str:
    """Generate the managed import block for all sibling .py files in pkg_dir."""
    lines = [MANAGED_START]
    all_syms = []
    for py in sorted(pkg_dir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        if py.stem.startswith("_"):
            continue
        symbols = get_public_symbols(py)
        if symbols:
            lines.append(f"from .{py.stem} import {', '.join(symbols)}")
            all_syms.extend(symbols)
        else:
            lines.append(f"from . import {py.stem}")
            all_syms.append(py.stem)
    lines.append("")
    lines.append("__all__ = [")
    for s in all_syms:
        lines.append(f'    "{s}",')
    lines.append("]")
    lines.append(MANAGED_END)
    lines.append("")
    return "\n".join(lines)

def write_init(pkg_dir: Path):
    """Write or update the __init__.py file in pkg_dir."""
    init_path = pkg_dir / "__init__.py"
    block = build_imports_block(pkg_dir)

    if init_path.exists():
        orig = init_path.read_text(encoding="utf-8")
        if MANAGED_START in orig and MANAGED_END in orig:
            before, _, tail = orig.partition(MANAGED_START)
            _, _, after = tail.partition(MANAGED_END)
            new = before + block + after
        else:
            new = orig.rstrip() + "\n\n" + block
    else:
        banner = '"""Package initializer (auto-generated)"""\n\n'
        new = banner + block

    init_path.write_text(new, encoding="utf-8")
    logger.info(f"Wrote imports to {init_path}")

def main():
    parser = argparse.ArgumentParser(description="Create/update __init__.py with imports.")
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Target directory (default: directory of this script)."
    )
    args = parser.parse_args()

    pkg_dir = Path(args.dir).resolve() if args.dir else Path(__file__).resolve().parent
    if not pkg_dir.exists() or not pkg_dir.is_dir():
        logger.error(f"Directory does not exist: {pkg_dir}")
        raise SystemExit(1)

    logger.info(f"Generating __init__.py in {pkg_dir}")
    write_init(pkg_dir)

if __name__ == "__main__":
    main()
