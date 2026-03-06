"""
Service Registry Builder (Improved)

Fixes:
- Proper newline handling
- Proper acronym handling (AI, VLM, UI)
- Clean snake_case conversion
- Sorted output
"""

import re
from pathlib import Path


CURRENT_DIR = Path(__file__).parent
OUTPUT_FILE = CURRENT_DIR / "registry.py"

SERVICE_CLASS_PATTERN = re.compile(r"class\s+(\w+Service)\b")


ACRONYM_FIXES = {
    "AI": "ai",
    "VLM": "vlm",
    "UI": "ui",
}


def find_service_classes():
    services = []

    for file in CURRENT_DIR.glob("*.py"):
        if file.name in {
            "registry.py",
            "build_service_registry.py",
            "__init__.py",
        }:
            continue

        content = file.read_text(encoding="utf-8")

        matches = SERVICE_CLASS_PATTERN.findall(content)
        for match in matches:
            services.append(
                {
                    "class_name": match,
                    "file_name": file.stem,
                }
            )

    return sorted(services, key=lambda x: x["class_name"])


def normalize_acronyms(name: str) -> str:
    for key, val in ACRONYM_FIXES.items():
        name = name.replace(key, val.capitalize())
    return name


def to_property_name(class_name: str) -> str:
    base = class_name[:-7]  # Remove "Service"

    base = normalize_acronyms(base)

    # CamelCase → snake_case
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()

    return snake


def build_registry_code(services):
    import_lines = []
    init_fields = []
    property_blocks = []

    for svc in services:
        class_name = svc["class_name"]
        file_name = svc["file_name"]
        prop_name = to_property_name(class_name)

        import_lines.append(
            f"from modules.services.{file_name} import {class_name}"
        )

        init_fields.append(f"        self._{prop_name} = None")

        property_blocks.append(f"""
    @property
    def {prop_name}(self) -> {class_name}:
        if self._{prop_name} is None:
            self._{prop_name} = {class_name}()
        return self._{prop_name}
""")

    # PRECOMPUTE STRINGS (fix for f-string backslash issue)
    imports_block = "\n".join(import_lines)
    init_block = "\n".join(init_fields)
    properties_block = "".join(property_blocks)

    registry_code = f'''"""
AUTO-GENERATED SERVICE REGISTRY

DO NOT EDIT MANUALLY.
Run build_service_registry.py to regenerate.
"""

from typing import Optional

{imports_block}


class ServiceRegistry:

    def __init__(self):
{init_block}

{properties_block}


# Global singleton
services = ServiceRegistry()
'''

    return registry_code



def main():
    print("Scanning for services...")

    services = find_service_classes()

    if not services:
        print("No services found.")
        return

    print(f"Found {len(services)} service(s):")
    for s in services:
        print(f"  - {s['class_name']}")

    code = build_registry_code(services)

    OUTPUT_FILE.write_text(code, encoding="utf-8")

    print(f"\nRegistry generated at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
