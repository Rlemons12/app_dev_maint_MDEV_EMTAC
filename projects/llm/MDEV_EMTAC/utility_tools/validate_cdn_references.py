import os
import re

# ---------------------------------------------------------
# Resolve the templates folder dynamically based on location
# ---------------------------------------------------------
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The root of scanning is the folder WHERE THIS SCRIPT LIVES
TEMPLATES_ROOT = THIS_SCRIPT_DIR

# File extensions considered template files
TEMPLATE_EXTENSIONS = (
    ".html", ".htm",
    ".jinja", ".jinja2",
    ".tpl", ".txt"
)

# Known CDN domains
CDN_PATTERNS = [
    r"https?://cdnjs\.cloudflare\.com",
    r"https?://cdn\.jsdelivr\.net",
    r"https?://cdn\.jsdelivr\.com",
    r"https?://cdn\.bootstrapcdn\.com",
    r"https?://stackpath\.bootstrapcdn\.com",
    r"https?://maxcdn\.bootstrapcdn\.com",
    r"https?://code\.jquery\.com",
    r"https?://fonts\.googleapis\.com",
    r"https?://fonts\.gstatic\.com",
    r"https?://unpkg\.com",
    r"https?://kit\.fontawesome\.com",
    r"https?://ajax\.googleapis\.com",
    r"https?://cdn\.fontawesome\.com",
    r"https?://gcore\.jsdelivr\.net",
]

CDN_REGEX = re.compile("|".join(CDN_PATTERNS))


# ---------------------------------------------------------
# Scan a single file
# ---------------------------------------------------------
def scan_file(path):
    results = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                if CDN_REGEX.search(line):
                    results.append((line_no, line.strip()))
    except Exception as e:
        print(f"[ERROR] Cannot read {path}: {e}")
    return results


# ---------------------------------------------------------
# Recursively scan ALL folders inside templates/
# ---------------------------------------------------------
def scan_templates_recursive():
    cdn_hits = []

    for root, dirs, files in os.walk(TEMPLATES_ROOT):
        for filename in files:

            # Skip this script itself
            if filename == os.path.basename(__file__):
                continue

            # Only scan template-like files
            if filename.lower().endswith(TEMPLATE_EXTENSIONS):
                full_path = os.path.join(root, filename)
                matches = scan_file(full_path)

                if matches:
                    cdn_hits.append((full_path, matches))

    return cdn_hits


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("===============================================")
    print(" CDN VALIDATOR — RUNNING INSIDE /templates/")
    print("===============================================")
    print(f"Scanning recursively from: {TEMPLATES_ROOT}\n")

    results = scan_templates_recursive()

    if not results:
        print("✔ SUCCESS — No CDN references found.")
        print("✔ Your templates folder is fully offline-ready.")
        return

    print("⚠ CDN REFERENCES FOUND — FIX REQUIRED ⚠\n")

    for filepath, matches in results:
        print(f"\nFILE: {filepath}")
        print("-" * (len(filepath) + 6))
        for line_num, text in matches:
            print(f"  Line {line_num}: {text}")

    print("\n===============================================")
    print(f"Summary: {len(results)} file(s) contain CDN references.")
    print("Replace with: {{ url_for('static', filename='vendor/...') }}")
    print("===============================================")


if __name__ == "__main__":
    main()
