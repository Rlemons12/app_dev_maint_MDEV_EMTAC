import os
import re

# ------------------------------------------------------------
# Run INSIDE the templates directory
# ------------------------------------------------------------
TEMPLATE_ROOT = "."


# ------------------------------------------------------------
# CDN → vendor replacement mapping
# ------------------------------------------------------------
REPLACEMENTS = {
    # -----------------------
    # jQuery
    # -----------------------
    r"https?://code\.jquery\.com/jquery-3\.6\.0\.min\.js":
        "vendor/jquery/jquery-3.6.0.min.js",

    r"https?://ajax\.googleapis\.com/ajax/libs/jquery/3\.5\.1/jquery\.min\.js":
        "vendor/jquery/jquery-3.5.1.min.js",

    r"https?://ajax\.googleapis\.com/ajax/libs/jquery/3\.6\.0/jquery\.min\.js":
        "vendor/jquery/jquery-3.6.0.min.js",

    # -----------------------
    # Bootstrap (jsDelivr + gcore + cdnjs)
    # -----------------------
    r"https?://cdn\.jsdelivr\.net/npm/bootstrap@5\.3\.0/dist/css/bootstrap\.min\.css":
        "vendor/bootstrap/bootstrap.min.css",

    r"https?://cdn\.jsdelivr\.net/npm/bootstrap@5\.3\.0/dist/js/bootstrap\.bundle\.min\.js":
        "vendor/bootstrap/bootstrap.bundle.min.js",

    r"https?://gcore\.jsdelivr\.net/npm/bootstrap@5\.3\.0/dist/css/bootstrap\.min\.css":
        "vendor/bootstrap/bootstrap.min.css",

    r"https?://gcore\.jsdelivr\.net/npm/bootstrap@5\.3\.0/dist/js/bootstrap\.bundle\.min\.js":
        "vendor/bootstrap/bootstrap.bundle.min.js",

    r"https?://cdnjs\.cloudflare\.com/ajax/libs/bootstrap/5\.3\.0/css/bootstrap\.min\.css":
        "vendor/bootstrap/bootstrap.min.css",

    r"https?://cdnjs\.cloudflare\.com/ajax/libs/bootstrap/5\.3\.0/js/bootstrap\.bundle\.min\.js":
        "vendor/bootstrap/bootstrap.bundle.min.js",

    # protocol-relative versions
    r"//cdnjs\.cloudflare\.com/ajax/libs/bootstrap/5\.3\.0/css/bootstrap\.min\.css":
        "vendor/bootstrap/bootstrap.min.css",

    r"//cdnjs\.cloudflare\.com/ajax/libs/bootstrap/5\.3\.0/js/bootstrap\.bundle\.min\.js":
        "vendor/bootstrap/bootstrap.bundle.min.js",

    # -----------------------
    # Select2 (4.1.0 + 4.0.13)
    # -----------------------
    r"https?://cdnjs\.cloudflare\.com/ajax/libs/select2/4\.1\.0-rc\.0/css/select2\.min\.css":
        "vendor/select2/select2.min.css",

    r"https?://cdnjs\.cloudflare\.com/ajax/libs/select2/4\.1\.0-rc\.0/js/select2\.min\.js":
        "vendor/select2/select2.min.js",

    r"https?://cdnjs\.cloudflare\.com/ajax/libs/select2/4\.0\.13/css/select2\.min\.css":
        "vendor/select2/select2_v4_0_13.min.css",

    r"https?://cdnjs\.cloudflare\.com/ajax/libs/select2/4\.0\.13/js/select2\.min\.js":
        "vendor/select2/select2_v4_0_13.min.js",

    # generic catch-all (any Select2 version)
    r"https?://cdnjs\.cloudflare\.com/ajax/libs/select2/[^\"']+/css/select2\.min\.css":
        "vendor/select2/select2.min.css",

    r"https?://cdnjs\.cloudflare\.com/ajax/libs/select2/[^\"']+/js/select2\.min\.js":
        "vendor/select2/select2.min.js",

    # protocol-relative Select2
    r"//cdnjs\.cloudflare\.com/ajax/libs/select2/[^\"']+/css/select2\.min\.css":
        "vendor/select2/select2.min.css",

    r"//cdnjs\.cloudflare\.com/ajax/libs/select2/[^\"']+/js/select2\.min\.js":
        "vendor/select2/select2.min.js",

    # -----------------------
    # IntroJS
    # -----------------------
    r"https?://cdnjs\.cloudflare\.com/ajax/libs/intro\.js/6\.0\.0/introjs\.min\.css":
        "vendor/introjs/introjs.min.css",

    r"https?://cdnjs\.cloudflare\.com/ajax/libs/intro\.js/6\.0\.0/introjs-rtl\.min\.css":
        "vendor/introjs/introjs-rtl.min.css",

    r"https?://cdnjs\.cloudflare\.com/ajax/libs/intro\.js/6\.0\.0/intro\.min\.js":
        "vendor/introjs/intro.min.js",

    # -----------------------
    # Toastr
    # -----------------------
    r"https?://cdnjs\.cloudflare\.com/ajax/libs/toastr\.js/latest/toastr\.min\.css":
        "vendor/toastr/toastr.min.css",

    r"https?://cdnjs\.cloudflare\.com/ajax/libs/toastr\.js/latest/toastr\.min\.js":
        "vendor/toastr/toastr.min.js",

    # -----------------------
    # Font Awesome
    # -----------------------
    r"https?://cdnjs\.cloudflare\.com/ajax/libs/font-awesome/6\.0\.0/css/all\.min\.css":
        "vendor/fontawesome/all.min.css",

    # -----------------------
    # Dropzone
    # -----------------------
    r"https?://cdnjs\.cloudflare\.com/ajax/libs/dropzone/5\.9\.3/min/dropzone\.min\.css":
        "vendor/dropzone/dropzone.min.css",

    # -----------------------
    # Material Icons (NEW)
    # -----------------------
    r"https?://fonts\.googleapis\.com/icon\?family=Material\+Icons":
        "vendor/material_icons/material-icons.css",

    # protocol-relative Material Icons
    r"//fonts\.googleapis\.com/icon\?family=Material\+Icons":
        "vendor/material_icons/material-icons.css",

    # Select2 — jsDelivr version (fix for missing pattern)
    r"https?://cdn\.jsdelivr\.net/npm/select2@[^\"']+/dist/css/select2\.min\.css":
        "vendor/select2/select2.min.css",

    r"https?://cdn\.jsdelivr\.net/npm/select2@[^\"']+/dist/js/select2\.min\.js":
        "vendor/select2/select2.min.js",

}


# Remaining CDN markers
CDN_MARKERS = [
    "cdnjs.cloudflare.com",
    "cdn.jsdelivr.net",
    "gcore.jsdelivr.net",
    "code.jquery.com",
    "ajax.googleapis.com",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
]


# ------------------------------------------------------------
# Brace Cleanup (Option 2)
# ------------------------------------------------------------
def cleanup_braces(line: str) -> str:
    return re.sub(r"\}{3,}", "}}", line)


# ------------------------------------------------------------
# Restore .bak files
# ------------------------------------------------------------
def restore_bak_files():
    print("\nChecking for .bak backups to restore...")
    restored = 0

    for root, _, files in os.walk(TEMPLATE_ROOT):
        for file in files:
            if not file.endswith(".bak"):
                continue

            src = os.path.join(root, file)
            dst = src[:-4]

            print(f"[RESTORE] {dst}")
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
            restored += 1

    print(
        "✔ Restored {} file(s).".format(restored)
        if restored else "No .bak files found."
    )


# ------------------------------------------------------------
# Apply replacements to a single file
# ------------------------------------------------------------
def fix_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    modified = []
    changes = 0

    for original_line in lines:
        line = original_line
        replaced = False

        for pattern, vendor_path in REPLACEMENTS.items():
            if re.search(pattern, line):

                changes += 1
                modified.append(f"<!-- ORIGINAL CDN: {original_line.strip()} -->\n")

                stripped = original_line.strip()

                if stripped.startswith("<link"):
                    new_line = (
                        f'<link rel="stylesheet" '
                        f'href="{{{{ url_for(\'static\', filename=\'{vendor_path}\') }}}}"/>\n'
                    )
                elif stripped.startswith("<script"):
                    new_line = (
                        f'<script src="{{{{ url_for(\'static\', filename=\'{vendor_path}\') }}}}"></script>\n'
                    )
                else:
                    new_line = (
                        f'{{{{ url_for(\'static\', filename=\'{vendor_path}\') }}}}\n'
                    )

                modified.append(cleanup_braces(new_line))
                replaced = True
                break

        if not replaced:
            modified.append(cleanup_braces(line))

    if changes == 0:
        return 0

    backup = path + ".bak"
    if os.path.exists(backup):
        os.remove(backup)
    os.replace(path, backup)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(modified)

    print(f"[UPDATED] {path} — {changes} changes")
    return changes


# ------------------------------------------------------------
# Scan for remaining CDN references
# ------------------------------------------------------------
def scan_for_cdn_links():
    print("\nScanning for remaining CDN links...")
    remaining = 0

    for root, _, files in os.walk(TEMPLATE_ROOT):
        for name in files:
            if not name.endswith((".html", ".jinja", ".jinja2", ".tpl", ".htm")):
                continue

            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    if any(marker in line for marker in CDN_MARKERS):
                        remaining += 1
                        print(f"[CDN] {path} (line {i}): {line.strip()}")

    print(
        f"⚠ Found {remaining} remaining CDN line(s)."
        if remaining else "✔ No remaining CDN references."
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("\n===============================================")
    print("   EXTENDED AUTO-FIXER: CDN → OFFLINE VENDOR")
    print("===============================================\n")

    restore_bak_files()

    total = 0
    for root, _, files in os.walk(TEMPLATE_ROOT):
        for name in files:
            if name.endswith((".html", ".jinja", ".jinja2", ".tpl", ".htm")):
                total += fix_file(os.path.join(root, name))

    print("\n===============================================")
    print(f" Completed — Replaced CDN links: {total}")
    print("===============================================\n")

    scan_for_cdn_links()


if __name__ == "__main__":
    main()
