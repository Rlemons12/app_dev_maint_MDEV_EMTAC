import os
import shutil

# Root folder where restoration must occur (the template folder)
TEMPLATE_ROOT = os.path.abspath(".")

def restore_from_bak():
    print("===============================================")
    print("   RESTORING ALL TEMPLATE FILES FROM .bak")
    print("===============================================")
    print(f"Template root: {TEMPLATE_ROOT}\n")

    restored = 0
    skipped = 0

    # Walk only inside templates folder
    for root, _, files in os.walk(TEMPLATE_ROOT):
        for fname in files:
            if fname.endswith(".bak"):
                bak_path = os.path.join(root, fname)
                original_path = os.path.join(root, fname[:-4])  # remove .bak

                print(f"[FOUND] Backup file: {bak_path}")

                # Check if original exists
                if os.path.exists(original_path):
                    print(f"  - Original exists. Replacing with backup...")
                    try:
                        os.remove(original_path)
                    except Exception as e:
                        print(f"  [ERROR] Unable to delete original: {e}")
                        skipped += 1
                        continue

                # Restore the backup
                try:
                    shutil.copy(bak_path, original_path)
                    print(f"  ✔ Restored → {original_path}")
                    restored += 1
                except Exception as e:
                    print(f"  [ERROR] Failed to restore backup: {e}")
                    skipped += 1

    print("\n===============================================")
    print(f"Restore Complete — {restored} file(s) restored from backup.")
    print(f"Skipped: {skipped} file(s).")
    print("===============================================")


if __name__ == "__main__":
    restore_from_bak()
