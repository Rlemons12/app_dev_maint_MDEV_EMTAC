"""
Validate MODEL_TRAINING directory structure.

- Confirms MODEL_TRAINING_BASE_DIR resolution
- Confirms intent_ner_training structure
- Prints existence status for all configured paths
- Safe to run from PyCharm or terminal
"""

from configuration.config import ALL_DIRS


def main():
    print("\nMODEL_TRAINING PATH VALIDATION")
    print("=" * 80)

    missing = []

    for path in ALL_DIRS:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"{status:8} | {path}")

        if not exists:
            missing.append(path)

    print("\nSUMMARY")
    print("=" * 80)

    if not missing:
        print("All required paths are present.")
    else:
        print(f"{len(missing)} paths are missing:")
        for m in missing:
            print(f"  - {m}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
