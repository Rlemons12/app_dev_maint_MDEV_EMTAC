from __future__ import annotations

# --------------------------------------------------
# Ensure project root is on PYTHONPATH (MUST BE FIRST)
# --------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Standard imports (SAFE after sys.path fix)
# --------------------------------------------------
from sqlalchemy import text
from modules.configuration.config_env import DatabaseConfig


# ============================================================
# Helpers
# ============================================================

def read_multiline() -> str:
    print("\nPaste prompt text.")
    print("Finish with Ctrl+Z + Enter (Windows) or Ctrl+D (Linux/macOS):\n")

    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass

    return "\n".join(lines).strip()


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


# ============================================================
# DB Session
# ============================================================

def _get_session():
    db = DatabaseConfig()
    return db.get_main_session()


# ============================================================
# DB Operations
# ============================================================

def list_prompts():
    session = _get_session()
    try:
        rows = session.execute(
            text("""
                SELECT id, purpose, prompt_name, is_active, created_at
                FROM ai_prompts
                ORDER BY purpose, created_at DESC
            """)
        ).fetchall()

        if not rows:
            print("No prompts found.")
            return

        for r in rows:
            print(
                f"ID={r.id:<4} | "
                f"purpose={r.purpose:<15} | "
                f"name={r.prompt_name:<20} | "
                f"active={str(r.is_active):<5} | "
                f"created={r.created_at}"
            )
    finally:
        session.close()


def show_active_prompt():
    purpose = input("Enter purpose (e.g. default): ").strip()
    if not purpose:
        print("Purpose required.")
        return

    session = _get_session()
    try:
        row = session.execute(
            text("""
                SELECT id, prompt_name, system_prompt
                FROM ai_prompts
                WHERE purpose = :purpose AND is_active = TRUE
                LIMIT 1
            """),
            {"purpose": purpose},
        ).fetchone()

        if not row:
            print("No active prompt found.")
            return

        print_header(
            f"Active Prompt | ID={row.id} | purpose={purpose} | name={row.prompt_name}"
        )
        print(row.system_prompt)
    finally:
        session.close()


def create_prompt():
    purpose = input("Enter purpose (e.g. default): ").strip()
    prompt_name = input("Enter prompt name (e.g. default_v1): ").strip()

    if not purpose or not prompt_name:
        print("Purpose and prompt name are required.")
        return

    prompt = read_multiline()
    if not prompt:
        print("Prompt cannot be empty.")
        return

    session = _get_session()
    try:
        # Deactivate existing active prompt for this purpose
        session.execute(
            text("""
                UPDATE ai_prompts
                SET is_active = FALSE, updated_at = NOW()
                WHERE purpose = :purpose AND is_active = TRUE
            """),
            {"purpose": purpose},
        )

        # Insert new prompt
        session.execute(
            text("""
                INSERT INTO ai_prompts
                    (purpose, prompt_name, system_prompt, is_active, created_at, updated_at)
                VALUES
                    (:purpose, :prompt_name, :prompt, TRUE, NOW(), NOW())
            """),
            {
                "purpose": purpose,
                "prompt_name": prompt_name,
                "prompt": prompt,
            },
        )

        session.commit()
        print("Prompt inserted and activated.")

    except Exception as e:
        session.rollback()
        print(f"ERROR: {e}")
    finally:
        session.close()


def activate_prompt():
    prompt_id = input("Enter prompt ID to activate: ").strip()
    if not prompt_id.isdigit():
        print("Invalid ID.")
        return

    session = _get_session()
    try:
        row = session.execute(
            text("""
                SELECT purpose
                FROM ai_prompts
                WHERE id = :id
            """),
            {"id": int(prompt_id)},
        ).fetchone()

        if not row:
            print("Prompt not found.")
            return

        purpose = row.purpose

        session.execute(
            text("""
                UPDATE ai_prompts
                SET is_active = FALSE, updated_at = NOW()
                WHERE purpose = :purpose
            """),
            {"purpose": purpose},
        )

        session.execute(
            text("""
                UPDATE ai_prompts
                SET is_active = TRUE, updated_at = NOW()
                WHERE id = :id
            """),
            {"id": int(prompt_id)},
        )

        session.commit()
        print("Prompt activated.")

    except Exception as e:
        session.rollback()
        print(f"ERROR: {e}")
    finally:
        session.close()


def deactivate_prompt():
    prompt_id = input("Enter prompt ID to deactivate: ").strip()
    if not prompt_id.isdigit():
        print("Invalid ID.")
        return

    session = _get_session()
    try:
        session.execute(
            text("""
                UPDATE ai_prompts
                SET is_active = FALSE, updated_at = NOW()
                WHERE id = :id
            """),
            {"id": int(prompt_id)},
        )
        session.commit()
        print("Prompt deactivated.")

    except Exception as e:
        session.rollback()
        print(f"ERROR: {e}")
    finally:
        session.close()


def delete_prompt():
    prompt_id = input("Enter prompt ID to DELETE: ").strip()
    if not prompt_id.isdigit():
        print("Invalid ID.")
        return

    confirm = input("Type DELETE to confirm: ")
    if confirm != "DELETE":
        print("Cancelled.")
        return

    session = _get_session()
    try:
        session.execute(
            text("DELETE FROM ai_prompts WHERE id = :id"),
            {"id": int(prompt_id)},
        )
        session.commit()
        print("Prompt deleted.")

    except Exception as e:
        session.rollback()
        print(f"ERROR: {e}")
    finally:
        session.close()


# ============================================================
# CLI Loop
# ============================================================

def main():
    while True:
        print_header("EMTAC Prompt Admin")

        print("1) List prompts")
        print("2) Show active prompt")
        print("3) Create new prompt")
        print("4) Activate prompt")
        print("5) Deactivate prompt")
        print("6) Delete prompt")
        print("0) Exit\n")

        choice = input("Select option: ").strip()

        if choice == "1":
            list_prompts()
        elif choice == "2":
            show_active_prompt()
        elif choice == "3":
            create_prompt()
        elif choice == "4":
            activate_prompt()
        elif choice == "5":
            deactivate_prompt()
        elif choice == "6":
            delete_prompt()
        elif choice == "0":
            sys.exit(0)
        else:
            print("Invalid option.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
