from __future__ import annotations
from typing import Optional, Dict

from sqlalchemy import text

from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
)

# ============================================================
# PromptService
# ============================================================

class PromptService:
    """
    PromptService is responsible ONLY for retrieving prompts
    from the database.

    - No prompt formatting
    - No AI execution
    - No business logic
    """

    @staticmethod
    def get_active_prompt(
        *,
        purpose: str,
        request_id: Optional[str] = None,
    ) -> str:
        sql = text("""
            SELECT system_prompt
            FROM ai_prompts
            WHERE purpose = :purpose
              AND is_active = TRUE
            LIMIT 1
        """)

        db = DatabaseConfig()

        try:
            with db.main_session() as session:
                row = session.execute(
                    sql,
                    {"purpose": purpose},
                ).fetchone()

            if not row:
                raise RuntimeError(
                    f"No active prompt found for purpose='{purpose}'"
                )

            info_id(
                f"[PromptService] Loaded active prompt (purpose={purpose})",
                request_id,
            )

            return row[0]

        except Exception as e:
            error_id(
                f"[PromptService] Failed to load prompt ({purpose}): {e}",
                request_id,
            )
            raise


# ============================================================
# PromptProvider
# ============================================================

class PromptProvider:
    """
    PromptProvider assembles the final prompt sent to the AI.

    Responsibilities:
    - Load system prompt (via PromptService)
    - Assemble context + question + answer scaffold
    - Enforce consistent structure
    """

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    @classmethod
    def build_prompt(
        cls,
        *,
        purpose: str,
        question: str,
        context: Optional[str] = None,
        request_id: Optional[str] = None,
        extra_sections: Optional[Dict[str, str]] = None,
    ) -> str:
        debug_id(
            f"[PromptProvider] Building prompt (purpose={purpose})",
            request_id,
        )

        system_prompt = cls._load_system_prompt(
            purpose=purpose,
            request_id=request_id,
        )

        prompt = cls._assemble_prompt(
            system_prompt=system_prompt,
            question=question,
            context=context,
            extra_sections=extra_sections,
        )

        info_id(
            f"[PromptProvider] Prompt built (length={len(prompt)})",
            request_id,
        )

        return prompt

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    @staticmethod
    def _load_system_prompt(
        *,
        purpose: str,
        request_id: Optional[str] = None,
    ) -> str:
        try:
            return PromptService.get_active_prompt(
                purpose=purpose,
                request_id=request_id,
            )
        except Exception:
            warning_id(
                f"[PromptProvider] Falling back to default system prompt "
                f"(purpose={purpose})",
                request_id,
            )
            return PromptProvider._default_system_prompt()

    @staticmethod
    def _assemble_prompt(
        *,
        system_prompt: str,
        question: str,
        context: Optional[str],
        extra_sections: Optional[Dict[str, str]],
    ) -> str:
        parts = []

        # --- System prompt ---
        parts.append(system_prompt.strip())

        # --- Context ---
        parts.append(
            "\n--- CONTEXT START ---\n"
            + (context.strip() if context else "No relevant context provided.")
            + "\n--- CONTEXT END ---\n"
        )

        # --- Optional extra sections ---
        if extra_sections:
            for title, content in extra_sections.items():
                if content:
                    parts.append(
                        f"\n--- {title.upper()} ---\n"
                        f"{content.strip()}\n"
                    )

        # --- Question / Answer ---
        parts.append(
            "\nQUESTION:\n"
            + question.strip()
            + "\n\nANSWER:\n"
        )

        return "\n".join(parts)

    # ---------------------------------------------------------
    # Hard fallback (last resort)
    # ---------------------------------------------------------
    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are an EMTAC maintenance and engineering assistant.\n\n"
            "Provide a clear, practical answer to the user's question.\n\n"
            "Use provided context only if it is relevant.\n"
            "If context is incomplete or missing, rely on general "
            "engineering and maintenance best practices.\n\n"
            "Do NOT invent specific values, limits, or procedures.\n"
            "Say 'I do not know' only if the question truly cannot be answered.\n"
        )
