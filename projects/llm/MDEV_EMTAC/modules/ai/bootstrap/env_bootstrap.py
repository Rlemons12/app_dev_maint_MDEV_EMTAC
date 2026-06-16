"""Environment bootstrap helpers extracted from legacy ai_models.py."""


from __future__ import annotations


from pathlib import Path

DEV_ENV = Path(r"E:\emtac\dev_env\.env")


PROJECT_ENV = Path(__file__).parent / ".env"
