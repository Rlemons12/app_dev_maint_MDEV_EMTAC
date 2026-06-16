"""
PgVectorIntentEngine
--------------------

High-accuracy intent classifier using pgvector similarity search.

This engine:
    - Embeds the user query using any embedding model (MiniLM or others)
    - Performs a <=> cosine distance search inside PostgreSQL
    - Returns the closest intent anchor

Requirements:
    The intent_anchor table must contain:
        id (int)
        intent (str)
        anchor_text (str)
        embedding (vector)
"""

import numpy as np
from sqlalchemy import text as sql_text


class PgVectorIntentEngine:
    def __init__(self, session_factory, embedder):
        """
        session_factory:
            Callable that returns a new SQLAlchemy session

        embedder:
            Any object with .encode(text) -> vector list or numpy array
        """
        self.session_factory = session_factory
        self.embedder = embedder

    # ----------------------------------------------------------------------
    # MAIN INTENT DETECTION METHOD
    # ----------------------------------------------------------------------
    def detect_intent(self, query_text: str) -> dict:
        """
        Returns:
            {
                "intent": str,
                "confidence": float
            }
        """

        # ------------------------------------
        # 1. GET QUERY EMBEDDING
        # ------------------------------------
        try:
            vec = self.embedder.encode(query_text)

            # Convert numpy array → list
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()

        except Exception as e:
            print(f"[PgVectorIntentEngine] ERROR: Could not embed text: {e}")
            return {"intent": "general_chat", "confidence": 0.0}

        # ------------------------------------
        # 2. SQL VECTOR SEARCH
        # ------------------------------------
        sql = sql_text("""
            SELECT intent,
                   1 - (embedding <=> :vec) AS confidence
            FROM intent_anchor
            ORDER BY embedding <=> :vec ASC
            LIMIT 1;
        """)

        session = self.session_factory()

        try:
            row = session.execute(sql, {"vec": vec}).fetchone()

            # No rows → fallback
            if not row:
                return {"intent": "general_chat", "confidence": 0.0}

            # -----------------------------------------------------
            # ROW PARSING — SUPPORT ALL TEST FORMATS:
            #   - SQLAlchemy Row (row.intent)
            #   - FakeRow object (.intent)
            #   - dict-like row
            #   - tuple/list row
            # -----------------------------------------------------

            # Attribute-style (SQLAlchemy / FakeRow)
            intent = getattr(row, "intent", None)
            confidence = getattr(row, "confidence", None)

            # Dict-like row
            if intent is None and isinstance(row, dict):
                intent = row.get("intent")
                confidence = row.get("confidence")

            # Tuple/list row (index 0 = intent, index 1 = confidence)
            if intent is None and isinstance(row, (tuple, list)):
                if len(row) >= 2:
                    intent, confidence = row[0], row[1]

            # If still missing → fail gracefully
            if intent is None or confidence is None:
                raise ValueError("Row missing expected intent/confidence fields")

            return {
                "intent": intent,
                "confidence": float(confidence)
            }

        except Exception as e:
            print(f"[PgVectorIntentEngine] ERROR running query: {e}")
            return {"intent": "general_chat", "confidence": 0.0}

        finally:
            session.close()
