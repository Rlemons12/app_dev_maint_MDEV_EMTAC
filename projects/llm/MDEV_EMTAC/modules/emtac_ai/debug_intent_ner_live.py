from __future__ import annotations

from pprint import pprint
from modules.emtac_ai.intent_ner.intent_orchestrator import IntentNEROrchestrator


def banner(text: str):
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def main():
    orch = IntentNEROrchestrator()

    banner("EMTAC INTENT → NER LIVE DEBUG (INTENT-DRIVEN)")
    print("Type a query and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        banner(f"QUERY: {query}")

        result = orch.process(query)

        # ---------------- Intent ----------------
        print("\n[ INTENT ]")
        print("-" * 40)
        print(f"Detected intent : {result['intent']}")
        print(f"Confidence      : {result['confidence']:.4f}")
        print(f"Primary domain  : {result['primary_category']}")

        # ---------------- NER -------------------
        print("\n[ NER OUTPUT ]")
        print("-" * 40)

        entities = result.get("entities", {})
        if not entities:
            print("(no entities extracted)")
        else:
            for label, values in entities.items():
                for v in values:
                    print(f"{label:<18} → {v}")

        # ---------------- Routing Info ----------
        print("\n[ ROUTING / EXPANSION ]")
        print("-" * 40)
        print(f"Expansion strategy : {result['expansion_strategy']}")
        print(f"Expand categories  : {result['expand_categories']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
