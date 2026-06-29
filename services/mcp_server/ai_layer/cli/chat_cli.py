from __future__ import annotations

from ai_layer.gateway import ask_ai_sync


def main() -> None:
    print("PostgreSQL MCP AI Chat")
    print("Type 'exit' or 'quit' to stop.")
    print("")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        try:
            answer = ask_ai_sync(question)
            print("")
            print("AI:")
            print(answer)
            print("")
        except Exception as exc:
            print("")
            print("Error:")
            print(str(exc))
            print("")


if __name__ == "__main__":
    main()
