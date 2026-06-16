from __future__ import annotations

from clients.postgres_mcp_ui_client import PostgresMcpUiClient


def main() -> None:
    client = PostgresMcpUiClient()

    print("Settings:")
    print(client.settings())

    print("")
    print("AI answer:")
    print(client.ask("List my PostgreSQL databases."))


if __name__ == "__main__":
    main()
