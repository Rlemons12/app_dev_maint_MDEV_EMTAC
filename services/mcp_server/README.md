# EMTAC MCP Coordinator

This project is now organized under `listed_server/` with:
- `listed_server.mcp_coordinator`
- `listed_server.mcp_postgres`
- `listed_server.mcp_grafana`

It exposes:
- coordinator tools for routing and capability inspection
- routing to a standalone PostgreSQL MCP server package (`listed_server.mcp_postgres`)
- optional routing to the official Grafana MCP server (`mcp-grafana`)
- placeholders for future downstream MCP servers (filesystem, git, github, browser, emtac_api, memory)

## Run

1. Create/update `.env` from `.env.example`.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Start the server:
   - `python -m listed_server.mcp_coordinator.server`
   - or compatibility entrypoint: `python server.py`

To run PostgreSQL MCP as a standalone server:
- `python -m listed_server.mcp_postgres.server`

To run Grafana MCP as a standalone wrapper around the official `mcp-grafana` server:
- `python -m listed_server.mcp_grafana.server`

## Coordinator tools

- `coordinator_health_check`
- `coordinator_list_capabilities`
- `coordinator_explain_route`
- `coordinator_route_request`

Safety behavior:
- `coordinator_explain_route` never executes.
- `coordinator_route_request(..., execute=False)` never executes.
- dangerous requests require `confirmed=True`.
- non-postgres capability execution is currently placeholder (`not_implemented`).

## PostgreSQL tools

PostgreSQL tools are served by `listed_server.mcp_postgres`, including:
- `postgres_health_check`, `postgres_whoami`
- database/schema/table management tools
- `postgres_read_query` (read-only validation + auto-limit)
- `postgres_write_execute`, `postgres_admin_execute`
- `postgres_insert_row`, `postgres_update_rows` (plus `postgres_delete_rows`)

## Configuration

See `.env.example` for:
- coordinator server name
- postgres role/connection values
- Grafana downstream MCP values
- per-capability downstream MCP wiring placeholders

For Grafana setup, see `README_GRAFANA_MCP.md`.

## Compatibility notes

- `mcp_server/server.py` now forwards to `listed_server.mcp_coordinator.server.main`.
- root `server.py` wrapper is provided for `python server.py`.

## TODO for downstream MCP wiring

- Implement real downstream transport in `listed_server/mcp_coordinator/clients/downstream_mcp_client.py`.
- Map routed requests to concrete tool arguments per downstream capability.
- Add authenticated/authorized execution policy per capability.
