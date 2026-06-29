from __future__ import annotations

import os
import subprocess
import sys

from listed_server.mcp_coordinator.settings import get_settings


def main() -> None:
    settings = get_settings()
    config = settings.grafana_mcp

    if not config.command:
        raise RuntimeError("GRAFANA_MCP_COMMAND is not configured.")

    env = os.environ.copy()
    env.update(config.env)

    command = [config.command, *config.args]
    completed = subprocess.run(command, env=env, check=False)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
