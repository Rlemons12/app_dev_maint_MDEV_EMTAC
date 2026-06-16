from __future__ import annotations

from listed_server.mcp_coordinator.settings import CoordinatorSettings


class CapabilityRegistry:
    def __init__(self, settings: CoordinatorSettings) -> None:
        self.settings = settings

    def list_capabilities(self) -> list[dict[str, object]]:
        return [
            {"name": "postgres", "enabled": self.settings.postgres_mcp_enabled, "mode": "built_in"},
            {"name": "filesystem", "enabled": self.settings.filesystem_mcp.enabled, "mode": "downstream_placeholder"},
            {"name": "git", "enabled": self.settings.git_mcp.enabled, "mode": "downstream_placeholder"},
            {"name": "github", "enabled": self.settings.github_mcp.enabled, "mode": "downstream_placeholder"},
            {"name": "grafana", "enabled": self.settings.grafana_mcp.enabled, "mode": "downstream_stdio"},
            {"name": "browser", "enabled": self.settings.browser_mcp.enabled, "mode": "downstream_placeholder"},
            {"name": "emtac_api", "enabled": self.settings.emtac_api_mcp.enabled, "mode": "downstream_placeholder"},
            {"name": "memory", "enabled": self.settings.memory_mcp.enabled, "mode": "downstream_placeholder"},
        ]
