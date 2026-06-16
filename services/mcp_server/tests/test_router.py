import unittest

from listed_server.mcp_coordinator.routing.router import CoordinatorRouter


class RouterTests(unittest.TestCase):
    def test_route_postgres_terms(self) -> None:
        decision = CoordinatorRouter().route("show me sql chat audit tables")
        self.assertEqual(decision.target_capability, "postgres")

    def test_route_filesystem_terms(self) -> None:
        decision = CoordinatorRouter().route("open file manual pdf log")
        self.assertEqual(decision.target_capability, "filesystem")

    def test_route_git_terms(self) -> None:
        decision = CoordinatorRouter().route("git diff commit status")
        self.assertEqual(decision.target_capability, "git")

    def test_route_grafana_terms(self) -> None:
        decision = CoordinatorRouter().route("search grafana dashboards")
        self.assertEqual(decision.target_capability, "grafana")
        self.assertEqual(decision.target_tool, "search_dashboards")
        self.assertEqual(decision.suggested_arguments, {"query": "search grafana dashboards"})

    def test_route_grafana_datasources_without_arguments(self) -> None:
        decision = CoordinatorRouter().route("list grafana datasources")
        self.assertEqual(decision.target_capability, "grafana")
        self.assertEqual(decision.target_tool, "list_datasources")
        self.assertEqual(decision.suggested_arguments, {})

    def test_route_grafana_dashboard_terms_win_over_database_terms(self) -> None:
        decision = CoordinatorRouter().route("count dashboards in the default PostgreSQL database")
        self.assertEqual(decision.target_capability, "grafana")
        self.assertEqual(decision.target_tool, "search_dashboards")

    def test_route_create_blank_grafana_dashboard(self) -> None:
        decision = CoordinatorRouter().route('create a blank Grafana dashboard called "MCP"')
        self.assertEqual(decision.target_capability, "grafana")
        self.assertEqual(decision.target_tool, "update_dashboard")
        self.assertEqual(decision.suggested_arguments["dashboard"]["title"], "MCP")
        self.assertEqual(decision.suggested_arguments["dashboard"]["uid"], "mcp")

    def test_dangerous_requires_confirmation(self) -> None:
        decision = CoordinatorRouter().route("drop database emtac")
        self.assertTrue(decision.needs_confirmation)


if __name__ == "__main__":
    unittest.main()
