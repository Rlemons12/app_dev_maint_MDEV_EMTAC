import unittest

from listed_server.mcp_coordinator.postgres.sql_safety import ensure_limit, validate_read_only_sql


class SqlSafetyTests(unittest.TestCase):
    def test_read_only_sql_blocks_write_keyword(self) -> None:
        with self.assertRaises(ValueError):
            validate_read_only_sql("select * from users; delete from users")

    def test_ensure_limit_select_without_limit(self) -> None:
        result = ensure_limit("select * from users", 100)
        self.assertIn("LIMIT 100", result)

    def test_ensure_limit_with_cte(self) -> None:
        result = ensure_limit("with x as (select 1) select * from x", 10)
        self.assertIn("LIMIT 10", result)

if __name__ == "__main__":
    unittest.main()
