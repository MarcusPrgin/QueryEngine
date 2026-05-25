"""
QueryEngine: the main public interface.

Usage:
    engine = QueryEngine()
    engine.register("orders", "data/orders.csv")
    result = engine.query("SELECT country, SUM(total) FROM orders GROUP BY country")
    print(result.pretty())
"""
from __future__ import annotations
import time
import src.executor.executor as _executor_module
from src.types import QueryResult
from src.parser.parser import parse
from src.planner.planner import build_plan, explain
from src.executor.executor import execute
from src.catalog.table_catalog import TableCatalog
from src.index.sorted_index import catalog as index_catalog


class QueryEngine:
    def __init__(self, data_dir: str | None = None, optimize: bool = True):
        self.catalog = TableCatalog()
        self._optimize = optimize

        if data_dir:
            self.catalog.register_directory(data_dir)

    def register(self, name: str, path: str) -> "QueryEngine":
        """Register a file as a named table. Chainable."""
        self.catalog.register(name, path)
        return self

    def create_index(self, table: str, column: str) -> dict:
        """Build a sorted index on a column. Returns stats."""
        path = self.catalog.resolve(table)
        idx, elapsed = index_catalog.create(path, column)
        stats = idx.stats()
        stats["build_time_ms"] = round(elapsed * 1000, 1)
        return stats

    def query(self, sql: str, show_plan: bool = False) -> QueryResult:
        """
        Execute a SQL query and return a QueryResult.
        
        Pipeline:
          SQL string
            → Lexer → Token list
            → Parser → AST (SelectStatement)
            → Planner → Logical plan tree
            → Optimizer → Optimized plan tree
            → Executor → Row generator
            → Materialise → QueryResult
        """
        start = time.perf_counter()

        # 1. Parse
        stmt = parse(sql)

        # 2. Plan
        plan = build_plan(stmt, self.catalog)

        # 3. Optimize
        if self._optimize:
            from src.optimizer.optimizer import optimize as opt
            plan = opt(plan)

        # 4. Explain
        plan_str = explain(plan)

        if show_plan:
            print("\nQuery plan:")
            print(plan_str)
            print()

        # 5. Execute — materialise all rows, counting how many the Scan emitted
        rows_scanned = [0]
        _executor_module._scan_counter = rows_scanned
        try:
            rows = list(execute(plan))
        finally:
            _executor_module._scan_counter = None

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Infer schema from first row
        schema = {k: type(v).__name__ for k, v in rows[0].items()} if rows else {}

        return QueryResult(
            rows=rows,
            schema=schema,
            plan=plan_str,
            rows_scanned=rows_scanned[0],
            rows_returned=len(rows),
            elapsed_ms=round(elapsed_ms, 2),
        )

    def explain(self, sql: str) -> str:
        """Return the query plan without executing."""
        stmt = parse(sql)
        plan = build_plan(stmt, self.catalog)
        if self._optimize:
            from src.optimizer.optimizer import optimize as opt
            plan = opt(plan)
        return explain(plan)

    def schema(self, table: str) -> dict:
        """Return the inferred schema for a table."""
        return self.catalog.schema(table)

    def tables(self) -> list[dict]:
        """List all registered tables."""
        return self.catalog.list_tables()

    def indexes(self) -> list[dict]:
        """List all built indexes."""
        return index_catalog.list_indexes()
