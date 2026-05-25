"""
QueryEngine: the main public interface.

Usage:
    engine = QueryEngine()
    engine.register("orders", "data/orders.csv")

    # Execute a query
    result = engine.query("SELECT country, SUM(total) FROM orders GROUP BY country")
    print(result.pretty())

    # EXPLAIN ANALYZE — run the query and get per-node timing
    result, stats = engine.analyze("SELECT * FROM orders WHERE total > 100")
    print(stats.pretty())
"""
from __future__ import annotations
import time
import src.executor.executor as _executor_module
from src.types import QueryResult, NodeStats
from src.parser.parser import parse
from src.planner.planner import build_plan, explain
from src.executor.executor import execute, analyze_execute
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

    # ── Plan helpers ───────────────────────────────────────────────────────

    def _build_optimized_plan(self, sql: str):
        stmt = parse(sql)
        plan = build_plan(stmt, self.catalog)
        if self._optimize:
            from src.optimizer.optimizer import optimize as opt
            plan = opt(plan)
        return plan, explain(plan)

    # ── Query ──────────────────────────────────────────────────────────────

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
        plan, plan_str = self._build_optimized_plan(sql)

        if show_plan:
            print("\nQuery plan:")
            print(plan_str)
            print()

        rows_scanned = [0]
        _executor_module._scan_counter = rows_scanned
        try:
            rows = list(execute(plan))
        finally:
            _executor_module._scan_counter = None

        elapsed_ms = (time.perf_counter() - start) * 1000
        schema = {k: type(v).__name__ for k, v in rows[0].items()} if rows else {}

        return QueryResult(
            rows=rows,
            schema=schema,
            plan=plan_str,
            rows_scanned=rows_scanned[0],
            rows_returned=len(rows),
            elapsed_ms=round(elapsed_ms, 2),
        )

    # ── EXPLAIN ANALYZE ────────────────────────────────────────────────────

    def analyze(self, sql: str) -> tuple[QueryResult, NodeStats]:
        """
        Execute a query and return both the result and a per-node stats tree.

        The NodeStats tree mirrors the plan structure. Each node records:
          - rows_in  : rows received from its child
          - rows_out : rows emitted to its parent
          - elapsed_ms : wall-clock time spent in this node (excluding children)

        Example::

            result, stats = engine.analyze("SELECT * FROM orders WHERE total > 100")
            print(stats.pretty())
            # Limit(100)  rows=100  selectivity=—  time=0.1ms
            #   Filter(total > 100)  rows=74823  selectivity=75%  time=38.2ms
            #     Scan(orders.csv, [*])  rows=100000  selectivity=—  time=31.4ms
        """
        start = time.perf_counter()
        plan, plan_str = self._build_optimized_plan(sql)

        rows_scanned = [0]
        _executor_module._scan_counter = rows_scanned
        try:
            gen, root_stats = analyze_execute(plan)
            rows = list(gen)
        finally:
            _executor_module._scan_counter = None

        elapsed_ms = (time.perf_counter() - start) * 1000
        schema = {k: type(v).__name__ for k, v in rows[0].items()} if rows else {}

        result = QueryResult(
            rows=rows,
            schema=schema,
            plan=plan_str,
            rows_scanned=rows_scanned[0],
            rows_returned=len(rows),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return result, root_stats

    # ── Introspection ──────────────────────────────────────────────────────

    def explain(self, sql: str) -> str:
        """Return the query plan without executing."""
        plan, plan_str = self._build_optimized_plan(sql)
        return plan_str

    def schema(self, table: str) -> dict:
        """Return the inferred schema for a table."""
        return self.catalog.schema(table)

    def tables(self) -> list[dict]:
        """List all registered tables."""
        return self.catalog.list_tables()

    def indexes(self) -> list[dict]:
        """List all built indexes."""
        return index_catalog.list_indexes()
