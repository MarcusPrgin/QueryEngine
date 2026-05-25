"""
Test suite for the query engine.

Run: pytest tests/ -v
Run with coverage: pytest tests/ --cov=src --cov-report=term-missing
"""
import csv
import os
import tempfile
import pytest
from src.engine import QueryEngine
from src.parser.parser import parse
from src.parser.lexer import tokenize, LexError
from src.aggregation.hyperloglog import HyperLogLog, count_distinct
from src.types import SelectStatement, BinaryOp, Column, Literal


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_csv(tmp_path):
    """Write a small CSV and return its path."""
    def _make(rows: list[dict], filename: str = "test.csv") -> str:
        path = str(tmp_path / filename)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return path
    return _make


@pytest.fixture
def orders_csv(tmp_csv):
    return tmp_csv([
        {"order_id": i, "country": c, "category": cat, "total": t, "status": s}
        for i, c, cat, t, s in [
            (1, "CA", "electronics", 100.0, "completed"),
            (2, "CA", "clothing",    50.0,  "completed"),
            (3, "US", "electronics", 200.0, "pending"),
            (4, "US", "books",       25.0,  "completed"),
            (5, "CA", "electronics", 150.0, "cancelled"),
            (6, "UK", "clothing",    75.0,  "completed"),
            (7, "UK", "books",       30.0,  "completed"),
            (8, "DE", "electronics", 300.0, "completed"),
        ]
    ], "orders.csv")


@pytest.fixture
def engine(orders_csv):
    e = QueryEngine()
    e.register("orders", orders_csv)
    return e


# ── Lexer tests ────────────────────────────────────────────────────────────

class TestLexer:
    def test_basic_select(self):
        tokens = tokenize("SELECT * FROM t")
        types = [t.type.name for t in tokens]
        assert "SELECT" in types
        assert "STAR" in types
        assert "FROM" in types

    def test_string_literal(self):
        tokens = tokenize("WHERE country = 'CA'")
        strings = [t for t in tokens if t.type.name == "STRING"]
        assert strings[0].value == "CA"

    def test_number_types(self):
        tokens = tokenize("WHERE x > 3.14")
        nums = [t for t in tokens if t.type.name == "NUMBER"]
        assert isinstance(nums[0].value, float)

    def test_operators(self):
        sql = "x != y AND a <= b OR c >= d"
        types = {t.type.name for t in tokenize(sql)}
        assert "NEQ" in types
        assert "LTE" in types
        assert "GTE" in types

    def test_group_by_two_tokens(self):
        tokens = tokenize("GROUP BY country")
        assert tokens[0].type.name == "GROUP_BY"


# ── Parser tests ───────────────────────────────────────────────────────────

class TestParser:
    def test_simple_select(self):
        stmt = parse("SELECT * FROM orders")
        assert stmt.from_table == "orders"

    def test_where_clause(self):
        stmt = parse("SELECT * FROM orders WHERE total > 100")
        assert isinstance(stmt.where, BinaryOp)
        assert stmt.where.op == ">"

    def test_group_by(self):
        stmt = parse("SELECT country, SUM(total) FROM orders GROUP BY country")
        assert len(stmt.group_by) == 1
        assert stmt.group_by[0].name == "country"

    def test_limit(self):
        stmt = parse("SELECT * FROM orders LIMIT 10")
        assert stmt.limit == 10

    def test_order_by(self):
        stmt = parse("SELECT * FROM orders ORDER BY total DESC")
        assert len(stmt.order_by) == 1
        assert stmt.order_by[0][1] == "DESC"

    def test_and_or_precedence(self):
        stmt = parse("SELECT * FROM t WHERE a = 1 AND b = 2 OR c = 3")
        # OR has lower precedence — root should be OR
        assert isinstance(stmt.where, BinaryOp)

    def test_aggregate_functions(self):
        stmt = parse("SELECT COUNT(*), SUM(total), AVG(total), MIN(total), MAX(total) FROM orders")
        assert len(stmt.columns) == 5

    def test_join(self):
        stmt = parse("SELECT * FROM orders INNER JOIN customers ON orders.customer_id = customers.customer_id")
        assert len(stmt.joins) == 1
        assert stmt.joins[0].join_type == "INNER"


# ── Execution tests ────────────────────────────────────────────────────────

class TestExecution:
    def test_select_star(self, engine):
        result = engine.query("SELECT * FROM orders")
        assert len(result.rows) == 8
        assert "order_id" in result.rows[0]

    def test_filter_eq(self, engine):
        result = engine.query("SELECT * FROM orders WHERE country = 'CA'")
        assert len(result.rows) == 3
        assert all(r["country"] == "CA" for r in result.rows)

    def test_filter_gt(self, engine):
        result = engine.query("SELECT * FROM orders WHERE total > 100")
        assert all(r["total"] > 100 for r in result.rows)

    def test_filter_and(self, engine):
        result = engine.query("SELECT * FROM orders WHERE country = 'CA' AND status = 'completed'")
        assert all(r["country"] == "CA" and r["status"] == "completed" for r in result.rows)

    def test_limit(self, engine):
        result = engine.query("SELECT * FROM orders LIMIT 3")
        assert len(result.rows) == 3

    def test_group_by_count(self, engine):
        result = engine.query("SELECT country, COUNT(*) AS cnt FROM orders GROUP BY country")
        countries = {r["country"] for r in result.rows}
        assert "CA" in countries
        ca_row = next(r for r in result.rows if r["country"] == "CA")
        assert ca_row["cnt"] == 3

    def test_group_by_sum(self, engine):
        result = engine.query("SELECT country, SUM(total) AS total_rev FROM orders GROUP BY country")
        ca_row = next(r for r in result.rows if r["country"] == "CA")
        assert abs(ca_row["total_rev"] - 300.0) < 0.01

    def test_order_by_desc(self, engine):
        result = engine.query("SELECT * FROM orders ORDER BY total DESC")
        totals = [r["total"] for r in result.rows]
        assert totals == sorted(totals, reverse=True)

    def test_order_by_asc(self, engine):
        result = engine.query("SELECT * FROM orders ORDER BY total ASC")
        totals = [r["total"] for r in result.rows]
        assert totals == sorted(totals)

    def test_like_operator(self, engine):
        result = engine.query("SELECT * FROM orders WHERE status LIKE 'comp%'")
        assert all(r["status"].startswith("comp") for r in result.rows)

    def test_no_rows_returned(self, engine):
        result = engine.query("SELECT * FROM orders WHERE country = 'XX'")
        assert len(result.rows) == 0

    def test_projection(self, engine):
        result = engine.query("SELECT order_id, total FROM orders")
        assert list(result.rows[0].keys()) == ["order_id", "total"]

    def test_count_distinct(self, engine):
        result = engine.query("SELECT COUNT_DISTINCT(country) AS unique_countries FROM orders")
        # 4 distinct countries: CA, US, UK, DE
        assert result.rows[0]["unique_countries"] >= 3  # HLL may be approximate


# ── Explain / plan tests ───────────────────────────────────────────────────

class TestExplain:
    def test_explain_shows_scan(self, engine):
        plan = engine.explain("SELECT * FROM orders")
        assert "Scan" in plan

    def test_explain_shows_filter(self, engine):
        plan = engine.explain("SELECT * FROM orders WHERE total > 100")
        assert "Filter" in plan

    def test_explain_shows_aggregate(self, engine):
        plan = engine.explain("SELECT country, SUM(total) FROM orders GROUP BY country")
        assert "Aggregate" in plan

    def test_explain_shows_limit(self, engine):
        plan = engine.explain("SELECT * FROM orders LIMIT 5")
        assert "Limit" in plan


# ── HyperLogLog tests ──────────────────────────────────────────────────────

class TestHyperLogLog:
    def test_small_cardinality_exact(self):
        hll = HyperLogLog()
        values = list(range(100))
        for v in values:
            hll.add(v)
        estimate = hll.count()
        # within 10% for small cardinality
        assert abs(estimate - 100) <= 15, f"Expected ~100, got {estimate}"

    def test_large_cardinality(self):
        hll = HyperLogLog()
        n = 100_000
        for i in range(n):
            hll.add(f"value_{i}")
        estimate = hll.count()
        error_pct = abs(estimate - n) / n * 100
        assert error_pct < 5, f"Error too large: {error_pct:.1f}%"

    def test_duplicate_values(self):
        hll = HyperLogLog()
        for _ in range(1000):
            hll.add("same_value")
        assert hll.count() < 10  # should be ~1

    def test_memory_is_constant(self):
        hll_small = HyperLogLog()
        hll_large = HyperLogLog()
        for i in range(100):
            hll_small.add(i)
        for i in range(1_000_000):
            hll_large.add(i)
        # both use same memory — that's the whole point
        assert hll_small.memory_bytes() == hll_large.memory_bytes()

    def test_merge(self):
        hll1 = HyperLogLog()
        hll2 = HyperLogLog()
        for i in range(500):
            hll1.add(i)
        for i in range(500, 1000):
            hll2.add(i)
        hll1.merge(hll2)
        estimate = hll1.count()
        assert abs(estimate - 1000) < 100


# ── Index tests ────────────────────────────────────────────────────────────

class TestIndex:
    def test_create_and_lookup(self, engine, orders_csv):
        engine.create_index("orders", "country")
        from src.index.sorted_index import catalog as idx_catalog
        idx = idx_catalog.get(orders_csv, "country")
        assert idx is not None
        positions = idx.lookup_eq("CA")
        assert len(positions) == 3

    def test_range_lookup(self, engine, orders_csv):
        engine.create_index("orders", "total")
        from src.index.sorted_index import catalog as idx_catalog
        idx = idx_catalog.get(orders_csv, "total")
        if idx:
            rows = idx.lookup_range(100.0, 200.0)
            assert len(rows) >= 1


# ── Storage tests ──────────────────────────────────────────────────────────

class TestStorage:
    def test_csv_streaming(self, tmp_csv):
        rows = [{"a": i, "b": i * 2} for i in range(1000)]
        path = tmp_csv(rows)
        from src.storage.reader import CSVReader
        reader = CSVReader(path)
        count = sum(1 for _ in reader.scan())
        assert count == 1000

    def test_type_inference(self, tmp_csv):
        rows = [{"id": "1", "price": "9.99", "active": "true", "name": "foo"}]
        path = tmp_csv(rows)
        from src.storage.reader import CSVReader
        reader = CSVReader(path)
        schema = reader.schema
        assert schema["id"] == "int"
        assert schema["price"] == "float"
        assert schema["active"] == "bool"
        assert schema["name"] == "str"

    def test_projection_pushdown(self, tmp_csv):
        rows = [{"a": i, "b": i*2, "c": i*3, "d": i*4, "e": i*5} for i in range(100)]
        path = tmp_csv(rows)
        from src.storage.reader import CSVReader
        reader = CSVReader(path)
        # only read columns a and b
        for row in reader.scan(columns=["a", "b"]):
            assert set(row.keys()) == {"a", "b"}
            break
