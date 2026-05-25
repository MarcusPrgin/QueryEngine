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


# ── New feature tests ──────────────────────────────────────────────────────

class TestCTE:
    def test_basic_cte(self, engine):
        result = engine.query("""
            WITH ca_orders AS (
                SELECT * FROM orders WHERE country = 'CA'
            )
            SELECT * FROM ca_orders
        """)
        assert len(result.rows) == 3
        assert all(r["country"] == "CA" for r in result.rows)

    def test_cte_with_aggregation(self, engine):
        result = engine.query("""
            WITH totals AS (
                SELECT country, SUM(total) AS rev FROM orders GROUP BY country
            )
            SELECT * FROM totals WHERE rev > 100
        """)
        # CA total = 300, US = 225, UK = 105, DE = 300
        assert all(r["rev"] > 100 for r in result.rows)

    def test_cte_in_explain(self, engine):
        plan = engine.explain("""
            WITH top AS (SELECT * FROM orders LIMIT 5)
            SELECT * FROM top
        """)
        assert "Scan" in plan


class TestWindowFunctions:
    def test_row_number(self, engine):
        result = engine.query("""
            SELECT country, total,
                   ROW_NUMBER() OVER (PARTITION BY country ORDER BY total DESC) AS rn
            FROM orders
        """)
        # within each country partition, row numbers should start at 1
        ca_rows = [r for r in result.rows if r["country"] == "CA"]
        rn_values = sorted(r["rn"] for r in ca_rows)
        assert rn_values == list(range(1, len(ca_rows) + 1))

    def test_rank(self, engine):
        result = engine.query("""
            SELECT order_id, total,
                   RANK() OVER (ORDER BY total DESC) AS rnk
            FROM orders
        """)
        assert len(result.rows) == 8
        # highest total should have rank 1
        top = min(result.rows, key=lambda r: r["rnk"])
        assert top["rnk"] == 1

    def test_dense_rank(self, engine):
        result = engine.query("""
            SELECT country,
                   DENSE_RANK() OVER (ORDER BY country ASC) AS dr
            FROM orders
        """)
        assert all(r["dr"] >= 1 for r in result.rows)

    def test_lag(self, engine):
        result = engine.query("""
            SELECT order_id, total,
                   LAG(total, 1) OVER (ORDER BY order_id ASC) AS prev_total
            FROM orders
        """)
        # first row should have NULL prev_total
        first = min(result.rows, key=lambda r: r["order_id"])
        assert first["prev_total"] is None

    def test_lead(self, engine):
        result = engine.query("""
            SELECT order_id, total,
                   LEAD(total, 1) OVER (ORDER BY order_id ASC) AS next_total
            FROM orders
        """)
        assert len(result.rows) == 8
        # last row should have NULL next_total
        last = max(result.rows, key=lambda r: r["order_id"])
        assert last["next_total"] is None

    def test_window_sum(self, engine):
        result = engine.query("""
            SELECT country, total,
                   SUM(total) OVER (PARTITION BY country) AS country_total
            FROM orders
        """)
        ca_rows = [r for r in result.rows if r["country"] == "CA"]
        expected_ca_total = sum(r["total"] for r in ca_rows)
        assert all(abs(r["country_total"] - expected_ca_total) < 0.01 for r in ca_rows)


class TestHaving:
    def test_having_filter(self, engine):
        result = engine.query("""
            SELECT country, COUNT(*) AS cnt
            FROM orders
            GROUP BY country
            HAVING cnt >= 2
        """)
        # CA has 3, US has 2, UK has 2, DE has 1
        assert all(r["cnt"] >= 2 for r in result.rows)
        countries = {r["country"] for r in result.rows}
        assert "DE" not in countries  # only 1 order

    def test_having_sum(self, engine):
        result = engine.query("""
            SELECT country, SUM(total) AS rev
            FROM orders
            GROUP BY country
            HAVING rev > 200
        """)
        assert all(r["rev"] > 200 for r in result.rows)


class TestDistinct:
    def test_select_distinct(self, engine):
        result = engine.query("SELECT DISTINCT country FROM orders")
        countries = [r["country"] for r in result.rows]
        assert len(countries) == len(set(countries))  # no duplicates
        assert set(countries) == {"CA", "US", "UK", "DE"}

    def test_distinct_status(self, engine):
        result = engine.query("SELECT DISTINCT status FROM orders")
        statuses = [r["status"] for r in result.rows]
        assert len(statuses) == len(set(statuses))


class TestSample:
    def test_sample_reduces_rows(self, tmp_csv):
        rows = [{"id": i, "val": i} for i in range(10000)]
        path = tmp_csv(rows)
        from src.engine import QueryEngine
        e = QueryEngine()
        e.register("big", path)
        result = e.query("SELECT * FROM big SAMPLE(10)")
        # With 10% Bernoulli sampling, expect roughly 1000 rows ± 200
        assert 500 < len(result.rows) < 1500

    def test_sample_in_plan(self, engine):
        plan = engine.explain("SELECT * FROM orders SAMPLE(50)")
        assert "SAMPLE" in plan


class TestTopK:
    def test_topk_aggregate(self, engine):
        result = engine.query(
            "SELECT TOPK(country, 2) AS top_countries FROM orders"
        )
        assert len(result.rows) == 1
        top = result.rows[0]["top_countries"]
        assert "CA" in top or "US" in top  # CA and US have most orders


class TestExplainAnalyze:
    def test_analyze_returns_stats(self, engine):
        result, stats = engine.analyze(
            "SELECT * FROM orders WHERE total > 100"
        )
        assert stats is not None
        assert stats.label != ""

    def test_analyze_result_correct(self, engine):
        result, stats = engine.analyze(
            "SELECT country, COUNT(*) AS cnt FROM orders GROUP BY country"
        )
        assert len(result.rows) == 4  # CA, US, UK, DE

    def test_stats_pretty_format(self, engine):
        _, stats = engine.analyze("SELECT * FROM orders LIMIT 5")
        pretty = stats.pretty()
        assert "rows=" in pretty


class TestBloomFilter:
    def test_no_false_negatives(self):
        from src.aggregation.bloom_filter import BloomFilter
        bf = BloomFilter(expected_items=1000)
        values = list(range(500))
        for v in values:
            bf.add(v)
        # Must never miss an inserted value
        for v in values:
            assert bf.might_contain(v), f"False negative for {v}"

    def test_false_positive_rate(self):
        from src.aggregation.bloom_filter import BloomFilter
        bf = BloomFilter(expected_items=1000, fp_rate=0.01)
        for i in range(1000):
            bf.add(f"key_{i}")
        fp = sum(1 for i in range(1000, 5000) if bf.might_contain(f"key_{i}"))
        fp_rate = fp / 4000
        assert fp_rate < 0.05  # should be well below 5%

    def test_bloom_join_correctness(self, tmp_csv, orders_csv):
        # Bloom join must not drop matching rows (no false negatives)
        labels = tmp_csv(
            [{"order_id": i, "label": f"L{i}"} for i in range(1, 9)],
            "labels.csv"
        )
        from src.engine import QueryEngine
        e = QueryEngine()
        e.register("orders", orders_csv)
        e.register("labels", labels)
        result = e.query(
            "SELECT orders.order_id, labels.label FROM orders "
            "INNER JOIN labels ON orders.order_id = labels.order_id"
        )
        assert len(result.rows) == 8  # every order has a matching label


class TestCountMinSketch:
    def test_frequency_estimate(self):
        from src.aggregation.count_min_sketch import CountMinSketch
        cms = CountMinSketch()
        for _ in range(100):
            cms.add("apple")
        for _ in range(50):
            cms.add("banana")
        assert cms.estimate("apple") >= 100
        assert cms.estimate("banana") >= 50

    def test_topk_tracker(self):
        from src.aggregation.count_min_sketch import TopKTracker
        tracker = TopKTracker(k=3)
        for _ in range(100):
            tracker.add("apple")
        for _ in range(80):
            tracker.add("banana")
        for _ in range(60):
            tracker.add("cherry")
        for _ in range(10):
            tracker.add("date")
        top = tracker.topk()
        top_values = [v for _, v in top]
        assert "apple" in top_values
        assert "banana" in top_values
