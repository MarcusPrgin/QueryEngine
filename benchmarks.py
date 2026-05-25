#!/usr/bin/env python3
"""
Benchmark suite — generates the performance table for the README.

Run: python benchmarks.py
Requires: data/ directory (run generate_data.py first)
"""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.engine import QueryEngine


def bench(engine: QueryEngine, label: str, sql: str, runs: int = 3) -> dict:
    times = []
    rows = 0
    for _ in range(runs):
        t = time.perf_counter()
        result = engine.query(sql)
        elapsed = (time.perf_counter() - t) * 1000
        times.append(elapsed)
        rows = len(result.rows)
    avg = sum(times) / len(times)
    return {"label": label, "avg_ms": round(avg, 1), "rows_returned": rows}


def main():
    if not os.path.exists("data/orders.csv"):
        print("Run generate_data.py first.")
        sys.exit(1)

    engine = QueryEngine()
    engine.register("orders", "data/orders.csv")
    engine.register("customers", "data/customers.csv")

    print("\nBuilding index on country column...")
    stats = engine.create_index("orders", "country")
    print(f"  Index built in {stats['build_time_ms']}ms — {stats['distinct_values']} distinct values\n")

    benchmarks = [
        ("Full scan (SELECT *)",
         "SELECT * FROM orders"),
        ("Filter by country",
         "SELECT * FROM orders WHERE country = 'CA'"),
        ("Filter by range",
         "SELECT * FROM orders WHERE total > 100"),
        ("GROUP BY + SUM",
         "SELECT country, SUM(total) AS rev FROM orders GROUP BY country"),
        ("GROUP BY + COUNT",
         "SELECT category, COUNT(*) AS cnt FROM orders GROUP BY category"),
        ("GROUP BY + AVG",
         "SELECT country, AVG(total) AS avg_order FROM orders GROUP BY country"),
        ("COUNT DISTINCT (HLL)",
         "SELECT COUNT_DISTINCT(customer_id) AS unique_customers FROM orders"),
        ("Multi-condition filter",
         "SELECT * FROM orders WHERE country = 'CA' AND status = 'completed' AND total > 50"),
        ("ORDER BY + LIMIT",
         "SELECT * FROM orders ORDER BY total DESC LIMIT 100"),
        ("Complex aggregation",
         "SELECT country, category, COUNT(*) AS cnt, SUM(total) AS rev FROM orders GROUP BY country, category"),
    ]

    results = []
    for label, sql in benchmarks:
        r = bench(engine, label, sql)
        results.append(r)
        print(f"  {label:40s} {r['avg_ms']:8.1f}ms  {r['rows_returned']:6} rows")

    print("\n\nMarkdown table (paste into README):")
    print("\n| Query | Avg time (ms) | Rows returned |")
    print("|---|---:|---:|")
    for r in results:
        print(f"| {r['label']} | {r['avg_ms']} | {r['rows_returned']:,} |")

    print("\nDataset: 100,000 rows, streaming (no full load into memory)")


if __name__ == "__main__":
    main()
