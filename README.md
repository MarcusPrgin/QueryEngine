# QueryEngine

A SQL query engine built from scratch in Python — no Pandas, no SQLite, no shortcuts. Executes SQL over CSV, JSONL, and Parquet files without loading them into memory.

Full pipeline: **lexer → parser → AST → logical planner → optimizer → Volcano executor**

---

## Architecture

```
SQL string
    │
    ▼
┌──────────┐
│  Lexer   │  character scan → token list
└────┬─────┘
     │ tokens
     ▼
┌──────────┐
│  Parser  │  recursive descent + Pratt precedence → AST
└────┬─────┘
     │ SelectStatement (+ CTEs, window specs, HAVING, DISTINCT, SAMPLE)
     ▼
┌──────────┐
│ Planner  │  AST → logical plan tree
└────┬─────┘
     │ Scan → Filter → HashJoin → Aggregate → Window → Project → Distinct → Sort → Limit
     ▼
┌───────────┐
│ Optimizer │  predicate pushdown · projection pruning
└────┬──────┘
     │ optimized plan
     ▼
┌──────────┐
│ Executor │  Volcano iterator model — pull-based, O(1) memory per operator
└────┬─────┘
     │ row stream
     ▼
  QueryResult  (rows · schema · plan string · timing · scan stats)
```

---

## SQL support

### Core
```sql
SELECT * FROM orders
SELECT order_id, country, ROUND(total, 2) FROM orders
SELECT * FROM orders WHERE total > 100 AND status = 'completed'
SELECT * FROM orders WHERE country LIKE 'C%'
```

### Aggregation
```sql
SELECT country, COUNT(*) AS cnt, SUM(total) AS rev FROM orders GROUP BY country
SELECT category, AVG(total), MIN(total), MAX(total) FROM orders GROUP BY category

-- post-aggregate filter
SELECT country, SUM(total) AS rev FROM orders GROUP BY country HAVING rev > 500

-- COUNT DISTINCT: HyperLogLog — O(1) memory, ~0.8% error
SELECT COUNT_DISTINCT(customer_id) AS unique_customers FROM orders

-- approximate top-k most frequent values via Count-Min Sketch
SELECT TOPK(country, 3) AS top_markets FROM orders
```

### Window functions
```sql
-- ranking
SELECT country, total,
       ROW_NUMBER() OVER (PARTITION BY country ORDER BY total DESC) AS rn
FROM orders

SELECT total, RANK() OVER (ORDER BY total DESC) AS rnk FROM orders
SELECT total, DENSE_RANK() OVER (ORDER BY total DESC) AS dr FROM orders
SELECT total, NTILE(4) OVER (ORDER BY total ASC) AS quartile FROM orders

-- offset
SELECT order_id, total,
       LAG(total, 1)  OVER (ORDER BY order_id) AS prev_total,
       LEAD(total, 1) OVER (ORDER BY order_id) AS next_total
FROM orders

-- running aggregates over a partition
SELECT country, total,
       SUM(total)   OVER (PARTITION BY country) AS country_total,
       AVG(total)   OVER (PARTITION BY country) AS country_avg,
       FIRST_VALUE(total) OVER (PARTITION BY country ORDER BY total DESC) AS best
FROM orders
```

### CTEs
```sql
WITH high_value AS (
    SELECT customer_id, SUM(total) AS lifetime_value
    FROM orders
    GROUP BY customer_id
),
vip AS (
    SELECT * FROM high_value WHERE lifetime_value > 1000
)
SELECT * FROM vip ORDER BY lifetime_value DESC LIMIT 10
```

### Joins
```sql
SELECT o.order_id, c.name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id

SELECT o.order_id, c.name
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
```

### Deduplication & Sampling
```sql
-- exact deduplication
SELECT DISTINCT country FROM orders

-- Bernoulli sample (each row included with probability p=10%)
SELECT * FROM orders SAMPLE(10)
```

### Sorting & Limiting
```sql
SELECT * FROM orders ORDER BY country ASC, total DESC
SELECT * FROM orders ORDER BY total DESC LIMIT 100
```

---

## Key design decisions

### 1. Streaming reads — O(1) memory

Every reader is a Python generator — a 10 GB file uses the same RAM as a 10 MB file. The 64 KB read buffer aligns with OS page cache multiples.

### 2. Volcano (pull-based) executor

Each operator is a generator. The top of the plan pulls rows from its child, all the way to the Scan. No operator materialises its full input:

```
Limit.__next__() → Filter.__next__() → Scan.__next__() → disk
```

Trade-off: ~100 ns per-row dispatch. Vectorised engines (DuckDB) cut this 1 000× by processing 1 024-row batches.

### 3. Predicate pushdown

Filters move as close to the Scan as possible. If a filter rejects 90% of rows, all operators above it do 10% of the work.

### 4. Projection pruning

The planner tells each Scan which columns to read. For a 50-column CSV where the query touches 3 columns, 47 columns are skipped per row.

### 5. Hash aggregation

`GROUP BY` builds a hash map of `group_key → accumulators`. O(distinct groups) memory.

### 6. HyperLogLog — O(1) COUNT DISTINCT

`COUNT_DISTINCT(x)` uses a 16 KB HLL sketch instead of a Python set, achieving ~0.8% error at any cardinality. This is the same algorithm used by Redshift, BigQuery, and Redis.

### 7. Count-Min Sketch — TOPK

`TOPK(col, k)` answers "what are the k most frequent values?" using a W×D frequency table (default: 2 000×7 = ~110 KB). Standard GROUP BY+ORDER BY requires materialising all groups; Count-Min does not.

### 8. Sorted index — true O(log n) lookups

Build once; every subsequent equality or range query is O(log n) via `bisect`. The index is a sidecar `.idx` file, invalidated by source mtime:

```python
engine.create_index("orders", "country")
# WHERE country = 'CA' → O(log n), not O(n)
```

### 9. Bloom-filter hash join

Before building the hash table on the right side, a Bloom filter is populated. The probe phase checks the filter first — a miss is definitive (no false negatives), so the hash lookup is skipped entirely for absent keys. Sparse joins (few matching rows) get the most benefit.

### 10. External merge sort

If a `ORDER BY` materialises more than `SORT_SPILL_ROWS` (default: 500 000) rows, the executor writes sorted runs to temp pickle files and k-way merges them via `heapq.merge`. In-memory RAM is bounded regardless of dataset size.

### 11. Window functions — correct multi-partition execution

`OVER (PARTITION BY ... ORDER BY ...)` materialises each partition, applies the window function with correct NULL handling (LAG/LEAD boundary), then emits rows. Supported: `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, and running `SUM/AVG/MIN/MAX/COUNT`.

### 12. Common Table Expressions (WITH)

CTEs are inlined at plan time — no new executor node needed. Each CTE name is resolved to its subquery plan when the FROM clause references it. Multiple CTEs in one WITH clause are supported.

### 13. EXPLAIN ANALYZE — per-node profiling

```python
result, stats = engine.analyze("SELECT * FROM orders WHERE total > 100")
print(stats.pretty())
# Filter(total > 100.0)  rows=6  selectivity=75%  time=0.3ms
#   Scan(orders.csv, [*])  rows=8  selectivity=—  time=0.2ms
```

Every plan node records rows in, rows out, and wall-clock time. Selectivity = rows_out / rows_in shows where the filter is doing work.

### 14. Parallel CSV scan

For CSV files ≥ 32 MB, `CSVReader.scan_parallel()` splits the file into byte-range chunks aligned on newline boundaries, scans each chunk in a separate process via `ProcessPoolExecutor`, and merges the results. Small files use the standard single-threaded path (parallelism overhead dominates below the threshold).

---

## EXPLAIN ANALYZE output

```bash
python cli.py --data data/ --analyze \
  "SELECT country, SUM(total) AS rev
   FROM orders WHERE status='completed'
   GROUP BY country ORDER BY rev DESC LIMIT 5"
```

```
Limit(5)  rows=5  selectivity=—  time=0.1ms
  Sort([(rev, DESC)])  rows=8  selectivity=—  time=0.4ms
    Aggregate(group=[country], agg=[rev=SUM(total)])  rows=8  ...  time=12.1ms
      Filter(status = 'completed')  rows=61823  selectivity=62%  time=18.3ms
        Scan(orders.csv, [country, total, status])  rows=100000  ...  time=30.2ms
```

---

## Benchmark results

Dataset: 100 000 rows, streaming (no full load into memory).

| Query | Avg time (ms) | Rows returned |
|---|---:|---:|
| Full scan (SELECT *) | — | 100,000 |
| Filter by country | — | ~10,000 |
| Filter by range | — | ~75,000 |
| GROUP BY + SUM | — | 10 |
| GROUP BY + COUNT | — | 8 |
| COUNT DISTINCT (HLL) | — | 1 |
| TOPK(country, 3) | — | 1 |
| Multi-condition filter | — | ~2,000 |
| ORDER BY + LIMIT 100 | — | 100 |
| Window function (ROW_NUMBER) | — | 100,000 |
| Complex aggregation | — | 80 |

Run `make bench` to populate with numbers from your machine.

---

## File format support

| Format | Streaming | Projection pushdown | Type inference | Parallel scan |
|---|:---:|:---:|:---:|:---:|
| CSV | ✓ | ✓ | ✓ | ✓ (≥32 MB) |
| JSONL / NDJSON | ✓ | ✓ | ✓ | — |
| Parquet | ✓ (batched) | ✓ (columnar) | native schema | — |

---

## Quick start

```bash
pip install pytest pyarrow   # pyarrow optional

make data    # generate 100K orders, 10K customers, 1K products
make test    # 64 tests, all passing
make demo    # interactive REPL
```

### Python API

```python
from src.engine import QueryEngine

engine = QueryEngine()
engine.register("orders", "data/orders.csv")
engine.register("customers", "data/customers.csv")

# Basic query
result = engine.query(
    "SELECT country, SUM(total) AS rev FROM orders GROUP BY country ORDER BY rev DESC"
)
print(result.pretty())
# +----------+-----------+
# | country  | rev       |
# +----------+-----------+
# | US       | 182034.5  |
# ...
# 10 row(s) | 100000 scanned | 42.3ms

# CTE + Window function
result = engine.query("""
    WITH ranked AS (
        SELECT customer_id, country, total,
               ROW_NUMBER() OVER (PARTITION BY country ORDER BY total DESC) AS rn
        FROM orders
    )
    SELECT * FROM ranked WHERE rn = 1
""")

# EXPLAIN ANALYZE — per-node profiling
result, stats = engine.analyze(
    "SELECT * FROM orders WHERE total > 500 LIMIT 10"
)
print(stats.pretty())

# Approximate top-k (Count-Min Sketch)
result = engine.query("SELECT TOPK(country, 3) AS top FROM orders")

# Bernoulli sampling
result = engine.query("SELECT * FROM orders SAMPLE(10)")  # ~10% of rows

# Build index → O(log n) lookups
engine.create_index("orders", "country")
```

### REPL commands

```
sql> SELECT country, COUNT(*) FROM orders GROUP BY country ORDER BY count DESC
sql> .tables
sql> .schema orders
sql> .explain SELECT * FROM orders WHERE total > 100
sql> .analyze SELECT country, SUM(total) FROM orders GROUP BY country
sql> .index orders country
sql> .plan        ← toggle showing plan before every query
```

---

## Bugs fixed during development

**1. Schema inference double-read** — initially inferred schema on every `scan()` call by reading the first row, causing off-by-one errors. Fix: separate `_read_raw(limit=N)` method that doesn't advance the main iterator.

**2. COUNT(*) on empty result** — aggregate on empty input returned zero rows instead of one row with `count=0`. SQL requires COUNT(*) with no GROUP BY to always return exactly one row.

**3. Mixed ASC/DESC ORDER BY** — single-pass sort silently sorted all columns in the first column's direction. Fixed using stable multi-pass sorting from least to most significant key.

**4. `rows_scanned` always zero** — the scan-counting closure was defined but the `execute()` call bypassed it. Fixed with a module-level `_scan_counter` that `execute_scan` increments when set.

**5. `lookup_eq` was O(n)** — `bisect` was imported but never called; the method used a linear scan. Fixed with `bisect.bisect_left` on the sorted key list, with a `_none_count` fast-skip for None-padded indices.

**6. `eval_expr` called twice in sort** — each sort-key evaluation computed `eval_expr(expr, row)` twice per row per key (once for `is None`, once for the value). Fixed using keyed tuples `(eval_expr(expr, row), row)`.

---

## What breaks at 10× scale (1M → 10M rows)

- Streaming holds up — memory stays flat
- Full-scan time grows linearly — need index coverage or Parquet
- Hash aggregation can exceed RAM if distinct group count is huge — needs spill-to-disk hash table
- Window functions materialise entire partitions — need streaming window frames
- CPython ~100 ns/row overhead dominates — need Cython, PyPy, or Rust for the hot path

---

## Tech stack

- **Pure Python, no Pandas** — forces understanding of what Pandas abstracts
- **Python generators** — `yield from execute(child)` *is* the Volcano model
- **No external query libraries** — `pyarrow` optional (Parquet), `pytest` for tests, everything else is stdlib
