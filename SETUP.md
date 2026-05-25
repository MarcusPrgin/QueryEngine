# Setup Guide

## What you need

### 1. Python 3.11+

**macOS:**
```bash
brew install python@3.12
```

**Windows:**
Download from https://www.python.org/downloads/ — check "Add to PATH" during install.

**Linux:**
```bash
sudo apt-get install python3.12 python3.12-venv python3-pip
```

Verify: `python3 --version` → should show 3.11 or higher.

---

### 2. That's it

No Docker. No databases. No cloud accounts. Pure Python.

---

## Project setup

```bash
# 1. Unzip and enter the project
cd queryengine

# 2. (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# OR
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install pytest pyarrow

# 4. Run the tests — all should pass
pytest tests/ -v

# 5. Generate sample data (takes ~10 seconds)
python generate_data.py

# 6. Start the interactive REPL
python cli.py --data data/
```

---

## Dependency summary

| Package | Required? | Purpose |
|---|---|---|
| Python 3.11+ | Yes | Language runtime |
| pytest | Yes (dev) | Running the test suite |
| pyarrow | Optional | Parquet file support |

Everything else is Python standard library: `csv`, `json`, `hashlib`, `bisect`, `pickle`, `fnmatch`.

---

## First queries to run

Once the REPL is open:

```sql
-- How many orders per country?
SELECT country, COUNT(*) AS orders FROM orders GROUP BY country ORDER BY orders DESC

-- Revenue by category
SELECT category, SUM(total) AS revenue, AVG(total) AS avg_order FROM orders GROUP BY category

-- Top 10 biggest orders
SELECT * FROM orders ORDER BY total DESC LIMIT 10

-- Filter + aggregate
SELECT country, COUNT(*) AS cnt FROM orders WHERE status = 'completed' AND total > 100 GROUP BY country

-- COUNT DISTINCT with HyperLogLog
SELECT COUNT_DISTINCT(customer_id) AS unique_buyers FROM orders

-- See the query plan
.explain SELECT country, SUM(total) FROM orders WHERE status = 'completed' GROUP BY country

-- Build an index
.index orders country
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
You're not running from the project root. Make sure you `cd queryengine` before running any commands.

**`pytest: command not found`**
Activate your virtual environment first: `source .venv/bin/activate`

**`No such file or directory: 'data/orders.csv'`**
Run `python generate_data.py` first to create the sample datasets.

**Parquet import error**
`pyarrow` is optional. CSV and JSONL work without it. Install with `pip install pyarrow` if you want Parquet support.
