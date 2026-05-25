.PHONY: all test bench data clean demo

all: test

# ── Test ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Data & benchmarks ──────────────────────────────────────────────────────
data:
	python generate_data.py

bench: data
	python benchmarks.py

# ── Demo ───────────────────────────────────────────────────────────────────
demo: data
	@echo "Starting interactive REPL with sample data..."
	python cli.py --data data/

# Run a single demo query
demo-query: data
	python cli.py --data data/ \
	  --query "SELECT country, COUNT(*) AS orders, SUM(total) AS revenue FROM orders GROUP BY country ORDER BY revenue DESC" \
	  --explain

# ── Install ────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	rm -rf data/ __pycache__ .pytest_cache .coverage
	find . -name "*.pyc" -delete
	find . -name "*.idx" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
