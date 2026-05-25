#!/usr/bin/env python3
"""
Query Engine CLI.

Usage:
    # Interactive REPL
    python cli.py --data data/

    # Run a single query
    python cli.py --data data/ --query "SELECT * FROM orders LIMIT 10"

    # Run from a SQL file
    python cli.py --data data/ --file query.sql

    # Show query plan
    python cli.py --data data/ --query "SELECT country, COUNT(*) FROM orders GROUP BY country" --explain
"""
import argparse
import sys
import time
from src.engine import QueryEngine


BANNER = """
  Query Engine v1.0
  Type a SQL query and press Enter. Type 'help' for commands, 'exit' to quit.
  ─────────────────────────────────────────────────────────────────────────
"""

HELP = """
  Commands:
    .tables              list registered tables
    .schema <table>      show table schema
    .indexes             list built indexes
    .index <t> <col>     build index on table.column
    .explain <sql>       show query plan without executing
    .plan                toggle showing plan before each query
    help                 show this message
    exit / quit          exit the REPL
"""


def run_query(engine: QueryEngine, sql: str, show_plan: bool = False):
    try:
        result = engine.query(sql, show_plan=show_plan)
        print(result.pretty())
        if show_plan:
            print(f"\nPlan:\n{result.plan}")
    except Exception as e:
        print(f"Error: {e}")


def repl(engine: QueryEngine):
    print(BANNER)
    show_plan = False
    buf = []

    while True:
        try:
            prompt = "  sql> " if not buf else "  ...> "
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        # meta-commands
        if line.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if line.lower() == "help":
            print(HELP)
            continue
        if line == ".tables":
            for t in engine.tables():
                print(f"  {t['name']:20s}  {t['path']}  ({t['size_bytes']:,} bytes)")
            continue
        if line.startswith(".schema "):
            tbl = line.split()[1]
            schema = engine.schema(tbl)
            for col, dtype in schema.items():
                print(f"  {col:30s}  {dtype}")
            continue
        if line == ".indexes":
            for idx in engine.indexes():
                print(f"  {idx['column']:20s}  {idx['distinct_values']:,} values  {idx['index_size_bytes']:,} bytes")
            continue
        if line.startswith(".index "):
            parts = line.split()
            if len(parts) != 3:
                print("Usage: .index <table> <column>")
                continue
            stats = engine.create_index(parts[1], parts[2])
            print(f"  Index built in {stats['build_time_ms']}ms — {stats['distinct_values']:,} distinct values")
            continue
        if line.startswith(".explain "):
            sql = line[9:]
            print(engine.explain(sql))
            continue
        if line == ".plan":
            show_plan = not show_plan
            print(f"  Show plan: {'on' if show_plan else 'off'}")
            continue

        # accumulate multi-line SQL
        buf.append(line)
        full = " ".join(buf)

        if full.rstrip().endswith(";") or "\n" not in full:
            sql = full.rstrip(";").strip()
            if sql:
                run_query(engine, sql, show_plan=show_plan)
            buf = []


def main():
    parser = argparse.ArgumentParser(description="Query Engine CLI")
    parser.add_argument("--data", "-d", help="Directory of data files to auto-register")
    parser.add_argument("--query", "-q", help="SQL query to run")
    parser.add_argument("--file", "-f", help="SQL file to run")
    parser.add_argument("--explain", "-e", action="store_true", help="Show query plan")
    parser.add_argument("--register", "-r", nargs=2, action="append",
                        metavar=("NAME", "PATH"), help="Register a table: --register orders data/orders.csv")
    args = parser.parse_args()

    engine = QueryEngine(data_dir=args.data)

    if args.register:
        for name, path in args.register:
            engine.register(name, path)

    if args.query:
        run_query(engine, args.query, show_plan=args.explain)
    elif args.file:
        with open(args.file) as f:
            sql = f.read()
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                run_query(engine, stmt, show_plan=args.explain)
    else:
        repl(engine)


if __name__ == "__main__":
    main()
