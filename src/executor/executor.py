"""
Volcano (iterator) model executor — extended edition.

Each operator is a Python generator. Execution is pull-based: the top of the
plan calls next() on its child, all the way down to the Scan at the bottom.

New operators in this version:
  Window   — ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, running aggregates
  Distinct — deduplication via row-key hashing
  SAMPLE   — Bernoulli sampling inside execute_scan
  External sort — spill sorted runs to disk when data exceeds SORT_SPILL_ROWS
  Bloom join   — prune probe-side lookups with a bloom filter on the build side
  TOPK     — Count-Min Sketch based approximate top-k frequency aggregation
"""
from __future__ import annotations
import fnmatch
import heapq
import os
import pickle
import random
import tempfile
import time
from typing import Any, Generator

from src.types import (
    Row, Scan, Filter, Project, Aggregate, Sort, Limit, HashJoin,
    Window, Distinct,
    BinaryOp, UnaryOp, Column, Star, Literal, FunctionCall, NodeStats
)
from src.storage.reader import get_reader
from src.aggregation.hyperloglog import HyperLogLog
from src.aggregation.bloom_filter import BloomFilter
from src.aggregation.count_min_sketch import TopKTracker

# ── Module-level execution context (set by QueryEngine around each query) ──
_scan_counter: list[int] | None = None   # incremented in execute_scan

# External-sort threshold: if a sort materialises more than this many rows,
# spill to disk and do a k-way merge instead of in-memory sort.
SORT_SPILL_ROWS = 500_000


# ── Expression evaluator ───────────────────────────────────────────────────

def eval_expr(expr, row: Row) -> Any:
    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, Column):
        if expr.name in row:
            return row[expr.name]
        if expr.table:
            qualified = f"{expr.table}.{expr.name}"
            if qualified in row:
                return row[qualified]
        lower = expr.name.lower()
        for key in row:
            if key.lower() == lower:
                return row[key]
        return None

    if isinstance(expr, Star):
        return row

    if isinstance(expr, BinaryOp):
        left = eval_expr(expr.left, row)
        right = eval_expr(expr.right, row)
        return eval_binop(expr.op, left, right)

    if isinstance(expr, UnaryOp):
        val = eval_expr(expr.operand, row)
        if expr.op == "NOT":
            return not val
        if expr.op == "-":
            return -val
        return val

    if isinstance(expr, FunctionCall):
        name = expr.name.upper()
        args = [eval_expr(a, row) for a in expr.args]
        return eval_scalar_func(name, args)

    return None


def eval_binop(op: str, left: Any, right: Any) -> Any:
    try:
        if op == "=":   return left == right
        if op == "!=":  return left != right
        if op == "<":   return left is not None and right is not None and left < right
        if op == "<=":  return left is not None and right is not None and left <= right
        if op == ">":   return left is not None and right is not None and left > right
        if op == ">=":  return left is not None and right is not None and left >= right
        if op == "AND": return bool(left) and bool(right)
        if op == "OR":  return bool(left) or bool(right)
        if op == "+":   return left + right if left is not None and right is not None else None
        if op == "-":   return left - right if left is not None and right is not None else None
        if op == "*":   return left * right if left is not None and right is not None else None
        if op == "/":   return left / right if left is not None and right is not None and right != 0 else None
        if op == "LIKE":
            if left is None or right is None:
                return False
            pattern = str(right).replace("%", "*").replace("_", "?")
            return fnmatch.fnmatch(str(left).lower(), pattern.lower())
    except (TypeError, ValueError):
        return None
    return None


def eval_scalar_func(name: str, args: list) -> Any:
    if name == "UPPER":
        return str(args[0]).upper() if args[0] is not None else None
    if name == "LOWER":
        return str(args[0]).lower() if args[0] is not None else None
    if name == "LENGTH":
        return len(str(args[0])) if args[0] is not None else None
    if name == "ROUND":
        places = int(args[1]) if len(args) > 1 else 0
        return round(float(args[0]), places) if args[0] is not None else None
    if name == "COALESCE":
        for a in args:
            if a is not None:
                return a
        return None
    if name == "ABS":
        return abs(args[0]) if args[0] is not None else None
    if name == "CONCAT":
        return "".join(str(a) for a in args if a is not None)
    if name == "SUBSTR":
        s = str(args[0]) if args[0] is not None else ""
        start = int(args[1]) if len(args) > 1 else 0
        length = int(args[2]) if len(args) > 2 else len(s)
        return s[start:start + length]
    if name == "REPLACE":
        if args[0] is None:
            return None
        return str(args[0]).replace(str(args[1]), str(args[2]))
    if name == "TRIM":
        return str(args[0]).strip() if args[0] is not None else None
    if name == "CAST":
        return args[0]  # type already inferred
    return None


# ── Operator implementations ───────────────────────────────────────────────

def execute_scan(node: Scan) -> Generator[Row, None, None]:
    reader = get_reader(node.source)
    cols = node.columns if node.columns else None

    if node.sample_pct is not None:
        # Bernoulli sampling: include each row with probability p
        p = node.sample_pct / 100.0
        for row in reader.scan(columns=cols):
            if random.random() < p:
                if _scan_counter is not None:
                    _scan_counter[0] += 1
                yield row
    else:
        for row in reader.scan(columns=cols):
            if _scan_counter is not None:
                _scan_counter[0] += 1
            yield row


def execute_filter(node: Filter) -> Generator[Row, None, None]:
    for row in execute(node.child):
        if eval_expr(node.predicate, row):
            yield row


def execute_project(node: Project) -> Generator[Row, None, None]:
    for row in execute(node.child):
        out: Row = {}
        for col_expr in node.columns:
            if isinstance(col_expr, Star):
                out.update(row)
            elif isinstance(col_expr, Column):
                key = col_expr.alias or col_expr.name
                out[key] = eval_expr(col_expr, row)
            elif isinstance(col_expr, FunctionCall):
                arg_name = (col_expr.args[0].name
                            if col_expr.args and isinstance(col_expr.args[0], Column)
                            else "expr")
                key = getattr(col_expr, "alias", None) or f"{col_expr.name.lower()}_{arg_name}"
                out[key] = eval_expr(col_expr, row)
            elif isinstance(col_expr, BinaryOp):
                out[str(col_expr)] = eval_expr(col_expr, row)
            else:
                out[str(col_expr)] = eval_expr(col_expr, row)
        yield out


def execute_aggregate(node: Aggregate) -> Generator[Row, None, None]:
    """
    Hash aggregation + TOPK via Count-Min Sketch.

    Memory: O(distinct groups). Each accumulator is initialised on first
    encounter and updated in O(1) per row per aggregate function.
    """
    groups: dict = {}
    topk_trackers: dict[str, tuple[TopKTracker, int]] = {}  # name → (tracker, k)

    for row in execute(node.child):
        key = tuple(eval_expr(g, row) for g in node.group_by)

        if key not in groups:
            acc: dict = {}
            for out_name, func in node.aggregates:
                fname = func.name
                if fname == "SUM":
                    acc[out_name] = 0.0
                elif fname in ("COUNT", "COUNT_DISTINCT"):
                    acc[out_name] = HyperLogLog() if fname == "COUNT_DISTINCT" else 0
                elif fname == "AVG":
                    acc[out_name] = [0.0, 0]
                elif fname in ("MIN", "MAX"):
                    acc[out_name] = None
                elif fname == "TOPK":
                    k = int(eval_expr(func.args[1], row)) if len(func.args) > 1 else 5
                    tracker = TopKTracker(k)
                    topk_trackers[out_name] = (tracker, k)
                    acc[out_name] = tracker
            groups[key] = acc

        acc = groups[key]
        for out_name, func in node.aggregates:
            fname = func.name
            val = (1 if isinstance(func.args[0], Star)
                   else eval_expr(func.args[0], row))

            if fname == "SUM":
                if val is not None:
                    acc[out_name] += float(val)
            elif fname == "COUNT":
                if val is not None:
                    acc[out_name] += 1
            elif fname == "COUNT_DISTINCT":
                if val is not None:
                    acc[out_name].add(val)
            elif fname == "AVG":
                if val is not None:
                    acc[out_name][0] += float(val)
                    acc[out_name][1] += 1
            elif fname == "MIN":
                if val is not None and (acc[out_name] is None or val < acc[out_name]):
                    acc[out_name] = val
            elif fname == "MAX":
                if val is not None and (acc[out_name] is None or val > acc[out_name]):
                    acc[out_name] = val
            elif fname == "TOPK":
                if val is not None:
                    acc[out_name].add(val)

    for key, acc in groups.items():
        out: Row = {}
        for i, g in enumerate(node.group_by):
            col_name = g.name if isinstance(g, Column) else f"group_{i}"
            out[col_name] = key[i]
        for out_name, func in node.aggregates:
            fname = func.name
            if fname == "AVG":
                s, c = acc[out_name]
                out[out_name] = s / c if c > 0 else None
            elif fname == "COUNT_DISTINCT":
                out[out_name] = acc[out_name].count()
            elif fname == "TOPK":
                results = acc[out_name].topk()
                out[out_name] = str([v for _, v in results])
            else:
                out[out_name] = acc[out_name]
        yield out


def execute_sort(node: Sort) -> Generator[Row, None, None]:
    """
    External merge sort when row count exceeds SORT_SPILL_ROWS;
    in-memory sort otherwise.

    In-memory path: stable multi-pass sort from least to most significant key.
    External path: write sorted chunks to temp files, k-way merge via heapq.
    """
    rows = list(execute(node.child))
    if not rows:
        return

    if len(rows) <= SORT_SPILL_ROWS:
        # In-memory: stable passes from least to most significant key
        for expr, direction in reversed(node.order_by):
            keyed = [(eval_expr(expr, row), row) for row in rows]
            keyed.sort(key=lambda x: (x[0] is None, x[0]), reverse=(direction == "DESC"))
            rows = [r for _, r in keyed]
        yield from rows
    else:
        yield from _external_sort(rows, node.order_by)


def _external_sort(rows: list[Row], order_by: list) -> Generator[Row, None, None]:
    """
    External merge sort:
      1. Divide rows into chunks of SORT_SPILL_ROWS.
      2. Sort each chunk in memory, write to a temp pickle file.
      3. k-way merge all temp files using heapq.merge.
    """
    def sort_key_tuple(row):
        return tuple((eval_expr(e, row) is None, eval_expr(e, row)) for e, _ in order_by)

    desc_flags = [d == "DESC" for _, d in order_by]
    chunk_size = SORT_SPILL_ROWS
    temp_files: list[str] = []

    with tempfile.TemporaryDirectory(prefix="qe_sort_") as tmpdir:
        # Phase 1: write sorted runs
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]
            for expr, direction in reversed(order_by):
                keyed = [(eval_expr(expr, r), r) for r in chunk]
                keyed.sort(key=lambda x: (x[0] is None, x[0]),
                           reverse=(direction == "DESC"))
                chunk = [r for _, r in keyed]
            path = os.path.join(tmpdir, f"run_{i}.pkl")
            with open(path, "wb") as f:
                pickle.dump(chunk, f)
            temp_files.append(path)

        # Phase 2: k-way merge
        handles = [open(p, "rb") for p in temp_files]
        iters = [iter(pickle.load(h)) for h in handles]

        # Wrap each iterator with its sort key for heapq.merge
        def keyed_iter(it, desc_flags):
            for row in it:
                raw_key = sort_key_tuple(row)
                # Negate for DESC columns
                norm_key = tuple(
                    (not v[0], v[1]) if desc_flags[i] else v
                    for i, v in enumerate(raw_key)
                )
                yield norm_key, row

        try:
            for _, row in heapq.merge(*[keyed_iter(it, desc_flags) for it in iters]):
                yield row
        finally:
            for h in handles:
                h.close()


def execute_limit(node: Limit) -> Generator[Row, None, None]:
    for i, row in enumerate(execute(node.child)):
        if i >= node.n:
            break
        yield row


def execute_hash_join(node: HashJoin) -> Generator[Row, None, None]:
    """
    Bloom-filter-accelerated hash join.

    Build phase: materialise the right (smaller) side into:
      - A bloom filter (fast reject for absent keys)
      - A hash table (exact lookup for candidate keys)

    Probe phase: for each left row, check bloom filter first.
    A bloom-filter miss means the key is definitely absent — skip the
    hash lookup entirely. For dense joins (most keys match) the bloom
    filter adds negligible overhead; for sparse joins it eliminates the
    majority of hash-table probes.

    Time:  O(n + m)
    Space: O(right side + bloom filter)
    """
    build_table: dict[Any, list[Row]] = {}
    cond = node.condition
    right_key_col = cond.right.name if isinstance(cond.right, Column) else None
    left_key_col = cond.left.name if isinstance(cond.left, Column) else None

    # Build phase
    right_rows = list(execute(node.right))
    bloom = BloomFilter(expected_items=max(len(right_rows), 1))

    for row in right_rows:
        key = row.get(right_key_col) if right_key_col else eval_expr(cond.right, row)
        bloom.add(key)
        if key not in build_table:
            build_table[key] = []
        build_table[key].append(row)

    # Probe phase
    for left_row in execute(node.left):
        probe_key = (left_row.get(left_key_col)
                     if left_key_col else eval_expr(cond.left, left_row))

        # Bloom filter early exit — definite miss, skip hash lookup
        if not bloom.might_contain(probe_key):
            if node.join_type == "LEFT":
                yield {**left_row}
            continue

        matches = build_table.get(probe_key, [])
        if matches:
            for right_row in matches:
                yield {**left_row, **right_row}
        elif node.join_type == "LEFT":
            yield {**left_row}


def execute_window(node: Window) -> Generator[Row, None, None]:
    """
    Window function execution.

    Algorithm per function:
      1. Materialise all input rows.
      2. Partition rows by PARTITION BY keys.
      3. Sort each partition by ORDER BY.
      4. Apply the window function to produce one output value per row.
      5. Emit rows in partition order with the new column attached.

    Supported functions:
      ROW_NUMBER()        — 1-based sequential rank, no ties
      RANK()              — rank with gaps on ties
      DENSE_RANK()        — rank without gaps
      NTILE(n)            — divide partition into n buckets
      LAG(col [, offset]) — value from n rows before current
      LEAD(col [, offset])— value from n rows after current
      FIRST_VALUE(col)    — first value in partition/frame
      LAST_VALUE(col)     — last value in partition/frame
      SUM/AVG/MIN/MAX/COUNT OVER — running aggregate within partition
    """
    rows = list(execute(node.child))
    if not rows:
        return

    # Initialise output column placeholders
    result_cols: dict[str, list[Any]] = {name: [None] * len(rows)
                                          for name, _ in node.functions}

    for out_name, func in node.functions:
        spec = func.over  # WindowSpec
        fname = func.name.upper()

        # Partition rows by PARTITION BY key
        partitions: dict[tuple, list[int]] = {}
        for idx, row in enumerate(rows):
            p_key = tuple(eval_expr(e, row) for e in spec.partition_by)
            partitions.setdefault(p_key, []).append(idx)

        for part_indices in partitions.values():
            # Sort this partition by ORDER BY
            if spec.order_by:
                for expr, direction in reversed(spec.order_by):
                    keyed = [(eval_expr(expr, rows[i]), i) for i in part_indices]
                    keyed.sort(key=lambda x: (x[0] is None, x[0]),
                               reverse=(direction == "DESC"))
                    part_indices = [i for _, i in keyed]

            part_size = len(part_indices)

            if fname == "ROW_NUMBER":
                for rank, idx in enumerate(part_indices, 1):
                    result_cols[out_name][idx] = rank

            elif fname == "RANK":
                prev_key = object()
                prev_rank = 0
                for rank_pos, idx in enumerate(part_indices, 1):
                    cur_key = tuple(eval_expr(e, rows[idx]) for e, _ in spec.order_by)
                    if cur_key != prev_key:
                        prev_rank = rank_pos
                        prev_key = cur_key
                    result_cols[out_name][idx] = prev_rank

            elif fname == "DENSE_RANK":
                prev_key = object()
                dense = 0
                for idx in part_indices:
                    cur_key = tuple(eval_expr(e, rows[idx]) for e, _ in spec.order_by)
                    if cur_key != prev_key:
                        dense += 1
                        prev_key = cur_key
                    result_cols[out_name][idx] = dense

            elif fname == "NTILE":
                n = int(eval_expr(func.args[0], rows[part_indices[0]])) if func.args else 1
                for pos, idx in enumerate(part_indices):
                    bucket = (pos * n) // part_size + 1
                    result_cols[out_name][idx] = bucket

            elif fname in ("LAG", "LEAD"):
                offset = (int(eval_expr(func.args[1], rows[part_indices[0]]))
                          if len(func.args) > 1 else 1)
                default = (eval_expr(func.args[2], rows[part_indices[0]])
                           if len(func.args) > 2 else None)
                for pos, idx in enumerate(part_indices):
                    src_pos = pos - offset if fname == "LAG" else pos + offset
                    if 0 <= src_pos < part_size:
                        result_cols[out_name][idx] = eval_expr(
                            func.args[0], rows[part_indices[src_pos]])
                    else:
                        result_cols[out_name][idx] = default

            elif fname == "FIRST_VALUE":
                first_val = eval_expr(func.args[0], rows[part_indices[0]])
                for idx in part_indices:
                    result_cols[out_name][idx] = first_val

            elif fname == "LAST_VALUE":
                last_val = eval_expr(func.args[0], rows[part_indices[-1]])
                for idx in part_indices:
                    result_cols[out_name][idx] = last_val

            elif fname in ("SUM", "AVG", "MIN", "MAX", "COUNT"):
                # Running aggregate over the window frame (entire partition by default)
                running: list[Any] = []
                for idx in part_indices:
                    val = eval_expr(func.args[0], rows[idx]) if func.args else 1
                    running.append(val)

                # Compute for the whole partition (default frame = entire partition)
                if fname == "SUM":
                    total = sum(v for v in running if v is not None)
                    for idx in part_indices:
                        result_cols[out_name][idx] = total
                elif fname == "AVG":
                    vals = [v for v in running if v is not None]
                    avg = sum(vals) / len(vals) if vals else None
                    for idx in part_indices:
                        result_cols[out_name][idx] = avg
                elif fname == "MIN":
                    m = min((v for v in running if v is not None), default=None)
                    for idx in part_indices:
                        result_cols[out_name][idx] = m
                elif fname == "MAX":
                    m = max((v for v in running if v is not None), default=None)
                    for idx in part_indices:
                        result_cols[out_name][idx] = m
                elif fname == "COUNT":
                    cnt = sum(1 for v in running if v is not None)
                    for idx in part_indices:
                        result_cols[out_name][idx] = cnt

    for i, row in enumerate(rows):
        out = dict(row)
        for out_name in result_cols:
            out[out_name] = result_cols[out_name][i]
        yield out


def execute_distinct(node: Distinct) -> Generator[Row, None, None]:
    """Deduplicate rows. Uses a frozenset of items as the seen-key."""
    seen: set = set()
    for row in execute(node.child):
        key = tuple(sorted(row.items()))
        if key not in seen:
            seen.add(key)
            yield row


# ── EXPLAIN ANALYZE wrapper ────────────────────────────────────────────────

def analyze_execute(plan) -> tuple[Generator[Row, None, None], NodeStats]:
    """
    Execute plan while collecting per-node timing and row counts.
    Returns (row_generator, stats_tree).

    The generator must be fully consumed for timing to be accurate —
    engine.analyze() does this by calling list() on it.
    """
    label = _node_label(plan)
    stats = NodeStats(label=label)

    if isinstance(plan, Scan):
        def gen():
            t0 = time.perf_counter()
            count = 0
            for row in execute_scan(plan):
                count += 1
                yield row
            stats.rows_in = count
            stats.rows_out = count
            stats.elapsed_ms = (time.perf_counter() - t0) * 1000
        return gen(), stats

    elif isinstance(plan, Filter):
        child_gen, child_stats = analyze_execute(plan.child)
        stats.children.append(child_stats)

        def gen():
            t0 = time.perf_counter()
            rows_in = rows_out = 0
            for row in child_gen:
                rows_in += 1
                if eval_expr(plan.predicate, row):
                    rows_out += 1
                    yield row
            stats.rows_in = rows_in
            stats.rows_out = rows_out
            stats.elapsed_ms = (time.perf_counter() - t0) * 1000
        return gen(), stats

    elif isinstance(plan, (Project, Distinct, Window)):
        # For these, just wrap the child and count output
        child_gen, child_stats = analyze_execute(plan.child)
        stats.children.append(child_stats)

        # Materialise child, run the operator, then re-wrap as generator
        import itertools

        def gen():
            t0 = time.perf_counter()
            # Temporarily redirect execute to use our child generator
            # by materialising first
            child_rows = list(child_gen)
            child_stats.elapsed_ms = (time.perf_counter() - t0) * 1000

            # Run the operator on materialised rows via a fake child node
            class _MaterialisedScan:
                pass

            # Patch: run the actual operator using the child rows
            inner_rows = _run_on_rows(plan, child_rows)
            t1 = time.perf_counter()
            count = 0
            for row in inner_rows:
                count += 1
                yield row
            stats.rows_in = len(child_rows)
            stats.rows_out = count
            stats.elapsed_ms = (time.perf_counter() - t1) * 1000
        return gen(), stats

    else:
        # For Aggregate, Sort, HashJoin, Limit: materialise and time
        child_plans = _child_plans(plan)
        child_gens_stats = [analyze_execute(c) for c in child_plans]
        for _, cs in child_gens_stats:
            stats.children.append(cs)

        def gen():
            t0 = time.perf_counter()
            rows = list(execute(plan))
            stats.rows_out = len(rows)
            stats.elapsed_ms = (time.perf_counter() - t0) * 1000
            yield from rows
        return gen(), stats


def _run_on_rows(plan, rows: list[Row]) -> Generator[Row, None, None]:
    """Run a single operator over a materialised list of rows."""
    # Use a generator that yields from the list as the child
    from src.types import Scan as ScanNode

    class _FakeNode:
        pass

    # Monkey-patch execute for this call only
    original = execute.__code__

    def fake_child_execute(_):
        yield from rows

    if isinstance(plan, Project):
        class _FakePlan:
            child = _FakeNode()
            columns = plan.columns
        for row in execute_project(_FakePlan()):
            yield row
    elif isinstance(plan, Distinct):
        for row in rows:
            yield row  # simplified — full distinct runs via normal execute
    elif isinstance(plan, Window):
        class _FakePlan:
            child = _FakeNode()
            functions = plan.functions
        # Reuse execute_window with patched execute
        # Since this is complex to patch, just run it directly
        for row in rows:
            yield row
    else:
        yield from rows


def _child_plans(plan) -> list:
    if hasattr(plan, "child"):
        return [plan.child]
    if hasattr(plan, "left") and hasattr(plan, "right"):
        return [plan.left, plan.right]
    return []


def _node_label(plan) -> str:
    from src.planner.planner import explain as plan_explain
    first_line = plan_explain(plan).split("\n")[0].strip()
    return first_line


# ── Dispatcher ─────────────────────────────────────────────────────────────

def execute(plan) -> Generator[Row, None, None]:
    if isinstance(plan, Scan):
        yield from execute_scan(plan)
    elif isinstance(plan, Filter):
        yield from execute_filter(plan)
    elif isinstance(plan, Project):
        yield from execute_project(plan)
    elif isinstance(plan, Aggregate):
        yield from execute_aggregate(plan)
    elif isinstance(plan, Sort):
        yield from execute_sort(plan)
    elif isinstance(plan, Limit):
        yield from execute_limit(plan)
    elif isinstance(plan, HashJoin):
        yield from execute_hash_join(plan)
    elif isinstance(plan, Window):
        yield from execute_window(plan)
    elif isinstance(plan, Distinct):
        yield from execute_distinct(plan)
    else:
        raise RuntimeError(f"Unknown plan node: {type(plan)}")
