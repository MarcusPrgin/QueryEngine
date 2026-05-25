"""
Volcano (iterator) model executor.

Every operator is a Python generator (or iterator) with a __next__ method.
Execution is pull-based: the top of the plan calls next() on its child,
which calls next() on ITS child, all the way down to the Scan at the bottom.

This is how MySQL, PostgreSQL, and SQLite execute queries.

Interview talking point: "The iterator model composes naturally — I can chain
operators without them knowing about each other. The downside is per-row
function call overhead. In Python that's ~100ns/row. At 1M rows that's 100ms
of pure dispatch overhead. Vectorized engines (DuckDB) process batches of
1024 rows per call, reducing dispatch cost by 1000×."
"""
from __future__ import annotations
import fnmatch
from typing import Any, Generator
from src.types import (
    Row, Scan, Filter, Project, Aggregate, Sort, Limit, HashJoin,
    BinaryOp, UnaryOp, Column, Star, Literal, FunctionCall
)
from src.storage.reader import get_reader
from src.aggregation.hyperloglog import HyperLogLog

# Set by QueryEngine during query execution to count rows emitted by Scan nodes.
# None when not in a query (avoids any overhead outside of a live query call).
_scan_counter: list[int] | None = None


class ExecutionError(Exception):
    pass


# ── Expression evaluator ───────────────────────────────────────────────────

def eval_expr(expr, row: Row) -> Any:
    """
    Evaluate an expression against a single row.
    This is the hot path — called for every row in the scan.
    """
    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, Column):
        # try exact match first, then case-insensitive
        if expr.name in row:
            return row[expr.name]
        if expr.table:
            qualified = f"{expr.table}.{expr.name}"
            if qualified in row:
                return row[qualified]
        # case-insensitive fallback
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
        # scalar functions only — aggregates are handled in execute_aggregate
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
    return None


# ── Operator implementations ───────────────────────────────────────────────

def execute_scan(node: Scan) -> Generator[Row, None, None]:
    reader = get_reader(node.source)
    cols = node.columns if node.columns else None
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
                name = col_expr.args[0].name if col_expr.args and isinstance(col_expr.args[0], Column) else "expr"
                key = getattr(col_expr, 'alias', None) or f"{col_expr.name.lower()}_{name}"
                out[key] = eval_expr(col_expr, row)
            elif isinstance(col_expr, BinaryOp):
                out[str(col_expr)] = eval_expr(col_expr, row)
            else:
                out[str(col_expr)] = eval_expr(col_expr, row)
        yield out


def execute_aggregate(node: Aggregate) -> Generator[Row, None, None]:
    """
    Hash aggregation: build a hash map of group_key -> running accumulators.
    Memory: O(distinct groups). Falls over if groups don't fit in RAM.
    
    Accumulators per aggregate function:
      SUM   -> running total
      COUNT -> running count (non-null)
      AVG   -> (sum, count) pair
      MIN   -> current minimum
      MAX   -> current maximum
      COUNT_DISTINCT -> HyperLogLog sketch (approximate) or set (exact)
    """
    groups: dict = {}

    for row in execute(node.child):
        # build group key
        key = tuple(eval_expr(g, row) for g in node.group_by)

        if key not in groups:
            # initialise accumulators
            acc = {}
            for out_name, func in node.aggregates:
                if func.name == "SUM":
                    acc[out_name] = 0.0
                elif func.name in ("COUNT", "COUNT_DISTINCT"):
                    acc[out_name] = HyperLogLog() if func.name == "COUNT_DISTINCT" else 0
                elif func.name == "AVG":
                    acc[out_name] = [0.0, 0]  # [sum, count]
                elif func.name == "MIN":
                    acc[out_name] = None
                elif func.name == "MAX":
                    acc[out_name] = None
            groups[key] = acc

        acc = groups[key]
        for out_name, func in node.aggregates:
            if isinstance(func.args[0], Star):
                val = 1
            else:
                val = eval_expr(func.args[0], row)

            if func.name == "SUM":
                if val is not None:
                    acc[out_name] += float(val)
            elif func.name == "COUNT":
                if val is not None:
                    acc[out_name] += 1
            elif func.name == "COUNT_DISTINCT":
                if val is not None:
                    acc[out_name].add(val)
            elif func.name == "AVG":
                if val is not None:
                    acc[out_name][0] += float(val)
                    acc[out_name][1] += 1
            elif func.name == "MIN":
                if val is not None and (acc[out_name] is None or val < acc[out_name]):
                    acc[out_name] = val
            elif func.name == "MAX":
                if val is not None and (acc[out_name] is None or val > acc[out_name]):
                    acc[out_name] = val

    for key, acc in groups.items():
        out: Row = {}
        # add group-by columns
        for i, g in enumerate(node.group_by):
            col_name = g.name if isinstance(g, Column) else f"group_{i}"
            out[col_name] = key[i]
        # finalise aggregates
        for out_name, func in node.aggregates:
            if func.name == "AVG":
                s, c = acc[out_name]
                out[out_name] = s / c if c > 0 else None
            elif func.name == "COUNT_DISTINCT":
                out[out_name] = acc[out_name].count()
            else:
                out[out_name] = acc[out_name]
        yield out


def execute_sort(node: Sort) -> Generator[Row, None, None]:
    """Sort requires materialising all rows — O(n log n).

    Uses stable multi-pass sorting (least-significant key first) so mixed
    ASC/DESC order works correctly without a custom comparator.  Each
    expression is evaluated exactly once per row per sort key.
    """
    rows = list(execute(node.child))
    if not rows:
        return
    for expr, direction in reversed(node.order_by):
        keyed = [(eval_expr(expr, row), row) for row in rows]
        keyed.sort(key=lambda x: (x[0] is None, x[0]), reverse=(direction == "DESC"))
        rows = [r for _, r in keyed]
    yield from rows


def execute_limit(node: Limit) -> Generator[Row, None, None]:
    for i, row in enumerate(execute(node.child)):
        if i >= node.n:
            break
        yield row


def execute_hash_join(node: HashJoin) -> Generator[Row, None, None]:
    """
    Hash join: build a hash table from the right (smaller) side,
    then probe it with each row from the left side.
    
    Time:  O(n + m)
    Space: O(right side)
    
    Limitation: right side must fit in memory.
    Fix for large right sides: grace hash join (spill to disk).
    """
    # Build phase: materialise the right side into a hash map
    build_table: dict[Any, list[Row]] = {}

    # extract join key column from condition
    cond = node.condition
    right_key_col = cond.right.name if isinstance(cond.right, Column) else None
    left_key_col = cond.left.name if isinstance(cond.left, Column) else None

    for row in execute(node.right):
        key = row.get(right_key_col) if right_key_col else eval_expr(cond.right, row)
        if key not in build_table:
            build_table[key] = []
        build_table[key].append(row)

    # Probe phase: for each left row, look up matching right rows
    for left_row in execute(node.left):
        probe_key = left_row.get(left_key_col) if left_key_col else eval_expr(cond.left, left_row)
        matches = build_table.get(probe_key, [])

        if matches:
            for right_row in matches:
                merged = {**left_row, **right_row}
                yield merged
        elif node.join_type == "LEFT":
            # LEFT JOIN: emit left row with nulls for right columns
            yield {**left_row}


# ── Dispatcher ─────────────────────────────────────────────────────────────

def execute(plan) -> Generator[Row, None, None]:
    """Route a plan node to its executor."""
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
    else:
        raise ExecutionError(f"Unknown plan node: {type(plan)}")
