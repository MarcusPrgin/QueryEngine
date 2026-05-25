"""
Query planner: converts a SelectStatement AST into a logical plan tree.

The plan is a tree of operator nodes (Scan, Filter, Project, Aggregate, etc.)
Each node wraps its child. Execution reads bottom-up.

Plan for: SELECT name, SUM(total) FROM orders WHERE country='CA' GROUP BY name
  Aggregate(group=[name], agg=[sum(total)])
    Filter(country = 'CA')
      Scan(orders.csv, [name, total, country])
"""
from __future__ import annotations
from src.types import (
    SelectStatement, Scan, Filter, Project, Aggregate,
    Sort, Limit, HashJoin, Column, Star, FunctionCall,
    BinaryOp, JoinClause
)
from src.catalog.table_catalog import TableCatalog


class PlanError(Exception):
    pass


def _collect_columns(expr) -> set[str]:
    """Walk an expression tree and collect all Column references."""
    if isinstance(expr, Column):
        return {expr.name}
    if isinstance(expr, BinaryOp):
        return _collect_columns(expr.left) | _collect_columns(expr.right)
    if isinstance(expr, FunctionCall):
        cols = set()
        for arg in expr.args:
            cols |= _collect_columns(arg)
        return cols
    return set()


def _collect_aggregates(columns) -> list[tuple[str, FunctionCall]]:
    """Find all aggregate function calls in the select list."""
    aggs = []
    for i, col in enumerate(columns):
        if isinstance(col, FunctionCall) and col.is_aggregate:
            # Determine output column name: alias > func_col > func_i
            if col.args and isinstance(col.args[0], Column):
                default_name = f"{col.name.lower()}_{col.args[0].name}"
            elif col.args and isinstance(col.args[0], Star):
                default_name = f"{col.name.lower()}_star"
            else:
                default_name = f"{col.name.lower()}_{i}"
            output_name = getattr(col, 'alias', None) or default_name
            aggs.append((output_name, col))
    return aggs


def build_plan(stmt: SelectStatement, catalog: TableCatalog):
    """
    Build a logical plan tree from a parsed SelectStatement.
    Returns the root operator node.
    """
    # ── Resolve source path ────────────────────────────────────────────────
    source = catalog.resolve(stmt.from_table)

    # ── Determine which columns we actually need (projection pushdown) ─────
    # Only read columns referenced anywhere in the query.
    needed_cols: set[str] = set()
    has_star = any(isinstance(c, Star) for c in stmt.columns)

    if not has_star:
        for col_expr in stmt.columns:
            needed_cols |= _collect_columns(col_expr)
        if stmt.where:
            needed_cols |= _collect_columns(stmt.where)
        for g in stmt.group_by:
            needed_cols |= _collect_columns(g)
        for o, _ in stmt.order_by:
            needed_cols |= _collect_columns(o)
        for j in stmt.joins:
            needed_cols |= _collect_columns(j.condition)

    scan_columns = sorted(needed_cols) if not has_star else []

    # ── Scan ───────────────────────────────────────────────────────────────
    plan = Scan(source=source, columns=scan_columns, alias=stmt.from_table)

    # ── Joins ──────────────────────────────────────────────────────────────
    for join in stmt.joins:
        right_source = catalog.resolve(join.table)
        right_scan = Scan(source=right_source, columns=[], alias=join.table)
        plan = HashJoin(
            left=plan,
            right=right_scan,
            condition=join.condition,
            join_type=join.join_type,
        )

    # ── Filter (WHERE) ─────────────────────────────────────────────────────
    if stmt.where:
        plan = Filter(child=plan, predicate=stmt.where)

    # ── Aggregate (GROUP BY) ───────────────────────────────────────────────
    aggregates = _collect_aggregates(stmt.columns)
    if aggregates or stmt.group_by:
        plan = Aggregate(
            child=plan,
            group_by=stmt.group_by,
            aggregates=aggregates,
        )

    # ── Project (SELECT columns) ───────────────────────────────────────────
    if not has_star and not aggregates:
        plan = Project(child=plan, columns=stmt.columns)

    # ── Sort (ORDER BY) ────────────────────────────────────────────────────
    if stmt.order_by:
        plan = Sort(child=plan, order_by=stmt.order_by)

    # ── Limit ──────────────────────────────────────────────────────────────
    if stmt.limit is not None:
        plan = Limit(child=plan, n=stmt.limit)

    return plan


def explain(plan, indent: int = 0) -> str:
    """
    Pretty-print the query plan tree.
    This is what gets printed in the README and shown in interview demos.
    
    Example output:
      Limit(10)
        Sort([('total', 'DESC')])
          Aggregate(group=[country], agg=[total=SUM(total)])
            Filter(country = 'CA')
              Scan(orders.csv, [name, total, country])
    """
    prefix = "  " * indent
    lines = []

    if isinstance(plan, Limit):
        lines.append(f"{prefix}Limit({plan.n})")
        lines.append(explain(plan.child, indent + 1))
    elif isinstance(plan, Sort):
        order_str = ", ".join(f"({_expr_str(e)}, {d})" for e, d in plan.order_by)
        lines.append(f"{prefix}Sort([{order_str}])")
        lines.append(explain(plan.child, indent + 1))
    elif isinstance(plan, Aggregate):
        groups = ", ".join(_expr_str(g) for g in plan.group_by)
        aggs = ", ".join(f"{n}={f.name}({', '.join(_expr_str(a) for a in f.args)})"
                         for n, f in plan.aggregates)
        lines.append(f"{prefix}Aggregate(group=[{groups}], agg=[{aggs}])")
        lines.append(explain(plan.child, indent + 1))
    elif isinstance(plan, Filter):
        lines.append(f"{prefix}Filter({_expr_str(plan.predicate)})")
        lines.append(explain(plan.child, indent + 1))
    elif isinstance(plan, Project):
        cols = ", ".join(_expr_str(c) for c in plan.columns)
        lines.append(f"{prefix}Project([{cols}])")
        lines.append(explain(plan.child, indent + 1))
    elif isinstance(plan, HashJoin):
        lines.append(f"{prefix}HashJoin({plan.join_type}, ON {_expr_str(plan.condition)})")
        lines.append(explain(plan.left, indent + 1))
        lines.append(explain(plan.right, indent + 1))
    elif isinstance(plan, Scan):
        cols = ", ".join(plan.columns) if plan.columns else "*"
        lines.append(f"{prefix}Scan({plan.source}, [{cols}])")
    else:
        lines.append(f"{prefix}{plan}")

    return "\n".join(lines)


def _expr_str(expr) -> str:
    if isinstance(expr, Column):
        base = f"{expr.table}.{expr.name}" if expr.table else expr.name
        return f"{base} AS {expr.alias}" if expr.alias else base
    if isinstance(expr, BinaryOp):
        return f"{_expr_str(expr.left)} {expr.op} {_expr_str(expr.right)}"
    if isinstance(expr, FunctionCall):
        args = ", ".join(_expr_str(a) for a in expr.args)
        return f"{expr.name}({args})"
    if hasattr(expr, 'value'):
        return repr(expr.value)
    if isinstance(expr, Star):
        return "*"
    return str(expr)
