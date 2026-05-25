"""
Query planner: converts a SelectStatement AST into a logical plan tree.

Plan construction order (bottom to top):
  Scan → [Filter(WHERE)] → [HashJoin] → [Aggregate(GROUP BY)]
       → [Filter(HAVING)] → [Window] → [Project] → [Distinct]
       → [Sort(ORDER BY)] → [Limit]
"""
from __future__ import annotations
from src.types import (
    SelectStatement, Scan, Filter, Project, Aggregate,
    Sort, Limit, HashJoin, Window, Distinct, Column, Star,
    FunctionCall, BinaryOp, JoinClause, WindowSpec
)
from src.catalog.table_catalog import TableCatalog


class PlanError(Exception):
    pass


def _collect_columns(expr) -> set[str]:
    if isinstance(expr, Column):
        return {expr.name}
    if isinstance(expr, BinaryOp):
        return _collect_columns(expr.left) | _collect_columns(expr.right)
    if isinstance(expr, FunctionCall):
        cols = set()
        for arg in expr.args:
            cols |= _collect_columns(arg)
        return cols
    if isinstance(expr, WindowSpec):
        cols = set()
        for e in expr.partition_by:
            cols |= _collect_columns(e)
        for e, _ in expr.order_by:
            cols |= _collect_columns(e)
        return cols
    return set()


def _collect_aggregates(columns) -> list[tuple[str, FunctionCall]]:
    """Find non-window aggregate function calls in the select list."""
    aggs = []
    for i, col in enumerate(columns):
        if isinstance(col, FunctionCall) and col.is_aggregate and col.over is None:
            if col.args and isinstance(col.args[0], Column):
                default_name = f"{col.name.lower()}_{col.args[0].name}"
            elif col.args and isinstance(col.args[0], Star):
                default_name = f"{col.name.lower()}_star"
            else:
                default_name = f"{col.name.lower()}_{i}"
            output_name = getattr(col, "alias", None) or default_name
            aggs.append((output_name, col))
    return aggs


def _collect_window_functions(columns) -> list[tuple[str, FunctionCall]]:
    """Find all window functions (those with OVER clause) in the select list."""
    wins = []
    for i, col in enumerate(columns):
        if isinstance(col, FunctionCall) and col.over is not None:
            default_name = f"{col.name.lower()}_{i}"
            output_name = getattr(col, "alias", None) or default_name
            wins.append((output_name, col))
    return wins


def build_plan(stmt: SelectStatement, catalog: TableCatalog,
               _ctes: dict | None = None):
    """
    Build a logical plan tree from a parsed SelectStatement.

    _ctes maps CTE name → SelectStatement and is threaded through recursive
    calls so that nested CTEs and multi-level WITH clauses work correctly.
    """
    if _ctes is None:
        _ctes = {}

    # Register CTEs defined in this statement (shallow copy so siblings
    # don't see each other's definitions — standard SQL scoping)
    local_ctes = dict(_ctes)
    for cte_name, cte_stmt in stmt.ctes:
        local_ctes[cte_name] = cte_stmt

    # ── Determine columns needed (projection pushdown) ─────────────────────
    has_star = any(isinstance(c, Star) for c in stmt.columns)
    needed_cols: set[str] = set()
    if not has_star:
        for col_expr in stmt.columns:
            needed_cols |= _collect_columns(col_expr)
        if stmt.where:
            needed_cols |= _collect_columns(stmt.where)
        if stmt.having:
            needed_cols |= _collect_columns(stmt.having)
        for g in stmt.group_by:
            needed_cols |= _collect_columns(g)
        for o, _ in stmt.order_by:
            needed_cols |= _collect_columns(o)
        for j in stmt.joins:
            needed_cols |= _collect_columns(j.condition)
        # Also pull columns referenced in window function specs
        for col_expr in stmt.columns:
            if isinstance(col_expr, FunctionCall) and col_expr.over:
                needed_cols |= _collect_columns(col_expr.over)

    scan_columns = sorted(needed_cols) if not has_star else []

    # ── Scan (or inline CTE) ───────────────────────────────────────────────
    if stmt.from_table in local_ctes:
        plan = build_plan(local_ctes[stmt.from_table], catalog, local_ctes)
    else:
        source = catalog.resolve(stmt.from_table)
        plan = Scan(
            source=source,
            columns=scan_columns,
            alias=stmt.from_table,
            sample_pct=stmt.sample_pct,
        )

    # ── Joins ──────────────────────────────────────────────────────────────
    for join in stmt.joins:
        if join.table in local_ctes:
            right_plan = build_plan(local_ctes[join.table], catalog, local_ctes)
        else:
            right_source = catalog.resolve(join.table)
            right_plan = Scan(source=right_source, columns=[], alias=join.table)
        plan = HashJoin(
            left=plan,
            right=right_plan,
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

    # ── Filter (HAVING) ────────────────────────────────────────────────────
    if stmt.having:
        plan = Filter(child=plan, predicate=stmt.having)

    # ── Window functions ───────────────────────────────────────────────────
    window_fns = _collect_window_functions(stmt.columns)
    if window_fns:
        plan = Window(child=plan, functions=window_fns)

    # ── Project (SELECT columns) ───────────────────────────────────────────
    if not has_star and not aggregates:
        # Window functions are already computed and stored in the row by the
        # Window node. Replace them with passthrough Column references so
        # Project can forward the pre-computed values without re-evaluating.
        project_cols = []
        for col in stmt.columns:
            if isinstance(col, FunctionCall) and col.over is not None:
                out_name = getattr(col, "alias", None) or f"{col.name.lower()}_0"
                project_cols.append(Column(name=out_name, alias=out_name))
            else:
                project_cols.append(col)
        plan = Project(child=plan, columns=project_cols)

    # ── DISTINCT ───────────────────────────────────────────────────────────
    if stmt.distinct:
        plan = Distinct(child=plan)

    # ── Sort (ORDER BY) ────────────────────────────────────────────────────
    if stmt.order_by:
        plan = Sort(child=plan, order_by=stmt.order_by)

    # ── Limit ──────────────────────────────────────────────────────────────
    if stmt.limit is not None:
        plan = Limit(child=plan, n=stmt.limit)

    return plan


def explain(plan, indent: int = 0) -> str:
    prefix = "  " * indent
    lines = []

    if isinstance(plan, Limit):
        lines.append(f"{prefix}Limit({plan.n})")
        lines.append(explain(plan.child, indent + 1))
    elif isinstance(plan, Sort):
        order_str = ", ".join(f"({_expr_str(e)}, {d})" for e, d in plan.order_by)
        lines.append(f"{prefix}Sort([{order_str}])")
        lines.append(explain(plan.child, indent + 1))
    elif isinstance(plan, Distinct):
        lines.append(f"{prefix}Distinct")
        lines.append(explain(plan.child, indent + 1))
    elif isinstance(plan, Window):
        fns = ", ".join(
            f"{n}={f.name}({', '.join(_expr_str(a) for a in f.args)})"
            f" OVER (PARTITION BY {', '.join(_expr_str(p) for p in f.over.partition_by)}"
            f" ORDER BY {', '.join(f'{_expr_str(e)} {d}' for e, d in f.over.order_by)})"
            for n, f in plan.functions
        )
        lines.append(f"{prefix}Window([{fns}])")
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
        sample = f" SAMPLE({plan.sample_pct}%)" if plan.sample_pct is not None else ""
        lines.append(f"{prefix}Scan({plan.source}, [{cols}]{sample})")
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
    if hasattr(expr, "value"):
        return repr(expr.value)
    if isinstance(expr, Star):
        return "*"
    return str(expr)
