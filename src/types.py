"""
Core types used throughout the query engine.
Everything depends on this module — keep it import-free.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator, Optional


# ── Value types ────────────────────────────────────────────────────────────

# A single row is a dict of column_name -> value
Row = dict[str, Any]

# A schema describes column names and their inferred types
Schema = dict[str, str]  # column_name -> "int" | "float" | "str" | "bool"


# ── Token types (used by lexer/parser) ────────────────────────────────────

class TokenType(Enum):
    # literals
    NUMBER    = auto()
    STRING    = auto()
    IDENT     = auto()
    # keywords
    SELECT    = auto()
    FROM      = auto()
    WHERE     = auto()
    GROUP_BY  = auto()
    ORDER_BY  = auto()
    PARTITION_BY = auto()
    LIMIT     = auto()
    AND       = auto()
    OR        = auto()
    NOT       = auto()
    AS        = auto()
    JOIN      = auto()
    ON        = auto()
    INNER     = auto()
    LEFT      = auto()
    WITH      = auto()
    OVER      = auto()
    HAVING    = auto()
    DISTINCT  = auto()
    SAMPLE    = auto()
    # operators
    EQ = auto(); NEQ = auto(); LT = auto(); LTE = auto()
    GT = auto(); GTE = auto(); LIKE = auto()
    PLUS = auto(); MINUS = auto(); STAR = auto(); SLASH = auto()
    PERCENT = auto()
    # punctuation
    LPAREN = auto(); RPAREN = auto(); COMMA = auto()
    DOT    = auto(); SEMICOLON = auto()
    EOF    = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    pos: int = 0

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r})"


# ── AST node types ─────────────────────────────────────────────────────────

@dataclass
class Column:
    name: str
    table: Optional[str] = None
    alias: Optional[str] = None

    def __str__(self):
        base = f"{self.table}.{self.name}" if self.table else self.name
        return f"{base} AS {self.alias}" if self.alias else base


@dataclass
class Star:
    """SELECT *"""
    pass


@dataclass
class Literal:
    value: Any
    dtype: str  # "int" | "float" | "str" | "bool" | "null"


@dataclass
class BinaryOp:
    op: str   # "=", "!=", "<", "<=", ">", ">=", "AND", "OR", "LIKE", "+", "-", "*", "/"
    left: Any
    right: Any


@dataclass
class UnaryOp:
    op: str   # "NOT", "-"
    operand: Any


@dataclass
class WindowSpec:
    """OVER (PARTITION BY ... ORDER BY ...) clause attached to a window function."""
    partition_by: list[Any] = field(default_factory=list)
    order_by: list[tuple[Any, str]] = field(default_factory=list)


@dataclass
class FunctionCall:
    name: str   # "SUM", "COUNT", "AVG", "MIN", "MAX", "COUNT_DISTINCT", "TOPK"
    args: list[Any]
    is_aggregate: bool = False
    alias: Optional[str] = None
    over: Optional[WindowSpec] = None   # set when OVER (...) follows the call


@dataclass
class SelectStatement:
    columns: list[Any]        # Column | Star | FunctionCall | BinaryOp
    from_table: str
    where: Optional[Any] = None
    group_by: list[Column] = field(default_factory=list)
    order_by: list[tuple[Any, str]] = field(default_factory=list)  # (expr, "ASC"|"DESC")
    limit: Optional[int] = None
    joins: list["JoinClause"] = field(default_factory=list)
    ctes: list[tuple[str, "SelectStatement"]] = field(default_factory=list)
    having: Optional[Any] = None
    distinct: bool = False
    sample_pct: Optional[float] = None   # FROM t SAMPLE(10) → 10% Bernoulli sample


@dataclass
class JoinClause:
    join_type: str   # "INNER" | "LEFT"
    table: str
    condition: BinaryOp


# ── Logical plan nodes ─────────────────────────────────────────────────────

@dataclass
class Scan:
    """Read from a file or registered table."""
    source: str            # file path or table name
    columns: list[str]     # projected columns (empty = all)
    alias: Optional[str] = None
    sample_pct: Optional[float] = None   # Bernoulli sample percentage (0–100)

    def __str__(self):
        cols = ", ".join(self.columns) if self.columns else "*"
        suffix = f", SAMPLE({self.sample_pct}%)" if self.sample_pct is not None else ""
        return f"Scan({self.source}, [{cols}]{suffix})"


@dataclass
class Filter:
    """Apply a predicate — only pass rows where predicate is truthy."""
    child: Any
    predicate: Any

    def __str__(self):
        return f"Filter({self.predicate})\n  {self.child}"


@dataclass
class Project:
    """Select and rename columns."""
    child: Any
    columns: list[Any]

    def __str__(self):
        return f"Project({[str(c) for c in self.columns]})\n  {self.child}"


@dataclass
class Aggregate:
    """GROUP BY with aggregate functions."""
    child: Any
    group_by: list[Any]
    aggregates: list[tuple[str, FunctionCall]]  # (output_name, func)

    def __str__(self):
        aggs = ", ".join(f"{n}={f.name}({f.args})" for n, f in self.aggregates)
        groups = ", ".join(str(g) for g in self.group_by)
        return f"Aggregate(group=[{groups}], agg=[{aggs}])\n  {self.child}"


@dataclass
class Sort:
    child: Any
    order_by: list[tuple[Any, str]]

    def __str__(self):
        return f"Sort({self.order_by})\n  {self.child}"


@dataclass
class Limit:
    child: Any
    n: int

    def __str__(self):
        return f"Limit({self.n})\n  {self.child}"


@dataclass
class HashJoin:
    """Hash join — build side is right child (should be smaller)."""
    left: Any
    right: Any
    condition: BinaryOp
    join_type: str = "INNER"   # "INNER" | "LEFT"

    def __str__(self):
        return f"HashJoin({self.join_type}, {self.condition})\n  {self.left}\n  {self.right}"


@dataclass
class Window:
    """Window functions: ROW_NUMBER, RANK, LAG, running aggregates, etc."""
    child: Any
    functions: list[tuple[str, FunctionCall]]  # (output_col_name, func_with_over)

    def __str__(self):
        fns = ", ".join(f"{n}={f.name}()" for n, f in self.functions)
        return f"Window([{fns}])\n  {self.child}"


@dataclass
class Distinct:
    """Deduplicate rows based on all projected columns."""
    child: Any

    def __str__(self):
        return f"Distinct\n  {self.child}"


# ── EXPLAIN ANALYZE result ─────────────────────────────────────────────────

@dataclass
class NodeStats:
    """Per-node execution statistics collected during EXPLAIN ANALYZE."""
    label: str
    rows_in: int = 0
    rows_out: int = 0
    elapsed_ms: float = 0.0
    children: list["NodeStats"] = field(default_factory=list)

    def pretty(self, indent: int = 0) -> str:
        prefix = "  " * indent
        ratio = f"{self.rows_out / self.rows_in * 100:.0f}%" if self.rows_in else "—"
        line = (
            f"{prefix}{self.label}"
            f"  rows={self.rows_out:,}  selectivity={ratio}"
            f"  time={self.elapsed_ms:.1f}ms"
        )
        lines = [line]
        for child in self.children:
            lines.append(child.pretty(indent + 1))
        return "\n".join(lines)

    def total_elapsed_ms(self) -> float:
        return self.elapsed_ms + sum(c.total_elapsed_ms() for c in self.children)


# ── Result type ────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    rows: list[Row]
    schema: Schema
    plan: str = ""
    rows_scanned: int = 0
    rows_returned: int = 0
    elapsed_ms: float = 0.0

    def pretty(self) -> str:
        if not self.rows:
            return "(no rows)"
        cols = list(self.rows[0].keys())
        widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in self.rows), default=0))
                  for c in cols}
        sep = "+" + "+".join("-" * (w + 2) for w in widths.values()) + "+"
        header = "|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|"
        lines = [sep, header, sep]
        for row in self.rows:
            lines.append("|" + "|".join(f" {str(row.get(c,'')):<{widths[c]}} " for c in cols) + "|")
        lines.append(sep)
        lines.append(f"{len(self.rows)} row(s) | {self.rows_scanned} scanned | {self.elapsed_ms:.1f}ms")
        return "\n".join(lines)
