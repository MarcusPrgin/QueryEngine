"""
Recursive descent parser.

Grammar (simplified):
  statement   := [with_clause] select_stmt
  with_clause := WITH cte (, cte)*
  cte         := ident AS ( select_stmt )
  select_stmt := SELECT [DISTINCT] select_list FROM ident [SAMPLE(n)]
                 [join_clause*] [WHERE expr] [GROUP BY expr_list]
                 [HAVING expr] [ORDER BY order_list] [LIMIT number]
  select_list := * | expr (AS ident)? (, expr (AS ident)?)*
  expr        := comparison (AND|OR comparison)*
  comparison  := additive (=|!=|<|<=|>|>= additive)? [NOT? LIKE string]
  additive    := multiplicative (+|- multiplicative)*
  multiplicative := unary (*|/ unary)*
  unary       := NOT? primary
  primary     := number | string | ident | func_call [OVER window_spec] | (expr)
  window_spec := ( [PARTITION BY expr_list] [ORDER BY order_list] )
"""
from __future__ import annotations
from src.types import (
    Token, TokenType, Column, Star, Literal, BinaryOp, UnaryOp,
    FunctionCall, SelectStatement, JoinClause, WindowSpec
)
from src.parser.lexer import tokenize

AGGREGATE_FUNCTIONS = {"sum", "count", "avg", "min", "max", "count_distinct", "topk"}
WINDOW_FUNCTIONS = {"row_number", "rank", "dense_rank", "lag", "lead", "ntile",
                    "sum", "avg", "min", "max", "count", "first_value", "last_value"}


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def error(self, msg: str) -> ParseError:
        tok = self.current()
        return ParseError(f"{msg} (got {tok.type.name} {tok.value!r} at position {tok.pos})")

    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else self.tokens[-1]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def expect(self, tt: TokenType) -> Token:
        if self.current().type != tt:
            raise self.error(f"Expected {tt.name}")
        return self.advance()

    def match(self, *types: TokenType) -> bool:
        return self.current().type in types

    # ── Entry point ────────────────────────────────────────────────────────

    def parse(self) -> SelectStatement:
        ctes = []
        if self.match(TokenType.WITH):
            ctes = self.parse_with_clause()
        stmt = self.parse_select()
        stmt.ctes = ctes
        return stmt

    # ── WITH / CTEs ────────────────────────────────────────────────────────

    def parse_with_clause(self) -> list[tuple[str, SelectStatement]]:
        self.expect(TokenType.WITH)
        ctes = [self._parse_one_cte()]
        while self.match(TokenType.COMMA):
            self.advance()
            ctes.append(self._parse_one_cte())
        return ctes

    def _parse_one_cte(self) -> tuple[str, SelectStatement]:
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.AS)
        self.expect(TokenType.LPAREN)
        stmt = self.parse_select()
        self.expect(TokenType.RPAREN)
        return (name, stmt)

    # ── SELECT statement ───────────────────────────────────────────────────

    def parse_select(self) -> SelectStatement:
        self.expect(TokenType.SELECT)

        distinct = False
        if self.match(TokenType.DISTINCT):
            self.advance()
            distinct = True

        columns = self.parse_select_list()
        self.expect(TokenType.FROM)
        from_table = self.expect(TokenType.IDENT).value

        # SAMPLE(n) or SAMPLE n
        sample_pct = None
        if self.match(TokenType.SAMPLE):
            self.advance()
            if self.match(TokenType.LPAREN):
                self.advance()
                sample_pct = float(self.expect(TokenType.NUMBER).value)
                # optional % sign
                if self.match(TokenType.PERCENT):
                    self.advance()
                self.expect(TokenType.RPAREN)
            else:
                sample_pct = float(self.expect(TokenType.NUMBER).value)
                if self.match(TokenType.PERCENT):
                    self.advance()

        joins = []
        while self.match(TokenType.INNER, TokenType.LEFT, TokenType.JOIN):
            joins.append(self.parse_join())

        where = None
        if self.match(TokenType.WHERE):
            self.advance()
            where = self.parse_expr()

        group_by = []
        if self.match(TokenType.GROUP_BY):
            self.advance()
            group_by = self.parse_expr_list()

        having = None
        if self.match(TokenType.HAVING):
            self.advance()
            having = self.parse_expr()

        order_by = []
        if self.match(TokenType.ORDER_BY):
            self.advance()
            order_by = self.parse_order_list()

        limit = None
        if self.match(TokenType.LIMIT):
            self.advance()
            limit = int(self.expect(TokenType.NUMBER).value)

        return SelectStatement(
            columns=columns,
            from_table=from_table,
            where=where,
            group_by=group_by,
            order_by=order_by,
            limit=limit,
            joins=joins,
            having=having,
            distinct=distinct,
            sample_pct=sample_pct,
        )

    # ── Select list ────────────────────────────────────────────────────────

    def parse_select_list(self) -> list:
        if self.match(TokenType.STAR):
            self.advance()
            return [Star()]
        items = [self.parse_select_item()]
        while self.match(TokenType.COMMA):
            self.advance()
            items.append(self.parse_select_item())
        return items

    def parse_select_item(self):
        expr = self.parse_expr()
        alias = None
        if self.match(TokenType.AS):
            self.advance()
            alias = self.expect(TokenType.IDENT).value
        if isinstance(expr, Column) and alias:
            expr.alias = alias
        elif isinstance(expr, FunctionCall) and alias:
            expr.alias = alias
        elif alias:
            expr = Column(name=str(expr), alias=alias)
        return expr

    # ── Join ───────────────────────────────────────────────────────────────

    def parse_join(self) -> JoinClause:
        join_type = "INNER"
        if self.match(TokenType.LEFT):
            self.advance()
            join_type = "LEFT"
        elif self.match(TokenType.INNER):
            self.advance()
        self.expect(TokenType.JOIN)
        table = self.expect(TokenType.IDENT).value
        self.expect(TokenType.ON)
        condition = self.parse_expr()
        return JoinClause(join_type=join_type, table=table, condition=condition)

    # ── Expression parsing (Pratt-style precedence) ─────────────────────

    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.match(TokenType.OR):
            self.advance()
            right = self.parse_and()
            left = BinaryOp("OR", left, right)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.match(TokenType.AND):
            self.advance()
            right = self.parse_not()
            left = BinaryOp("AND", left, right)
        return left

    def parse_not(self):
        if self.match(TokenType.NOT):
            self.advance()
            return UnaryOp("NOT", self.parse_comparison())
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_additive()
        ops = {
            TokenType.EQ: "=", TokenType.NEQ: "!=",
            TokenType.LT: "<", TokenType.LTE: "<=",
            TokenType.GT: ">", TokenType.GTE: ">=",
            TokenType.LIKE: "LIKE",
        }
        if self.current().type in ops:
            op = ops[self.advance().type]
            right = self.parse_additive()
            return BinaryOp(op, left, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplicative()
            left = BinaryOp(op, left, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.match(TokenType.STAR, TokenType.SLASH):
            op = self.advance().value
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left

    def parse_unary(self):
        if self.match(TokenType.MINUS):
            self.advance()
            return UnaryOp("-", self.parse_primary())
        return self.parse_primary()

    def parse_primary(self):
        tok = self.current()

        if tok.type == TokenType.NUMBER:
            self.advance()
            dtype = "float" if isinstance(tok.value, float) else "int"
            return Literal(tok.value, dtype)

        if tok.type == TokenType.STRING:
            self.advance()
            return Literal(tok.value, "str")

        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN)
            return expr

        if tok.type == TokenType.IDENT:
            name = self.advance().value
            # function call?
            if self.match(TokenType.LPAREN):
                self.advance()
                is_agg = name.lower() in AGGREGATE_FUNCTIONS
                if self.match(TokenType.STAR):
                    self.advance()
                    args = [Star()]
                elif self.match(TokenType.RPAREN):
                    args = []
                else:
                    args = self.parse_expr_list()
                self.expect(TokenType.RPAREN)
                func = FunctionCall(name=name.upper(), args=args, is_aggregate=is_agg)
                # window function: check for OVER (...)
                if self.match(TokenType.OVER):
                    self.advance()
                    func.over = self.parse_window_spec()
                    func.is_aggregate = False  # window fns are not group aggregates
                return func
            # table.column reference?
            if self.match(TokenType.DOT):
                self.advance()
                col = self.expect(TokenType.IDENT).value
                return Column(name=col, table=name)
            return Column(name=name)

        if tok.type == TokenType.STAR:
            self.advance()
            return Star()

        raise self.error("Unexpected token in expression")

    # ── Window spec ────────────────────────────────────────────────────────

    def parse_window_spec(self) -> WindowSpec:
        """Parse OVER (PARTITION BY ... ORDER BY ...)."""
        self.expect(TokenType.LPAREN)
        partition_by = []
        order_by = []
        if self.match(TokenType.PARTITION_BY):
            self.advance()
            partition_by = self.parse_expr_list()
        if self.match(TokenType.ORDER_BY):
            self.advance()
            order_by = self.parse_order_list()
        self.expect(TokenType.RPAREN)
        return WindowSpec(partition_by=partition_by, order_by=order_by)

    # ── Helpers ────────────────────────────────────────────────────────────

    def parse_expr_list(self) -> list:
        items = [self.parse_expr()]
        while self.match(TokenType.COMMA):
            self.advance()
            items.append(self.parse_expr())
        return items

    def parse_order_list(self) -> list[tuple]:
        items = []
        expr = self.parse_expr()
        direction = "ASC"
        if self.match(TokenType.IDENT) and self.current().value.upper() in ("ASC", "DESC"):
            direction = self.advance().value.upper()
        items.append((expr, direction))
        while self.match(TokenType.COMMA):
            self.advance()
            expr = self.parse_expr()
            direction = "ASC"
            if self.match(TokenType.IDENT) and self.current().value.upper() in ("ASC", "DESC"):
                direction = self.advance().value.upper()
            items.append((expr, direction))
        return items


def parse(sql: str) -> SelectStatement:
    tokens = tokenize(sql)
    return Parser(tokens).parse()
