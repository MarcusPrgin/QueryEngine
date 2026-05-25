"""
Lexer: converts a SQL string into a flat list of Tokens.

Design note: we scan character by character using a cursor.
The lexer doesn't understand SQL grammar — it only recognises
individual tokens. The parser handles structure.
"""
from __future__ import annotations
from src.types import Token, TokenType


KEYWORDS: dict[str, TokenType] = {
    "select": TokenType.SELECT,
    "from":   TokenType.FROM,
    "where":  TokenType.WHERE,
    "group":  TokenType.GROUP_BY,   # "GROUP BY" handled as one token
    "order":  TokenType.ORDER_BY,   # "ORDER BY" handled as one token
    "limit":  TokenType.LIMIT,
    "and":    TokenType.AND,
    "or":     TokenType.OR,
    "not":    TokenType.NOT,
    "as":     TokenType.AS,
    "join":   TokenType.JOIN,
    "on":     TokenType.ON,
    "inner":  TokenType.INNER,
    "left":   TokenType.LEFT,
    "like":   TokenType.LIKE,
    "by":     None,   # consumed as part of GROUP BY / ORDER BY
}

AGGREGATE_FUNCTIONS = {"sum", "count", "avg", "min", "max", "count_distinct"}


class LexError(Exception):
    pass


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens: list[Token] = []

    def error(self, msg: str) -> LexError:
        snippet = self.text[max(0, self.pos-10):self.pos+10]
        return LexError(f"{msg} at position {self.pos} (near {snippet!r})")

    def peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.text[idx] if idx < len(self.text) else ""

    def advance(self) -> str:
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def skip_whitespace_and_comments(self):
        while self.pos < len(self.text):
            if self.text[self.pos] in " \t\n\r":
                self.pos += 1
            elif self.text[self.pos:self.pos+2] == "--":
                # single-line comment
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
            else:
                break

    def read_number(self) -> Token:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == "."):
            self.pos += 1
        raw = self.text[start:self.pos]
        value = float(raw) if "." in raw else int(raw)
        return Token(TokenType.NUMBER, value, start)

    def read_string(self) -> Token:
        start = self.pos
        quote = self.advance()   # consume opening quote
        buf: list[str] = []
        while self.pos < len(self.text):
            ch = self.advance()
            if ch == "\\" and self.peek() == quote:
                buf.append(self.advance())
            elif ch == quote:
                break
            else:
                buf.append(ch)
        else:
            raise self.error("Unterminated string literal")
        return Token(TokenType.STRING, "".join(buf), start)

    def read_ident_or_keyword(self) -> Token:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1
        raw = self.text[start:self.pos]
        lower = raw.lower()

        # handle multi-word keywords
        if lower == "group":
            self.skip_whitespace_and_comments()
            if self.text[self.pos:self.pos+2].lower() == "by":
                self.pos += 2
            return Token(TokenType.GROUP_BY, "GROUP BY", start)
        if lower == "order":
            self.skip_whitespace_and_comments()
            if self.text[self.pos:self.pos+2].lower() == "by":
                self.pos += 2
            return Token(TokenType.ORDER_BY, "ORDER BY", start)

        tt = KEYWORDS.get(lower)
        if tt is not None:
            return Token(tt, lower, start)
        if lower in AGGREGATE_FUNCTIONS:
            return Token(TokenType.IDENT, raw, start)
        return Token(TokenType.IDENT, raw, start)

    def tokenize(self) -> list[Token]:
        while True:
            self.skip_whitespace_and_comments()
            if self.pos >= len(self.text):
                self.tokens.append(Token(TokenType.EOF, None, self.pos))
                break

            pos = self.pos
            ch = self.text[self.pos]

            if ch.isdigit() or (ch == "-" and self.peek(1).isdigit() and (not self.tokens or self.tokens[-1].type in (TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.LTE, TokenType.GT, TokenType.GTE, TokenType.COMMA, TokenType.LPAREN))):
                self.tokens.append(self.read_number())
            elif ch in ("'", '"'):
                self.tokens.append(self.read_string())
            elif ch.isalpha() or ch == "_":
                self.tokens.append(self.read_ident_or_keyword())
            elif ch == "*":
                self.advance(); self.tokens.append(Token(TokenType.STAR, "*", pos))
            elif ch == ",":
                self.advance(); self.tokens.append(Token(TokenType.COMMA, ",", pos))
            elif ch == "(":
                self.advance(); self.tokens.append(Token(TokenType.LPAREN, "(", pos))
            elif ch == ")":
                self.advance(); self.tokens.append(Token(TokenType.RPAREN, ")", pos))
            elif ch == ".":
                self.advance(); self.tokens.append(Token(TokenType.DOT, ".", pos))
            elif ch == ";":
                self.advance(); self.tokens.append(Token(TokenType.SEMICOLON, ";", pos))
            elif ch == "+":
                self.advance(); self.tokens.append(Token(TokenType.PLUS, "+", pos))
            elif ch == "/":
                self.advance(); self.tokens.append(Token(TokenType.SLASH, "/", pos))
            elif ch == "=":
                self.advance(); self.tokens.append(Token(TokenType.EQ, "=", pos))
            elif ch == "!" and self.peek(1) == "=":
                self.pos += 2; self.tokens.append(Token(TokenType.NEQ, "!=", pos))
            elif ch == "<" and self.peek(1) == "=":
                self.pos += 2; self.tokens.append(Token(TokenType.LTE, "<=", pos))
            elif ch == ">" and self.peek(1) == "=":
                self.pos += 2; self.tokens.append(Token(TokenType.GTE, ">=", pos))
            elif ch == "<":
                self.advance(); self.tokens.append(Token(TokenType.LT, "<", pos))
            elif ch == ">":
                self.advance(); self.tokens.append(Token(TokenType.GT, ">", pos))
            elif ch == "-":
                self.advance(); self.tokens.append(Token(TokenType.MINUS, "-", pos))
            else:
                raise self.error(f"Unexpected character {ch!r}")

        return self.tokens


def tokenize(sql: str) -> list[Token]:
    return Lexer(sql).tokenize()
