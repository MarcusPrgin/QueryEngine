"""
Table catalog: maps logical table names to physical file paths.

Allows queries like SELECT * FROM orders instead of SELECT * FROM /data/orders.csv
Also handles schema caching and table registration.
"""
from __future__ import annotations
import os
from src.types import Schema
from src.storage.reader import get_reader


class CatalogError(Exception):
    pass


class TableCatalog:
    def __init__(self):
        self._tables: dict[str, str] = {}  # name -> file path
        self._schemas: dict[str, Schema] = {}

    def register(self, name: str, path: str):
        """Register a file as a named table."""
        if not os.path.exists(path):
            raise CatalogError(f"File not found: {path}")
        self._tables[name] = os.path.abspath(path)

    def register_directory(self, directory: str):
        """Auto-register all supported files in a directory as tables."""
        for fname in os.listdir(directory):
            name, ext = os.path.splitext(fname)
            if ext.lower() in (".csv", ".jsonl", ".parquet"):
                self.register(name, os.path.join(directory, fname))

    def resolve(self, name: str) -> str:
        """Return the file path for a table name, or treat name as a path."""
        if name in self._tables:
            return self._tables[name]
        if os.path.exists(name):
            return name
        raise CatalogError(
            f"Table '{name}' not found. "
            f"Register it with: engine.catalog.register('{name}', '/path/to/file.csv')\n"
            f"Available tables: {list(self._tables.keys())}"
        )

    def schema(self, name: str) -> Schema:
        if name not in self._schemas:
            path = self.resolve(name)
            reader = get_reader(path)
            self._schemas[name] = reader.schema
        return self._schemas[name]

    def list_tables(self) -> list[dict]:
        result = []
        for name, path in self._tables.items():
            size = os.path.getsize(path) if os.path.exists(path) else 0
            result.append({"name": name, "path": path, "size_bytes": size})
        return result
