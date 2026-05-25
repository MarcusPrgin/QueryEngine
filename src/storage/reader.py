"""
Storage layer: streaming readers for CSV and JSON files.

Key design decisions:
1. All readers are generators — they never load the full file into memory.
2. Type inference happens on the first N rows (configurable).
3. Buffer size is 64KB — aligns with OS page cache, keeps syscall cost low.
4. Parquet support is optional (requires pyarrow) — degrades gracefully.
"""
from __future__ import annotations
import csv
import io
import json
import os
from typing import Any, Generator, Optional
from src.types import Row, Schema

# Read buffer size — 64KB is a sweet spot: fits in CPU L2 cache,
# aligns with OS page size multiples, and keeps read() syscall count low.
BUFFER_SIZE = 64 * 1024


def infer_type(value: str) -> tuple[Any, str]:
    """Try to cast a string value to its most specific type."""
    if value == "" or value.lower() in ("null", "none", "na", "n/a"):
        return None, "null"
    try:
        return int(value), "int"
    except ValueError:
        pass
    try:
        return float(value), "float"
    except ValueError:
        pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true", "bool"
    return value, "str"


def infer_schema(rows: list[Row]) -> Schema:
    """Infer schema from a sample of rows. Most specific type wins."""
    if not rows:
        return {}
    schema: Schema = {}
    for col in rows[0]:
        types = set()
        for row in rows:
            _, dtype = infer_type(str(row.get(col, "")))
            types.add(dtype)
        # type precedence: null < bool < int < float < str
        if "str" in types:
            schema[col] = "str"
        elif "float" in types:
            schema[col] = "float"
        elif "int" in types:
            schema[col] = "int"
        elif "bool" in types:
            schema[col] = "bool"
        else:
            schema[col] = "null"
    return schema


class CSVReader:
    """
    Streaming CSV reader. Reads one row at a time, never loads the full file.
    
    The streaming trick: csv.reader works on any iterable, including a file
    object opened in text mode. We wrap it in a generator and yield one dict
    per row. Memory usage is O(1) relative to file size.
    """
    def __init__(self, path: str, infer_types: bool = True,
                 sample_rows: int = 100, encoding: str = "utf-8"):
        self.path = path
        self.infer_types = infer_types
        self.sample_rows = sample_rows
        self.encoding = encoding
        self._schema: Optional[Schema] = None

    @property
    def schema(self) -> Schema:
        if self._schema is None:
            sample = list(self._read_raw(limit=self.sample_rows))
            self._schema = infer_schema(sample)
        return self._schema

    def _read_raw(self, limit: Optional[int] = None) -> Generator[Row, None, None]:
        """Read rows as strings — no type casting."""
        with open(self.path, "r", encoding=self.encoding,
                  buffering=BUFFER_SIZE, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                yield dict(row)

    def scan(self, columns: Optional[list[str]] = None) -> Generator[Row, None, None]:
        """
        Stream rows from the CSV file.
        columns: if set, only include those columns (projection pushdown).
        """
        schema = self.schema if self.infer_types else {}
        with open(self.path, "r", encoding=self.encoding,
                  buffering=BUFFER_SIZE, newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                row: Row = {}
                for key, val in raw_row.items():
                    if columns and key not in columns:
                        continue
                    if self.infer_types and key in schema:
                        typed_val, _ = infer_type(val)
                        row[key] = typed_val
                    else:
                        row[key] = val
                yield row

    def row_count(self) -> int:
        """Count rows without loading into memory."""
        count = 0
        with open(self.path, "r", encoding=self.encoding,
                  buffering=BUFFER_SIZE, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for _ in reader:
                count += 1
        return count


class JSONLReader:
    """
    Streaming newline-delimited JSON (JSONL) reader.
    Each line is a separate JSON object.
    """
    def __init__(self, path: str, infer_types: bool = True, sample_rows: int = 100):
        self.path = path
        self.infer_types = infer_types
        self.sample_rows = sample_rows
        self._schema: Optional[Schema] = None

    @property
    def schema(self) -> Schema:
        if self._schema is None:
            sample = []
            for row in self.scan():
                sample.append(row)
                if len(sample) >= self.sample_rows:
                    break
            self._schema = infer_schema(sample)
        return self._schema

    def scan(self, columns: Optional[list[str]] = None) -> Generator[Row, None, None]:
        with open(self.path, "r", buffering=BUFFER_SIZE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if columns:
                    row = {k: v for k, v in row.items() if k in columns}
                yield row


class ParquetReader:
    """
    Columnar Parquet reader using pyarrow.
    Falls back gracefully if pyarrow is not installed.
    
    Why Parquet matters: columnar layout means a query touching 2 of 50 columns
    reads ~4% of the data. CSV always reads 100% regardless of column selection.
    """
    def __init__(self, path: str):
        self.path = path
        self._pf = None

    def _open(self):
        try:
            import pyarrow.parquet as pq
            self._pf = pq.ParquetFile(self.path)
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet support. "
                "Install it with: pip install pyarrow"
            )

    @property
    def schema(self) -> Schema:
        if self._pf is None:
            self._open()
        type_map = {
            "int32": "int", "int64": "int",
            "float32": "float", "float64": "float",
            "bool": "bool",
        }
        result = {}
        for field in self._pf.schema_arrow:
            dtype = str(field.type)
            result[field.name] = type_map.get(dtype, "str")
        return result

    def scan(self, columns: Optional[list[str]] = None) -> Generator[Row, None, None]:
        if self._pf is None:
            self._open()
        # read in batches of 10K rows — columnar batches are very cache-friendly
        for batch in self._pf.iter_batches(batch_size=10_000, columns=columns):
            table = batch.to_pydict()
            keys = list(table.keys())
            for i in range(len(table[keys[0]])):
                yield {k: table[k][i] for k in keys}


def get_reader(path: str):
    """Return the appropriate reader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return CSVReader(path)
    elif ext in (".jsonl", ".ndjson"):
        return JSONLReader(path)
    elif ext == ".parquet":
        return ParquetReader(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .csv, .jsonl, .parquet")
