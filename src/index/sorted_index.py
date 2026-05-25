"""
Sorted index for fast column lookups.

Design:
- Index maps column values -> list of row positions (0-based line numbers)
- Persisted as a sidecar .idx file next to the data file
- Invalidated when source file mtime changes
- Built lazily on first query that references an indexed column

Interview talking point: this index is "sparse" relative to a B-tree.
We store all distinct values, so lookup is O(log n) via bisect.
A B-tree would give O(log n) with better cache locality for range scans,
but implementing a B-tree from scratch is a week of work.
"""
from __future__ import annotations
import bisect
import json
import os
import pickle
import time
from typing import Any, Optional
from src.storage.reader import CSVReader


class IndexError(Exception):
    pass


class SortedIndex:
    """
    Sorted index on a single column of a CSV file.
    
    Storage format: pickle file containing:
        {
            "source_mtime": float,
            "column": str,
            "keys": [val, val, ...],        # sorted unique values
            "offsets": [[pos,...], [pos,..]] # row positions per key
        }
    """

    def __init__(self, source_path: str, column: str):
        self.source_path = source_path
        self.column = column
        self.index_path = f"{source_path}.{column}.idx"
        self._keys: list[Any] = []
        self._offsets: list[list[int]] = []
        self._none_count: int = 0   # number of leading None entries (sorted first)
        self._built = False

    def _is_stale(self) -> bool:
        if not os.path.exists(self.index_path):
            return True
        with open(self.index_path, "rb") as f:
            meta = pickle.load(f)
        source_mtime = os.path.getmtime(self.source_path)
        return meta.get("source_mtime", 0) != source_mtime

    def build(self, force: bool = False) -> float:
        """
        Build the index by scanning the source file once.
        Returns build time in seconds.
        This is the O(n) cost you pay once to get O(log n) lookups forever.
        """
        if self._built and not force:
            return 0.0
        if not force and not self._is_stale():
            self._load()
            return 0.0

        start = time.perf_counter()
        reader = CSVReader(self.source_path)

        # Build: value -> [row_positions]
        index: dict[Any, list[int]] = {}
        for row_num, row in enumerate(reader.scan(columns=[self.column])):
            val = row.get(self.column)
            if val not in index:
                index[val] = []
            index[val].append(row_num)

        # Sort keys for binary search (None values sort first)
        self._keys = sorted(index.keys(), key=lambda x: (x is None, x))
        self._offsets = [index[k] for k in self._keys]
        self._none_count = sum(1 for k in self._keys if k is None)
        elapsed = time.perf_counter() - start

        # Persist
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "source_mtime": os.path.getmtime(self.source_path),
                "column": self.column,
                "keys": self._keys,
                "offsets": self._offsets,
            }, f)

        self._built = True
        return elapsed

    def _load(self):
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self._keys = data["keys"]
        self._offsets = data["offsets"]
        self._none_count = sum(1 for k in self._keys if k is None)
        self._built = True

    def _ensure_built(self):
        if not self._built:
            self.build()

    def lookup_eq(self, value: Any) -> list[int]:
        """Return row positions where column == value. O(log n)."""
        self._ensure_built()
        if not self._keys:
            return []
        if value is None:
            return self._offsets[0] if self._none_count > 0 else []
        # Non-None values start at index _none_count (Nones sort first)
        start = self._none_count
        try:
            i = bisect.bisect_left(self._keys, value, start)
            if i < len(self._keys) and self._keys[i] == value:
                return self._offsets[i]
        except TypeError:
            for i in range(start, len(self._keys)):
                if self._keys[i] == value:
                    return self._offsets[i]
        return []

    def lookup_range(self, lo: Any, hi: Any,
                     lo_inclusive: bool = True,
                     hi_inclusive: bool = True) -> list[int]:
        """Return row positions for a range query. O(log n + k) where k = matching rows."""
        self._ensure_built()
        if not self._keys:
            return []
        start = self._none_count
        try:
            i_lo = bisect.bisect_left(self._keys, lo, start)
            if not lo_inclusive and i_lo < len(self._keys) and self._keys[i_lo] == lo:
                i_lo += 1
            i_hi = bisect.bisect_right(self._keys, hi, start)
            if not hi_inclusive and i_hi > 0 and self._keys[i_hi - 1] == hi:
                i_hi -= 1
            result: list[int] = []
            for offsets in self._offsets[i_lo:i_hi]:
                result.extend(offsets)
            return sorted(result)
        except TypeError:
            result = []
            for i in range(start, len(self._keys)):
                k = self._keys[i]
                try:
                    above_lo = (k >= lo) if lo_inclusive else (k > lo)
                    below_hi = (k <= hi) if hi_inclusive else (k < hi)
                    if above_lo and below_hi:
                        result.extend(self._offsets[i])
                except TypeError:
                    pass
            return sorted(result)

    def stats(self) -> dict:
        self._ensure_built()
        return {
            "column": self.column,
            "distinct_values": len(self._keys),
            "index_path": self.index_path,
            "index_size_bytes": os.path.getsize(self.index_path) if os.path.exists(self.index_path) else 0,
        }


class IndexCatalog:
    """Registry of all indexes for a file."""

    def __init__(self):
        self._indexes: dict[tuple[str, str], SortedIndex] = {}

    def create(self, source: str, column: str, force: bool = False) -> tuple[SortedIndex, float]:
        key = (source, column)
        if key not in self._indexes:
            self._indexes[key] = SortedIndex(source, column)
        idx = self._indexes[key]
        elapsed = idx.build(force=force)
        return idx, elapsed

    def get(self, source: str, column: str) -> Optional[SortedIndex]:
        key = (source, column)
        idx = self._indexes.get(key)
        if idx and idx._built:
            return idx
        return None

    def list_indexes(self) -> list[dict]:
        return [idx.stats() for idx in self._indexes.values() if idx._built]


# Module-level singleton
catalog = IndexCatalog()
