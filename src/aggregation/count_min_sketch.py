"""
Count-Min Sketch: sub-linear frequency estimation for streaming data.

Answers "approximately how many times has value X appeared?" in O(1) time
and O(width * depth) space — independent of the number of distinct values.

Used to implement TOPK(column, k) — the approximate top-k most frequent
values in a column. Standard COUNT(DISTINCT) or GROUP BY requires O(n)
memory; Count-Min + heap requires O(width*depth + k).

Error bound: estimate ≤ true_count + ε * N  with probability ≥ 1 - δ
    where ε = e/width, δ = 1/e^depth   (e ≈ 2.718)

Default: width=2000, depth=7 → ε ≈ 0.14%, δ ≈ 0.09%
"""
from __future__ import annotations
import heapq
import array


class CountMinSketch:
    def __init__(self, width: int = 2000, depth: int = 7):
        self.width = width
        self.depth = depth
        # Use array.array('l') for memory efficiency vs list of ints
        self._table = [array.array("l", [0] * width) for _ in range(depth)]
        self._seeds = [i * 2_654_435_761 & 0xFFFF_FFFF for i in range(depth)]

    def _hash(self, value, seed: int) -> int:
        return (hash(value) ^ seed) % self.width

    def add(self, value, count: int = 1) -> None:
        for d in range(self.depth):
            j = self._hash(value, self._seeds[d])
            self._table[d][j] += count

    def estimate(self, value) -> int:
        return min(
            self._table[d][self._hash(value, self._seeds[d])]
            for d in range(self.depth)
        )

    def memory_bytes(self) -> int:
        return self.width * self.depth * 8  # 8 bytes per long


class TopKTracker:
    """
    Tracks the approximate top-k most frequent values using a Count-Min
    Sketch for frequency estimation and a min-heap of size k for candidates.

    Add all values via add(), then call topk() to retrieve results.
    """

    def __init__(self, k: int, width: int = 2000, depth: int = 7):
        self.k = k
        self._cms = CountMinSketch(width, depth)
        self._candidates: set = set()
        self._heap: list[tuple[int, str]] = []  # (neg_count, str_val) min-heap

    def add(self, value) -> None:
        self._cms.add(value)
        self._candidates.add(value)

    def topk(self) -> list[tuple[int, any]]:
        """Return list of (estimated_count, value) sorted by count desc."""
        scored = [(self._cms.estimate(v), v) for v in self._candidates]
        # partial sort — only need top k
        top = heapq.nlargest(self.k, scored, key=lambda x: x[0])
        return top
