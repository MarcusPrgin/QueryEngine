"""
Bloom filter: probabilistic set membership in O(1) time and O(m) space.

Used in hash joins to skip probe-side lookups for keys guaranteed absent
from the build side. A miss is definitive (no false negatives); a hit may be
wrong (false positive rate controlled by m and k).

Math:
    optimal k (hash fns) = (m/n) * ln2
    false positive rate  = (1 - e^(-kn/m))^k

With m=8*expected_items bits and k=6 hash functions, FP rate ≈ 2%.
"""
from __future__ import annotations
import math
import struct
import hashlib


class BloomFilter:
    """
    Bit-array bloom filter backed by a bytearray.

    Parameters
    ----------
    expected_items : int   estimated number of distinct values to insert
    fp_rate        : float desired false-positive rate (default 0.02 = 2%)
    """

    def __init__(self, expected_items: int = 10_000, fp_rate: float = 0.02):
        if expected_items <= 0:
            expected_items = 1
        # Optimal bit-array size and hash function count
        m = max(64, -int(expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        k = max(1, int((m / expected_items) * math.log(2)))
        self._m = m
        self._k = k
        self._bits = bytearray((m + 7) // 8)
        self._count = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _hashes(self, value) -> list[int]:
        """
        Generate k independent hash values using double-hashing:
        h_i(x) = (h1(x) + i * h2(x)) % m
        This avoids computing k independent cryptographic hashes.
        """
        raw = str(value).encode()
        h1 = int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")
        h2 = int.from_bytes(hashlib.sha256(raw).digest()[4:8], "big") | 1  # must be odd
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def _set_bit(self, i: int):
        self._bits[i >> 3] |= 1 << (i & 7)

    def _get_bit(self, i: int) -> bool:
        return bool(self._bits[i >> 3] & (1 << (i & 7)))

    # ── public ────────────────────────────────────────────────────────────────

    def add(self, value) -> None:
        for h in self._hashes(value):
            self._set_bit(h)
        self._count += 1

    def might_contain(self, value) -> bool:
        """Return False only if value is definitely absent (no false negatives)."""
        return all(self._get_bit(h) for h in self._hashes(value))

    def __contains__(self, value) -> bool:
        return self.might_contain(value)

    @property
    def size_bytes(self) -> int:
        return len(self._bits)

    @property
    def item_count(self) -> int:
        return self._count

    def estimated_fp_rate(self) -> float:
        """Current empirical false-positive rate given items added so far."""
        if self._count == 0:
            return 0.0
        return (1 - math.exp(-self._k * self._count / self._m)) ** self._k
