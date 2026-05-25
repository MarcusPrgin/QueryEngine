"""
HyperLogLog: approximate COUNT DISTINCT in O(1) memory.

This is the single most impressive addition you can make to the aggregation
layer. Standard COUNT DISTINCT requires storing every distinct value in a
set — O(n) memory. HyperLogLog uses ~12KB regardless of cardinality, with
~0.8% error.

How it works (the intuition):
- Hash each value to a uniform bit string
- Count the longest run of leading zeros in any hash
- The longer the run, the more values you've likely seen
- "If I saw a hash with 20 leading zeros, I probably hashed ~2^20 values"
- Use multiple buckets (registers) to reduce variance, then take harmonic mean

This is what Redshift, BigQuery, and Redis use for HLL COUNT DISTINCT.

Interview line: "I added HyperLogLog for COUNT DISTINCT. It uses 12KB of
memory regardless of cardinality, with ~0.8% error. The tradeoff vs. exact
counting: you can't recover the individual values, only the cardinality estimate.
That's fine for analytics — nobody needs the exact count to 15 significant figures."
"""
from __future__ import annotations
import hashlib
import math
import struct


class HyperLogLog:
    """
    HyperLogLog cardinality estimator.
    
    b = 14 registers → ~0.8% error, 16KB memory
    Suitable for cardinalities from ~100 to ~10^18.
    """

    def __init__(self, b: int = 14):
        self.b = b                        # number of register bits
        self.m = 1 << b                   # number of registers = 2^b
        self.registers = [0] * self.m
        self._alpha = self._compute_alpha(self.m)

    @staticmethod
    def _compute_alpha(m: int) -> float:
        if m == 16:   return 0.673
        if m == 32:   return 0.697
        if m == 64:   return 0.709
        return 0.7213 / (1 + 1.079 / m)

    def _hash(self, value) -> int:
        """Hash any value to a 64-bit integer."""
        h = hashlib.md5(str(value).encode()).digest()
        return struct.unpack(">Q", h[:8])[0]

    def add(self, value):
        """Add a value to the HLL sketch."""
        x = self._hash(value)
        # use top b bits as register index
        j = x >> (64 - self.b)
        # count leading zeros in remaining 64-b bits
        w = x & ((1 << (64 - self.b)) - 1)
        rho = self._leading_zeros(w, 64 - self.b)
        # keep maximum
        if rho + 1 > self.registers[j]:
            self.registers[j] = rho + 1

    @staticmethod
    def _leading_zeros(x: int, max_bits: int) -> int:
        if x == 0:
            return max_bits
        count = 0
        for i in range(max_bits - 1, -1, -1):
            if x & (1 << i):
                break
            count += 1
        return count

    def count(self) -> int:
        """Estimate the cardinality."""
        # harmonic mean of 2^register values
        z = sum(2.0 ** (-r) for r in self.registers)
        raw = self._alpha * self.m * self.m / z

        # small range correction
        if raw <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros > 0:
                raw = self.m * math.log(self.m / zeros)

        # large range correction (not needed for most practical cases)
        elif raw > (1 << 32) / 30:
            raw = -(1 << 32) * math.log(1 - raw / (1 << 32))

        return int(raw)

    def merge(self, other: "HyperLogLog"):
        """Merge another HLL into this one — enables parallel counting."""
        assert self.b == other.b, "Cannot merge HLLs with different precision"
        for i in range(self.m):
            self.registers[i] = max(self.registers[i], other.registers[i])

    def memory_bytes(self) -> int:
        return self.m  # 1 byte per register


class ExactCounter:
    """Exact COUNT DISTINCT using a set. O(n) memory."""

    def __init__(self):
        self._seen: set = set()

    def add(self, value):
        self._seen.add(value)

    def count(self) -> int:
        return len(self._seen)


def count_distinct(values, approximate: bool = True) -> int:
    """
    Count distinct values, either exactly or approximately.
    
    approximate=True: HyperLogLog, O(1) memory, ~0.8% error
    approximate=False: exact set, O(n) memory, 0% error
    """
    if approximate:
        hll = HyperLogLog(b=14)
        for v in values:
            hll.add(v)
        return hll.count()
    else:
        return len(set(values))
