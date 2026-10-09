"""Metric math for repeated-trial evals.

With ``n`` trials of a case and ``c`` successes:

* ``pass@1  = c / n``
* ``pass@k  = 1 - C(n-c, k) / C(n, k)``  — P(at least one of k random trials passes)
* ``pass^k  = C(c, k) / C(n, k)``        — P(all k random trials pass), as in
  tau-bench (Yao et al., 2024). It measures *reliability*, and drops fast when
  a node is flaky even if pass@1 looks fine.

Both are unbiased estimators when ``n >= k``; with ``n == k`` pass^k is 1 only
if every trial passed. Dataset-level values are the mean over cases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["mean", "pass_at_k", "pass_hat_k", "percentile"]


def _check(n: int, c: int, k: int) -> None:
    if n < 1 or k < 1:
        raise ValueError("n and k must be >= 1")
    if not 0 <= c <= n:
        raise ValueError("successes must be within [0, n]")
    if k > n:
        raise ValueError(f"k={k} exceeds the number of trials n={n}")


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: probability that k trials drawn without replacement all pass."""
    _check(n, c, k)
    return math.comb(c, k) / math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k: probability that at least one of k drawn trials passes."""
    _check(n, c, k)
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in [0, 100]); ``None`` if empty."""
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None
