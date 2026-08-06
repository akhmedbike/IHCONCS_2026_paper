"""Paired bootstrap significance test for the correction experiments.

Reviewer #2 explicitly requested a statistical-significance test between the
proposed system and the strongest baseline. The natural pairing unit is the
(fold, seed) run: both the three-way and the binary configurations are
evaluated on the *same* corrupted sentence pairs in `eval_correction.py`, so
their per-run F₀.₅ vectors are aligned index-by-index. This module implements
a paired bootstrap (resampling the paired differences with replacement) and a
two-sided Wilcoxon signed-rank fallback, both computed from numpy only — no
scipy dependency.

References the method to the protocol the paper promises (Section III,
"Statistical analysis"): paired bootstrap, 10,000 resamples, two-sided.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test of H0: mean(a - b) == 0.

    Resamples the paired differences (a_i - b_i) with replacement and counts
    how often the resampled mean difference is <= 0 (for a positive observed
    difference). The reported p_value is two-sided: 2 * min(p_one_sided,
    1 - p_one_sided), clipped to [0, 1].

    Returns:
        {
          "mean_diff": float,         # mean(a) - mean(b)
          "ci95": [lo, hi],           # 95% percentile CI of the mean difference
          "p_value": float,           # two-sided
          "n_resamples": int,
          "n_pairs": int,
        }
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape or a_arr.ndim != 1:
        raise ValueError("a and b must be 1-D sequences of equal length")
    n = a_arr.size
    if n < 2:
        raise ValueError("need at least 2 paired observations")

    diffs = a_arr - b_arr
    obs_mean = float(diffs.mean())

    rng = np.random.default_rng(seed)
    # Resample indices with replacement, take the mean of the resampled diffs.
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)

    # One-sided fraction at/below zero (for a positive observed mean).
    p_one = float(np.mean(boot_means <= 0.0))
    p_two = 2.0 * min(p_one, 1.0 - p_one)
    p_two = min(max(p_two, 0.0), 1.0)

    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))

    return {
        "mean_diff": obs_mean,
        "ci95": [lo, hi],
        "p_value": p_two,
        "n_resamples": n_resamples,
        "n_pairs": n,
    }


def wilcoxon_signed_rank(
    a: Sequence[float],
    b: Sequence[float],
) -> dict:
    """Two-sided Wilcoxon signed-rank test (exact statistic, normal-approx p).

    A distribution-free cross-check on the bootstrap. Computed without scipy:
    the test statistic W is the sum of ranks of the absolute differences,
    signed by the sign of the difference; the p-value uses the normal
    approximation (valid for n >= ~10; with ties handled by the standard
    variance correction).

    Returns {"w_statistic": float, "z": float, "p_value": float, "n_pairs": int}.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diffs = a_arr - b_arr
    # Drop exact zeros (no preference) — standard Wilcoxon handling.
    nonzero = diffs[diffs != 0]
    n = nonzero.size
    if n < 1:
        return {"w_statistic": 0.0, "z": 0.0, "p_value": 1.0, "n_pairs": int(n)}

    abs_d = np.abs(nonzero)
    # Average ranks for ties.
    order = np.argsort(abs_d, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    signs = np.sign(nonzero)
    w_plus = float(np.sum(ranks[signs > 0]))
    w_minus = float(np.sum(ranks[signs < 0]))
    w_stat = min(w_plus, w_minus)

    # Normal approximation for the null distribution of W (n >= 10).
    # Mean = n(n+1)/4; tie-corrected variance.
    tie_counts = np.unique(abs_d, return_counts=True)[1]
    tie_term = float(np.sum(tie_counts ** 3 - tie_counts))
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return {"w_statistic": w_stat, "z": 0.0, "p_value": 1.0, "n_pairs": int(n)}
    z = (w_stat - mu) / math.sqrt(var)
    # Two-sided p from the standard normal survival function (erfc).
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"w_statistic": w_stat, "z": float(z), "p_value": float(p), "n_pairs": int(n)}
