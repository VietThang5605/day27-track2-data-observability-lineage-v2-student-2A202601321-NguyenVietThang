from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _ks_two_sample(cur: np.ndarray, base: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (no scipy dependency)."""
    cur_sorted = np.sort(cur)
    base_sorted = np.sort(base)
    all_vals = np.concatenate([cur_sorted, base_sorted])
    cdf_cur = np.searchsorted(cur_sorted, all_vals, side="right") / cur_sorted.size
    cdf_base = np.searchsorted(base_sorted, all_vals, side="right") / base_sorted.size
    return float(np.max(np.abs(cdf_cur - cdf_base)))


def detect_distribution_shift(
    current_values: Iterable[float],
    baseline_values: Iterable[float],
    *,
    ratio_threshold: float = 3.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Distribution-shift detector: two-sample KS test over the ECDFs.

    The starter mean-ratio misses shape drift (variance/quantile changes with
    a stable mean). KS compares the full empirical distributions; the critical
    value 1.36*sqrt((n+m)/(n*m)) corresponds to significance level ~0.05.
    A mean-ratio check is kept as an OR condition so extreme location shifts
    on tiny samples are still caught.
    """
    cur = np.asarray(list(current_values), dtype=float)
    base = np.asarray(list(baseline_values), dtype=float)
    if cur.size == 0 or base.size == 0:
        return {"is_anomaly": False, "score": 0.0, "method": "ks", "reason": "empty_input"}

    cur_mean = float(np.mean(cur))
    base_mean = float(np.mean(base))

    ks_stat = _ks_two_sample(cur, base)
    critical = 1.36 * float(np.sqrt((cur.size + base.size) / (cur.size * base.size)))
    ks_anomaly = ks_stat > critical

    if base_mean == 0:
        mean_score = float("inf") if cur_mean != 0 else 1.0
    else:
        mean_score = (
            max(abs(cur_mean / base_mean), abs(base_mean / cur_mean))
            if cur_mean != 0
            else float("inf")
        )
    mean_anomaly = mean_score >= ratio_threshold

    is_anomaly = bool(ks_anomaly or mean_anomaly)
    if ks_anomaly:
        reason = (
            f"ks_stat={ks_stat:.3f} > critical={critical:.3f} (alpha={alpha}); "
            f"baseline_mean={base_mean:.3f}, current_mean={cur_mean:.3f}"
        )
    elif mean_anomaly:
        reason = f"mean_ratio={mean_score:.3f} >= {ratio_threshold}; ks_stat={ks_stat:.3f}"
    else:
        reason = (
            f"ks_stat={ks_stat:.3f} <= critical={critical:.3f}; "
            f"baseline_mean={base_mean:.3f}, current_mean={cur_mean:.3f}"
        )
    return {
        "is_anomaly": is_anomaly,
        "score": float(ks_stat),
        "method": "ks",
        "reason": reason,
        "ks_stat": float(ks_stat),
        "ks_critical": float(critical),
        "mean_ratio": float(mean_score),
    }
