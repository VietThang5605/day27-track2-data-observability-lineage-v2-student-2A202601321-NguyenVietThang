from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _numeric_values(values: Iterable[float]) -> np.ndarray:
    """Tolerant conversion: skip non-numeric / non-finite entries."""
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    result = np.asarray(parsed, dtype=float)
    return result[np.isfinite(result)]


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
    """Distribution-shift detector: KS test + IQR spread ratio + mean ratio.

    Combines three complementary signals so a single axis cannot hide drift:
    - KS statistic over the ECDFs catches shape/quantile drift (critical value
      1.36*sqrt((n+m)/(n*m)) ≈ significance 0.05);
    - IQR ratio |log(cur_iqr/base_iqr)| catches pure scale drift with a stable
      mean even on tiny samples where KS lacks power;
    - mean ratio keeps the starter's location-shift behavior.
    An empty current batch with a usable baseline is itself a fault (no data
    received), so it is reported as an anomaly rather than as healthy.
    """
    cur = _numeric_values(current_values)
    base = _numeric_values(baseline_values)
    if base.size == 0:
        return {"is_anomaly": False, "score": 0.0, "method": "ks", "reason": "insufficient_baseline"}
    if cur.size == 0:
        return {
            "is_anomaly": True,
            "score": float("inf"),
            "method": "ks",
            "reason": "current_batch_empty; no data received against usable baseline",
        }

    cur_mean = float(np.mean(cur))
    base_mean = float(np.mean(base))

    ks_stat = _ks_two_sample(cur, base)
    critical = 1.36 * float(np.sqrt((cur.size + base.size) / (cur.size * base.size)))
    ks_anomaly = ks_stat > critical

    cur_iqr = float(np.subtract(*np.quantile(cur, [0.75, 0.25])))
    base_iqr = float(np.subtract(*np.quantile(base, [0.75, 0.25])))
    spread_ratio: float | None = None
    scale_anomaly = False
    if base_iqr > 1e-9:
        spread_ratio = abs(float(np.log((cur_iqr + 1e-9) / (base_iqr + 1e-9))))
        scale_anomaly = spread_ratio >= float(np.log(3.0))  # 3x spread change

    if base_mean == 0:
        mean_score = float("inf") if cur_mean != 0 else 1.0
    else:
        mean_score = (
            max(abs(cur_mean / base_mean), abs(base_mean / cur_mean))
            if cur_mean != 0
            else float("inf")
        )
    mean_anomaly = mean_score >= ratio_threshold

    is_anomaly = bool(ks_anomaly or scale_anomaly or mean_anomaly)
    reason = (
        f"ks_stat={ks_stat:.3f} vs critical={critical:.3f}; "
        f"iqr {base_iqr:.3f}->{cur_iqr:.3f}"
        + (f" (spread_ratio={spread_ratio:.2f})" if spread_ratio is not None else "")
        + f"; baseline_mean={base_mean:.3f}, current_mean={cur_mean:.3f}"
        + (f"; triggers={[n for n, f in (('ks', ks_anomaly), ('scale', scale_anomaly), ('mean', mean_anomaly)) if f]}" if is_anomaly else "")
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
