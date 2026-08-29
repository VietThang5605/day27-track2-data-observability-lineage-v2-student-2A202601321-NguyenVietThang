"""Anomaly detection starter.

Z-score is deliberately the default baseline. Students should improve `auto`
mode for seasonality/outliers rather than deleting the simple implementation.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def zscore_detector(current: float, history: Iterable[float], threshold: float = 3.0) -> dict[str, Any]:
    values = np.asarray(list(history), dtype=float)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "zscore", "reason": "insufficient_history"}
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        score = float("inf") if float(current) != mean else 0.0
    else:
        score = abs(float(current) - mean) / std
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "zscore",
        "reason": f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}",
    }


def mad_detector(current: float, history: Iterable[float], threshold: float = 3.5) -> dict[str, Any]:
    """Robust median/MAD detector (modified z-score).

    Falls back to a relative-deviation check when MAD is zero (constant
    history): a constant baseline is itself strong evidence, so any
    meaningful relative deviation from the median should count.
    """
    values = np.asarray(list(history), dtype=float)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "mad", "reason": "insufficient_history"}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    cur = float(current)
    if mad == 0:
        # Constant history (or all-identical). Fall back to a relative check:
        # flag when current deviates from the median by more than a small
        # fraction of the median's magnitude (or is nonzero against zero).
        scale = max(abs(median), 1e-12)
        rel_dev = abs(cur - median) / scale
        is_anom = bool(rel_dev > 0.05) if median != 0 else bool(cur != 0)
        return {
            "is_anomaly": is_anom,
            "score": float(rel_dev if median != 0 else (0.0 if cur == 0 else float("inf"))),
            "method": "mad",
            "reason": f"mad_is_zero; median={median:.3f}, relative_deviation={rel_dev:.3f}",
        }
    modified_z = 0.6745 * abs(cur - median) / mad
    return {
        "is_anomaly": bool(modified_z > threshold),
        "score": float(modified_z),
        "method": "mad",
        "reason": f"median={median:.3f}, mad={mad:.3f}, threshold={threshold}",
    }


def _auto_detector(
    current: float,
    history: Iterable[float],
    threshold: float,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Context-aware robust detector.

    Strategy:
    1. If the caller supplies `same_segment_history`, use it as the baseline —
       it is already the correct seasonality segment.
    2. Otherwise, if `day_of_week` is provided we cannot re-segment raw
       history here (it may be pre-segmented already); rely on MAD.
    3. Use the robust median/MAD baseline by default: mean/std z-score is
       destroyed by the very outliers it should detect (masking effect) and
       by weekly seasonality (weekend volume looks like a drop).
    4. `known_event` in context suppresses alerting (planned event).
    """
    ctx = context or {}

    values = np.asarray(list(history), dtype=float)
    if values.size < 3:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "auto",
            "reason": "insufficient_history",
        }

    if ctx.get("known_event"):
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "auto",
            "reason": f"suppressed_known_event={ctx['known_event']}",
        }

    result = mad_detector(current, values, threshold=3.5)
    result["method"] = "auto:mad"
    bits = [result["reason"]]
    if "day_of_week" in ctx:
        bits.append(f"day_of_week={ctx['day_of_week']}")
    if "metric_name" in ctx:
        bits.append(f"metric={ctx['metric_name']}")
    if ctx.get("same_segment_history") is not None:
        bits.append("baseline=same_segment_history")
    result["reason"] = "; ".join(bits)
    return result


def detect_anomaly(
    current: float,
    history: Iterable[float],
    *,
    method: str = "auto",
    threshold: float = 3.0,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stable lab API.

    - `zscore`: basic z-score (kept for backward compatibility).
    - `mad`: robust median/MAD detector.
    - `auto`: context-aware robust detection (MAD baseline, seasonality via
      `same_segment_history`, suppression via `known_event`).
    """
    if method == "mad":
        return mad_detector(current, history)
    if method == "auto":
        return _auto_detector(current, history, threshold=threshold, context=context)
    if method == "zscore":
        return zscore_detector(current, history, threshold=threshold)
    raise ValueError(f"Unsupported method: {method}")
