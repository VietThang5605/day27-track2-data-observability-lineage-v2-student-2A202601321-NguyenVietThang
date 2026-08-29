"""Robust, context-aware anomaly detection (stable lab API).

- `zscore`: classic z-score (kept for backward compatibility).
- `mad`: modified z-score on median/MAD (Iglewicz–Hoaglin), needs >= 5 points.
- `auto`: context-aware: `same_segment_history` wins when available, robust
  MAD for >= 5 points, z-score fallback for 3–4 point histories,
  `known_event` suppression, non-finite current values are anomalies.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _coerce_values(history: Iterable[float]) -> np.ndarray:
    """Tolerant conversion: skip non-numeric / non-finite entries."""
    parsed: list[float] = []
    for value in history:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    values = np.asarray(parsed, dtype=float)
    return values[np.isfinite(values)]


def _coerce_current(current: float) -> float | None:
    try:
        value = float(current)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _not_finite_result(method: str) -> dict[str, Any]:
    # A non-finite current value means the metric itself broke: report it.
    return {"is_anomaly": True, "score": float("inf"), "method": method, "reason": "current_not_finite"}


def zscore_detector(current: float, history: Iterable[float], threshold: float = 3.0) -> dict[str, Any]:
    current_value = _coerce_current(current)
    if current_value is None:
        return _not_finite_result("zscore")
    values = _coerce_values(history)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "zscore", "reason": "insufficient_history"}
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        score = float("inf") if current_value != mean else 0.0
    else:
        score = abs(current_value - mean) / std
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "zscore",
        "reason": f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}",
    }


def mad_detector(current: float, history: Iterable[float], threshold: float = 3.5) -> dict[str, Any]:
    """Modified z-score on median/MAD. A perfectly constant baseline is strong
    evidence: any (non-float-noise) deviation from the median is an anomaly."""
    current_value = _coerce_current(current)
    if current_value is None:
        return _not_finite_result("mad")
    values = _coerce_values(history)
    if values.size < 5:
        return {"is_anomaly": False, "score": 0.0, "method": "mad", "reason": "insufficient_history"}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        eps = 1e-9 * max(1.0, abs(median))
        if abs(current_value - median) <= eps:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "method": "mad",
                "reason": f"mad_is_zero; current matches constant baseline median={median:.3f}",
            }
        return {
            "is_anomaly": True,
            "score": float("inf"),
            "method": "mad",
            "reason": f"mad_is_zero; baseline is constant at median={median:.3f}, current={current_value:.3f}",
        }
    modified_z = 0.6745 * abs(current_value - median) / mad
    return {
        "is_anomaly": bool(modified_z > threshold),
        "score": float(modified_z),
        "method": "mad",
        "reason": f"median={median:.3f}, mad={mad:.3f}, threshold={threshold}",
    }


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
    - `mad`: robust median/MAD detector (>= 5 points).
    - `auto`: context-aware robust detection — `same_segment_history` wins
      over mixed history; MAD for >= 5 points, z-score for 3–4 points;
      `known_event` suppresses alerting.
    """
    if _coerce_current(current) is None:
        return _not_finite_result(method)

    if method == "mad":
        return mad_detector(current, history)
    if method == "zscore":
        return zscore_detector(current, history, threshold=threshold)
    if method != "auto":
        raise ValueError(f"Unsupported method: {method}")

    ctx = context or {}
    if ctx.get("known_event"):
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "auto",
            "reason": f"suppressed_known_event={ctx['known_event']}",
        }

    segment = _coerce_values(ctx.get("same_segment_history", []))
    if segment.size >= 5:
        result = mad_detector(current, segment, threshold=3.5)
        result["method"] = "auto:mad_segment"
        result["reason"] += f"; segment_size={segment.size}"
        return result
    if segment.size >= 3:
        result = zscore_detector(current, segment, threshold=threshold)
        result["method"] = "auto:zscore_segment"
        result["reason"] += f"; segment_size={segment.size}"
        return result

    values = _coerce_values(history)
    if values.size >= 5:
        result = mad_detector(current, values, threshold=3.5)
        result["method"] = "auto:mad"
        return result
    result = zscore_detector(current, values, threshold=threshold)
    result["method"] = "auto:zscore"
    return result
