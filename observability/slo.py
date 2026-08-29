from __future__ import annotations

from typing import Any


def calculate_slo(target: float, bad_events: int, total_events: int) -> dict[str, Any]:
    if not 0 < target < 1:
        raise ValueError("target must be between 0 and 1 (exclusive)")
    if bad_events < 0 or total_events < 0 or bad_events > total_events:
        raise ValueError("invalid event counts")
    allowed_bad_rate = 1.0 - target
    if total_events == 0:
        return {
            "target": target,
            "actual_bad_rate": 0.0,
            "allowed_bad_rate": allowed_bad_rate,
            "burn_rate": 0.0,
            "remaining_error_budget_fraction": 1.0,
            "breached": False,
        }
    actual_bad_rate = bad_events / total_events
    burn_rate = actual_bad_rate / allowed_bad_rate
    consumed_fraction = min(1.0, actual_bad_rate / allowed_bad_rate)
    return {
        "target": target,
        "actual_bad_rate": actual_bad_rate,
        "allowed_bad_rate": allowed_bad_rate,
        "burn_rate": burn_rate,
        "remaining_error_budget_fraction": max(0.0, 1.0 - consumed_fraction),
        "breached": bool(actual_bad_rate > allowed_bad_rate),
    }


def evaluate_multiwindow_burn(
    *,
    short_window_burn: float,
    long_window_burn: float,
    policy: str = "sre-multiwindow",
) -> dict[str, Any]:
    """Multi-window, multi-burn-rate alert policy (Google SRE Workbook).

    `short_window_burn` is the fast/recent window; `long_window_burn` is the
    slow/sustained window. A page requires BOTH windows to burn: a transient
    spike (high short, low long) must not page, and a slow sustained leak
    (low short, high long) should open a ticket without paging.

    Thresholds follow the canonical SRE recommendation (14.4x/6x for fast
    burn, 6x/1x for slow burn), adapted to two windows.
    """
    if short_window_burn < 0 or long_window_burn < 0:
        raise ValueError("burn rates must be non-negative")

    if short_window_burn > 14.4 and long_window_burn > 6.0:
        page, severity = True, "critical"
        reason = (
            f"fast sustained burn: short={short_window_burn:.2f}x (>14.4) "
            f"and long={long_window_burn:.2f}x (>6)"
        )
    elif short_window_burn > 6.0 and long_window_burn > 6.0:
        page, severity = True, "critical"
        reason = (
            f"sustained burn: short={short_window_burn:.2f}x (>6) "
            f"and long={long_window_burn:.2f}x (>6)"
        )
    elif long_window_burn > 1.0:
        page, severity = False, "warning"
        reason = (
            f"slow sustained burn: long={long_window_burn:.2f}x (>1) with "
            f"short={short_window_burn:.2f}x — open ticket, do not page"
        )
    else:
        page, severity = False, "info"
        reason = (
            f"no significant burn: short={short_window_burn:.2f}x, "
            f"long={long_window_burn:.2f}x (transient spike or healthy)"
        )

    return {
        "page": page,
        "severity": severity,
        "reason": reason,
        "short_window_burn": short_window_burn,
        "long_window_burn": long_window_burn,
        "policy": "sre-multiwindow",
    }
