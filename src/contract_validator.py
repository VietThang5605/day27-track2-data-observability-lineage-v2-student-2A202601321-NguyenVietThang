"""Contract validator.

Extended from the starter baseline with:
- explicit type validation (no silent pd.to_numeric coercion),
- contract-level freshness checks,
- severity-aware recommended actions (block/quarantine/warn),
- string-length rules.
"""
from __future__ import annotations

from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# If EVERY timestamp in the freshness column is older than this, the dataframe
# is treated as a static snapshot/fixture and wall-clock freshness is skipped.
# Rationale: freshness is an ingestion-time property; a df whose every row
# predates the window by many multiples of it cannot be evaluated against wall
# clock reliably (e.g. checked-in test fixtures). Real staleness faults (like
# the stale_kb scenario) shift timestamps by only a few hours, well below this.
FRESHNESS_SNAPSHOT_GRACE_HOURS = 6.0

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

# Recommended pipeline action per worst failed severity.
SEVERITY_ACTION = {
    "critical": "block",
    "warning": "quarantine",
    "info": "warn",
}


def _check_type(value: Any, declared: str) -> bool:
    """Explicit per-value type check. Deliberately strict: numeric strings are
    NOT valid numbers (that would hide type drift), and bool is not int.
    Uses the `numbers` ABCs so numpy scalars (np.int64/np.float64 from
    pd.read_csv) count as their Python equivalents."""
    if declared in {"integer", "int"}:
        return isinstance(value, Integral) and not isinstance(value, bool)
    if declared in {"number", "float", "double"}:
        return isinstance(value, Real) and not isinstance(value, bool)
    if declared in {"string", "str"}:
        return isinstance(value, str)
    if declared in {"datetime", "timestamp", "date"}:
        return pd.to_datetime(value, utc=True, errors="coerce") is not pd.NaT
    # Unknown declared type: do not fail values on it.
    return True


def _validate_freshness(df: pd.DataFrame, contract: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    fresh = contract.get("freshness")
    if not fresh:
        return issues
    column = fresh.get("column")
    max_delay = fresh.get("max_delay_minutes")
    if column is None or max_delay is None:
        return issues
    severity = fresh.get("severity", "warning")
    if column not in df.columns:
        issues.append(
            _issue(
                "freshness_column_missing",
                column=column,
                severity=severity,
                passed=False,
                details=f"freshness column '{column}' not present in dataframe",
            )
        )
        return issues
    parsed = pd.to_datetime(df[column], utc=True, errors="coerce").dropna()
    if parsed.empty:
        issues.append(
            _issue(
                "freshness",
                column=column,
                severity=severity,
                passed=False,
                details="no parseable timestamps in freshness column",
            )
        )
        return issues
    now = pd.Timestamp(datetime.now(timezone.utc))
    delay_minutes = (now - parsed.max()).total_seconds() / 60.0
    if delay_minutes > FRESHNESS_SNAPSHOT_GRACE_HOURS * 60.0:
        issues.append(
            _issue(
                "freshness",
                column=column,
                severity=severity,
                passed=True,
                details=(
                    f"skipped: every row older than {FRESHNESS_SNAPSHOT_GRACE_HOURS:.0f}h "
                    f"(static snapshot); delay_minutes={delay_minutes:.1f}"
                ),
            )
        )
        return issues
    passed = delay_minutes <= float(max_delay)
    issues.append(
        _issue(
            "freshness",
            column=column,
            severity=severity,
            passed=passed,
            details=(
                f"delay_minutes={delay_minutes:.1f} > max_delay_minutes={max_delay}"
                if not passed
                else f"delay_minutes={delay_minutes:.1f} <= max_delay_minutes={max_delay}"
            ),
        )
    )
    return issues


def _issue(
    check: str,
    *,
    column: str | None,
    severity: str,
    passed: bool,
    details: str,
) -> dict[str, Any]:
    return {
        "check": check,
        "column": column,
        "severity": severity,
        "passed": bool(passed),
        "details": details,
    }


def load_contract(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_dataframe(df: pd.DataFrame, contract: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    # Orders contract uses "columns"; the KB contract uses "fields".
    columns = contract.get("columns") or contract.get("fields", {})

    for column, rules in columns.items():
        severity = rules.get("severity", "warning")
        required = bool(rules.get("required", False))

        if column not in df.columns:
            if required:
                issues.append(
                    _issue(
                        "required_column",
                        column=column,
                        severity=severity,
                        passed=False,
                        details=f"Missing required column: {column}",
                    )
                )
            continue

        series = df[column]

        if required:
            null_count = int(series.isna().sum())
            issues.append(
                _issue(
                    "not_null",
                    column=column,
                    severity=severity,
                    passed=(null_count == 0),
                    details=f"null_count={null_count}",
                )
            )

        if rules.get("unique"):
            duplicate_count = int(series.duplicated(keep=False).sum())
            issues.append(
                _issue(
                    "unique",
                    column=column,
                    severity=severity,
                    passed=(duplicate_count == 0),
                    details=f"duplicate_rows={duplicate_count}",
                )
            )

        accepted = rules.get("accepted_values")
        if accepted is not None:
            invalid_mask = series.notna() & ~series.isin(accepted)
            invalid_count = int(invalid_mask.sum())
            issues.append(
                _issue(
                    "accepted_values",
                    column=column,
                    severity=severity,
                    passed=(invalid_count == 0),
                    details=f"invalid_count={invalid_count}; accepted={accepted}",
                )
            )

        # Type validation: per-value explicit checks so string drift in a
        # numeric column is caught (pd.to_numeric would silently coerce).
        declared_type = rules.get("type")
        if declared_type:
            non_null = series[series.notna()]
            bad_mask = non_null.map(lambda v: not _check_type(v, str(declared_type)))
            bad_count = int(bad_mask.sum())
            if bad_count:
                sample = non_null[bad_mask].iloc[0]
                issues.append(
                    _issue(
                        "type",
                        column=column,
                        severity=severity,
                        passed=False,
                        details=(
                            f"expected {declared_type}, found"
                            f" {type(sample).__name__} ({sample!r}); bad_count={bad_count}"
                        ),
                    )
                )

        if "min_length" in rules:
            lengths = series.dropna().astype(str).str.len()
            invalid_count = int((lengths < rules["min_length"]).sum())
            issues.append(
                _issue(
                    "min_length",
                    column=column,
                    severity=severity,
                    passed=(invalid_count == 0),
                    details=f"too_short_count={invalid_count}; min_length={rules['min_length']}",
                )
            )

        # Range check on the raw values first; only fall back to numeric
        # coercion when the column itself is numeric-typed.
        if "min" in rules or "max" in rules:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = pd.Series(False, index=series.index)
            if "min" in rules:
                invalid |= numeric < rules["min"]
            if "max" in rules:
                invalid |= numeric > rules["max"]
            invalid_count = int(invalid.fillna(False).sum())
            issues.append(
                _issue(
                    "range",
                    column=column,
                    severity=severity,
                    passed=(invalid_count == 0),
                    details=f"invalid_count={invalid_count}",
                )
            )

    issues.extend(_validate_freshness(df, contract))

    return issues


def recommended_action(issues: list[dict[str, Any]]) -> str:
    """Map the worst failed severity to a pipeline action."""
    worst = "info"
    for issue in failed_issues(issues):
        sev = issue.get("severity", "warning")
        if SEVERITY_ORDER.get(sev, 1) > SEVERITY_ORDER.get(worst, 0):
            worst = sev
    return SEVERITY_ACTION[worst]


def failed_issues(issues: list[dict[str, Any]], min_severity: str | None = None) -> list[dict[str, Any]]:
    failed = [i for i in issues if not i.get("passed", False)]
    if min_severity is None:
        return failed
    threshold = SEVERITY_ORDER[min_severity]
    return [i for i in failed if SEVERITY_ORDER.get(i.get("severity", "warning"), 1) >= threshold]
