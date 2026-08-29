#!/usr/bin/env python3
"""Great Expectations Core 1.21 — Suite + ValidationDefinition + Checkpoint flow.

Upgraded from the single-expectation starter into a reusable validation flow:
- ExpectationSuite packaged from the contract's critical rules,
- ValidationDefinition binding suite <-> batch definition,
- Checkpoint with an action that persists a JSON evidence report to
  reports/gx_results.json (used by the incident report).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import great_expectations as gx
    from great_expectations.checkpoint import Checkpoint, UpdateDataDocsAction
    from great_expectations.core.expectation_suite import ExpectationSuite
    from great_expectations.core.validation_definition import ValidationDefinition
    from great_expectations.expectations import (
        ExpectColumnValuesToBeBetween,
        ExpectColumnValuesToBeInSet,
        ExpectColumnValuesToNotBeNull,
        ExpectColumnValuesToBeUnique,
    )
except ImportError as exc:  # friendlier classroom failure
    raise SystemExit("great_expectations is not installed. Run: pip install -r requirements.txt") from exc

GX_REPORT = ROOT / "reports" / "gx_results.json"


def build_orders_suite() -> ExpectationSuite:
    """Critical rules from contracts/orders_contract.yaml, packaged as a Suite."""
    expectations = [
        ExpectColumnValuesToNotBeNull(column="order_id", severity="critical"),
        ExpectColumnValuesToBeUnique(column="order_id", severity="critical"),
        ExpectColumnValuesToNotBeNull(column="customer_id", severity="critical"),
        ExpectColumnValuesToBeBetween(column="amount", min_value=0, severity="critical"),
        ExpectColumnValuesToBeInSet(column="currency", value_set=["USD", "VND"], severity="critical"),
        ExpectColumnValuesToBeInSet(
            column="status",
            value_set=["pending", "completed", "refunded", "cancelled"],
            severity="warning",
        ),
        ExpectColumnValuesToNotBeNull(column="created_at", severity="critical"),
        ExpectColumnValuesToNotBeNull(column="updated_at", severity="critical"),
    ]
    suite = ExpectationSuite(name="orders_critical_suite")
    for exp in expectations:
        suite.add_expectation(exp)
    return suite


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "incoming" / "orders.csv")
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("orders_pandas")
    asset = data_source.add_dataframe_asset(name="orders_dataframe")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_orders")

    suite = build_orders_suite()
    suite = context.suites.add(suite)

    validation_def = ValidationDefinition(
        name="orders_critical_validation",
        data=batch_definition,
        suite=suite,
    )
    validation_def = context.validation_definitions.add(validation_def)

    checkpoint = Checkpoint(
        name="orders_critical_checkpoint",
        validation_definitions=[validation_def],
        actions=[UpdateDataDocsAction(name="update_data_docs")],
        result_format={"result_format": "SUMMARY"},
    )
    context.checkpoints.add(checkpoint)

    result = checkpoint.run(batch_parameters={"dataframe": df})

    # Persist evidence for the incident report.
    validation_result = result.run_results[next(iter(result.run_results))]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "orders_critical_checkpoint",
        "success": bool(result.success),
        "statistics": dict(validation_result.statistics) if hasattr(validation_result, "statistics") else {},
        "failed_expectations": [
            {
                "expectation": r.expectation_config.type,
                "column": r.expectation_config.kwargs.get("column"),
                "severity": r.expectation_config.kwargs.get("severity", "warning"),
                "result": str(r.result)[:300] if r.result else None,
            }
            for r in validation_result.results
            if not r.success
        ],
    }
    GX_REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"GX checkpoint {'PASS' if result.success else 'FAIL'} (evidence: {GX_REPORT.relative_to(ROOT)})")
    for failed in payload["failed_expectations"]:
        print(
            f"  FAIL {failed['expectation']} column={failed['column']} "
            f"severity={failed['severity']}"
        )
    if not payload["failed_expectations"]:
        print("  all expectations met")


if __name__ == "__main__":
    main()
