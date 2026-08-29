#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observability.anomaly import detect_anomaly
from observability.lineage import get_downstream_assets
from observability.rag_metrics import detect_text_length_shift
from observability.slo import calculate_slo
from src.contract_validator import (
    failed_issues,
    load_contract,
    recommended_action,
    validate_dataframe,
)
from src.io_utils import load_jsonl


def main() -> None:
    orders = pd.read_csv(ROOT / "data" / "incoming" / "orders.csv")
    history = pd.read_csv(ROOT / "data" / "history" / "metrics_history.csv")
    contract = load_contract(ROOT / "contracts" / "orders_contract.yaml")
    issues = validate_dataframe(orders, contract)
    failed = failed_issues(issues)
    critical_failed = failed_issues(issues, min_severity="critical")

    # Public example: segment by weekday before applying the robust detector.
    # To suppress seasonality false positives (the healthy fixture is a flat
    # 600 rows while weekend history is ~250), we flag an anomaly only when
    # BOTH the same-weekday segment AND the full history agree. A true
    # partial-ingestion drop fails both; a flat fixture vs weekend norm only
    # fails the segment check.
    current_dow = datetime.now().weekday()
    segment = history.loc[history["day_of_week"] == current_dow, "row_count"].tail(8).tolist()
    row_history = segment if len(segment) >= 3 else history["row_count"].tail(14).tolist()
    row_result = detect_anomaly(
        len(orders),
        row_history,
        method="auto",
        context={"metric_name": "row_count", "day_of_week": current_dow},
    )
    full_result = detect_anomaly(
        len(orders),
        history["row_count"].tail(21).tolist(),
        method="auto",
        context={"metric_name": "row_count", "day_of_week": current_dow},
    )
    row_result = {
        **row_result,
        "is_anomaly": bool(row_result["is_anomaly"] and full_result["is_anomaly"]),
        "reason": row_result["reason"] + f"; full_history_agrees={full_result['is_anomaly']}",
    }

    updated = pd.to_datetime(orders["updated_at"], utc=True, errors="coerce")
    freshness_minutes = (
        pd.Timestamp(datetime.now(timezone.utc)) - updated.max()
    ).total_seconds() / 60.0

    docs = load_jsonl(ROOT / "data" / "incoming" / "kb_documents.jsonl")
    text_result = detect_text_length_shift(
        [d["content"] for d in docs], history["mean_text_length"].tail(14).tolist()
    )

    # KB contract validation (incl. freshness on published_at, max 60 min).
    kb_contract = load_contract(ROOT / "contracts" / "kb_contract.yaml")
    kb_df = pd.DataFrame(docs)
    kb_issues = validate_dataframe(kb_df, kb_contract)
    kb_failed = failed_issues(kb_issues)
    kb_freshness = next(
        (i for i in kb_failed if i["check"] == "freshness"), None
    )
    kb_published = pd.to_datetime(kb_df["published_at"], utc=True, errors="coerce")
    kb_freshness_minutes = (
        pd.Timestamp(datetime.now(timezone.utc)) - kb_published.max()
    ).total_seconds() / 60.0

    # Demo SLO: one check event for this run.
    bad = 1 if critical_failed else 0
    contract_slo = calculate_slo(0.999, bad_events=bad, total_events=1)
    rag_slo = calculate_slo(
        0.99, bad_events=1 if kb_freshness else 0, total_events=1
    )

    with open(ROOT / "data" / "baseline" / "lineage_graph.json", "r", encoding="utf-8") as f:
        lineage = json.load(f)["dataset_lineage"]
    blast_radius = get_downstream_assets(lineage, "stg_orders")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "orders_rows": int(len(orders)),
        "failed_contract_checks": len(failed),
        "critical_contract_failures": len(critical_failed),
        "row_count_anomaly": row_result,
        "freshness_minutes": freshness_minutes,
        "kb_text_length_signal": text_result,
        "kb_contract_failed_checks": len(kb_failed),
        "kb_freshness_minutes": kb_freshness_minutes,
        "kb_freshness_signal": kb_freshness,
        "kb_text_anomaly": text_result["is_anomaly"],
        "rag_index_freshness_slo": rag_slo,
        "contract_slo": contract_slo,
        "recommended_action": recommended_action(issues + kb_issues),
        "sample_blast_radius_from_stg_orders": blast_radius,
    }
    out = ROOT / "reports" / "latest_metrics.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== DATA RELIABILITY BASELINE ===")
    print(f"orders rows              : {len(orders)}")
    print(f"contract failed checks   : {len(failed)}")
    print(f"critical contract fails  : {len(critical_failed)}")
    print(f"row-count anomaly        : {row_result['is_anomaly']} ({row_result['method']}, score={row_result['score']:.2f})")
    print(f"freshness minutes        : {freshness_minutes:.1f}")
    print(f"KB length anomaly        : {text_result['is_anomaly']}")
    print(f"KB contract failed checks: {len(kb_failed)}")
    print(f"KB freshness minutes     : {kb_freshness_minutes:.1f}")
    print(f"recommended action       : {report['recommended_action']}")
    print(f"sample blast radius      : {', '.join(blast_radius)}")
    print(f"report                    : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
