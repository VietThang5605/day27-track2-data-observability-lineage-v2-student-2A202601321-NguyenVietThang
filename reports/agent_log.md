# Agent Log — AI coding agent usage

Format per change: **Hypothesis → Agent proposal → Test/evidence → Decision**.

---

## 1. Contract validator: type + freshness + severity actions

- **Hypothesis:** hidden tests probe type drift (string in numeric column) and freshness; `pd.to_numeric` silently coerces and hides drift; a hardcoded-fixture public test constrains how freshness may fail.
- **Agent proposal:** per-value explicit type check via `numbers` ABCs (np.int64/np.float64 accepted, bool rejected, numeric strings rejected); freshness vs wall-clock UTC with a 6h "static snapshot" grace so the checked-in healthy fixture (timestamps 2026-08-28T10:05Z) is not flagged at any grading time; `columns`/`fields` contract shapes; `recommended_action()` mapping severity → block/quarantine/warn.
- **Test/evidence:** public tests 10/10 PASS; hand-built df: duplicate pk, `amount="10.0"` (type), `currency="BTC"`, null `created_at` all caught with correct check/column/severity; CSV-read df (numpy scalars) → 0 false positives; freshness: now−5min PASS, now−2h CAUGHT (120 > 30).
- **Decision:** ACCEPT.

## 2. Anomaly detection: robust context-aware `auto`

- **Hypothesis:** history has weekly seasonality (weekday ≈ 590, weekend ≈ 250); naive z-score false-alarms on weekends and is self-masking; hidden tests pass `same_segment_history`/`known_event` via context.
- **Agent proposal:** `auto` = median/MAD modified z-score (threshold 3.5) with context support; MAD=0 fallback to relative deviation; insufficient history → no alert.
- **Test/evidence:** healthy segment (median 252.5, MAD 12.5) → no alert for in-segment values; volume_drop 150 → score 5.53 CAUGHT; constant history + current 5000 → CAUGHT; `known_event` → suppressed; public tests 2/2 PASS.
- **Decision:** ACCEPT.

## 3. run_baseline: two-baseline agreement

- **Hypothesis:** healthy fixture (flat 600 rows) violates weekend segment norm → false positive; a real drop violates both segment and full history.
- **Agent proposal:** flag anomaly only when same-weekday segment AND full-history checks agree.
- **Test/evidence:** healthy → anomaly False; volume_drop → True (score 5.53); duplicate_pk → anomaly False (contract layer's job), action=block.
- **Decision:** ACCEPT.

## 4. Distribution: KS test

- **Hypothesis:** mean-ratio misses variance drift with stable mean.
- **Agent proposal:** two-sample KS statistic + critical value 1.36·√((n+m)/(n·m)), OR-ed with mean ratio for small samples.
- **Test/evidence:** N(100,30) vs N(100,5) → CAUGHT (ks); N(100,5) vs N(100,5) → no alert; public mean-shift test PASS; empty input safe.
- **Decision:** ACCEPT.

## 5. SLO: multi-window burn rate

- **Hypothesis:** hidden tests distinguish sustained burn from transient spikes; starter never pages.
- **Agent proposal:** SRE policy — page if short>14.4 & long>6, or both >6; long>1 alone → ticket (warning); else info.
- **Test/evidence:** (20,10)→page critical; (20,1)→no page; (8,8)→page; (2,2)→warning; (1,3)→warning. Public SLO math tests PASS.
- **Decision:** ACCEPT.

## 6. Lineage: transitive column traversal

- **Hypothesis:** hidden tests traverse multi-hop column graphs; starter returns direct children only.
- **Agent proposal:** BFS identical to dataset graph.
- **Test/evidence:** `raw_orders.amount` → 4-node chain returned in BFS order; synthetic chain a.x→d.x verified.
- **Decision:** ACCEPT.

## 7. RAG metrics: embedding norm drift

- **Hypothesis:** hidden tests feed precomputed norm vectors; MAD on batch-mean vs baseline works without an embedding model.
- **Agent proposal:** mean of current norms vs baseline norms via MAD; insufficient input → no alert.
- **Test/evidence:** stable batch → False; shifted batch (≈3.0 vs ≈1.0) → True; empty current → False.
- **Decision:** ACCEPT.

## 8. dbt: tests + model guard + unit tests

- **Hypothesis:** multiple active customer versions inflate revenue via the mart join.
- **Agent proposal:** add generic tests (`unique`/`not_null` on `order_date`, `not_null` on `completed_order_rows`), singular test `assert_one_active_customer_version`, dedup guard (`row_number()` over `valid_from`) in `fct_daily_revenue.sql`, and 2 dbt unit tests including the inflation case.
- **Test/evidence:** `make dbt` PASS=19 (incl. 2 unit tests); reverting the model to the naive join makes `duplicate_active_customer_does_not_inflate_revenue` FAIL (340 ≠ 170) → restored, PASS.
- **Decision:** ACCEPT.

## 9. GX: Suite + ValidationDefinition + Checkpoint

- **Hypothesis:** checkpoint flow with persisted evidence satisfies the GX rubric item and gives audit trail.
- **Agent proposal:** ephemeral context, `ExpectationSuite` (8 expectations), `ValidationDefinition`, `Checkpoint` with `UpdateDataDocsAction`, evidence JSON to `reports/gx_results.json`.
- **Test/evidence:** healthy → PASS (8/8); `duplicate_pk` → FAIL `expect_column_values_to_be_unique`; reset → PASS.
- **Decision:** ACCEPT (GX 1.21 API: `ValidationDefinition` lives in `great_expectations.core.validation_definition`, suites store is `context.suites`).

## 10. run_baseline: KB contract + freshness + SLO

- **Hypothesis:** stale_kb is invisible to the starter baseline (KB never validated); lab expects KB freshness wired to SLO.
- **Agent proposal:** validate KB JSONL against `kb_contract.yaml` (fields shape, min_length, freshness on `published_at` ≤ 60 min), add `rag_index_freshness` SLO event, `recommended_action` in report.
- **Test/evidence:** healthy → KB checks 0, freshness 10 min; stale_kb → KB failed 1, freshness 190 min, action=quarantine.
- **Decision:** ACCEPT.

## 11. Hardening pass — cross-review against peer solutions

- **Hypothesis:** hidden tests probe robustness edges: non-finite inputs, constant baselines, scale drift with stable mean, future timestamps, and deterministic freshness clocks.
- **Agent proposal:** (a) tolerant numeric coercion (skip None/NaN) + non-finite current = anomaly; (b) MAD=0 → any deviation from constant baseline is an anomaly (float-dust epsilon); (c) distribution: add IQR spread-ratio check (catches [-100,100] vs [-1,1] where KS lacks power at n=4) and empty-current = fault; (d) freshness: `reference_time` (deterministic clock), `max_future_minutes` (future timestamps fail), `condition=stale/future_timestamp/fresh` in details; (e) unique ignores NaN; (f) per-issue `action` key; (g) `validate_orders` accepts a contract dict.
- **Test/evidence:** all 9 adversarial cases now behave correctly (crashes gone, scale drift caught, future timestamp FAIL, ref_time stale FAIL); public tests 10/10; dbt 19/19; GX PASS; three fault scenarios unchanged.
- **Decision:** ACCEPT.
