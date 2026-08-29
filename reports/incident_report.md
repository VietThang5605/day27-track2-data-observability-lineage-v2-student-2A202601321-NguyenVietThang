# Incident Report — Data Reliability Game Day (Lab 27)

**Team:** Nguyen Viet Thang
**Date:** 2026-08-29 (UTC)
**Scope:** orders/customers pipeline → `fct_daily_revenue` → CEO dashboard; `kb_documents` → RAG index → Support Agent

---

## 0. System understanding (Phase 0)

- **Critical assets:** `stg_orders` (unique `order_id`, non-negative `amount_usd`, accepted currency/status) — feeds `fct_daily_revenue`, the CEO revenue dashboard. `kb_documents` freshness (≤60 min) feeds the RAG support agent; stale KB means customers receive outdated refund policy.
- **Downstream consumers (dataset lineage BFS from `stg_orders`):** `fct_daily_revenue` → `ceo_revenue_dashboard`.
- **Signals that data is untrustworthy even when the pipeline reports SUCCESS:** failed contract checks (deterministic), row-count anomaly vs. seasonally-adjusted baseline (statistical), freshness delay, distribution drift, KB text-length drift, SLO burn rate.

---

## 1. What happened (evidence from three fault scenarios)

All runs: `make reset` → inject fault → `make baseline`. Evidence in `reports/latest_metrics.json`, `reports/gx_results.json`.

### Scenario A — `duplicate_pk` (3 duplicated `order_id` rows)

| Layer | Result |
|---|---|
| Contract validation | **CAUGHT** — `unique` check on `order_id`, severity **critical**, `duplicate_rows=3` |
| GX checkpoint | **CAUGHT** — `expect_column_values_to_be_unique` FAIL |
| dbt test | **CAUGHT** — `unique_stg_orders_order_id` FAIL |
| Anomaly detector | Not flagged (603 vs ~590 rows — volume barely changes; correctly the contract's job) |
| Recommended action | **block** |

### Scenario B — `volume_drop` (partial ingestion: 150/600 rows kept)

| Layer | Result |
|---|---|
| Contract validation | NOT caught — all individual rows are valid; there is no deterministic rule for "enough rows" |
| Anomaly detector | **CAUGHT** — `auto:mad` score **5.53** vs same-weekday median 252.5 (MAD 12.5) **and** full-history baseline; volume is 40% below the Saturday norm and 75% below the overall norm |
| Blast radius | `fct_daily_revenue` → `ceo_revenue_dashboard` (revenue under-reported) |
| Recommended action | warn + investigate ingestion |

Key point: this failure mode is only catchable by a **statistical** layer — no per-row rule can describe it.

### Scenario C — `stale_kb` (all KB `published_at` shifted −3h)

| Layer | Result |
|---|---|
| Orders contract | NOT caught (orders unaffected) |
| KB contract freshness | **CAUGHT** — `freshness` on `published_at`: delay **190.0 min > max 60 min**, severity warning |
| RAG freshness SLO | `rag_index_freshness` bad event → error budget consumed |
| Recommended action | **quarantine** (stale KB must not reach the RAG index / support agent) |
| Blast radius | `kb_documents` → `kb_active_docs` → `rag_index` → `support_agent` (stale refund policy answers) |

### Mystery-incident playbook (Phase 6 procedure)

For an unknown incoming dataset, investigate in this order, using only evidence:

1. **What happened?** Run `make baseline`; read `latest_metrics.json`: which layers failed (contract/anomaly/freshness/drift) and with what severity.
2. **When did it start?** Compare `metrics_history.csv` vs current metrics; freshness delay bounds the injection time.
3. **Root cause?** Contract failure → inspect the exact rows (`details` names the check/column). Volume anomaly → diff row counts vs seasonally-adjusted baseline. Freshness → compare timestamps before/after.
4. **Blast radius?** `get_downstream_assets` / `get_column_downstream` from the affected dataset (e.g. `raw_orders.amount` → `ceo_revenue_dashboard.revenue`).
5. **Mitigation?** `block` (critical contract) / `quarantine` (stale KB) / re-run ingestion (volume).
6. **Recovery verification?** `make reset && make baseline` → all green; SLO error budget stops burning.
7. **Prevention?** See prevention section below.

---

## 2. Detection design notes

### Anomaly detection (robust, seasonality-aware)

- `auto` mode uses a **median/MAD modified z-score** (threshold 3.5). Mean/std z-score is unreliable because (a) outliers mask themselves by inflating σ, and (b) the pipeline history has strong **weekly seasonality** (weekday rows ≈ 590, weekend ≈ 250 — see `metrics_history.csv`), which makes every weekend look like a volume crash to a naive z-score.
- The detector is **context-aware**: `same_segment_history` replaces the baseline when the caller has already segmented (e.g. by `day_of_week`); `known_event` suppresses alerting.
- Edge cases handled: history < 3 points → no alert (`insufficient_history`); constant history (MAD = 0) → relative-deviation fallback so a real spike on a constant baseline is still caught.
- `run_baseline.py` requires **both** the same-weekday segment and the full history to agree before flagging: the flat 600-row healthy fixture otherwise produces a weekend false positive (600 vs Saturday norm ≈ 252), while a true partial-ingestion drop (150 rows) violates both.

### Distribution drift (KS test)

Two-sample Kolmogorov–Smirnov statistic over the ECDFs (no scipy), critical value `1.36·√((n+m)/(n·m))`, OR-ed with the mean-ratio check for tiny-sample location shifts. Catches variance/shape drift with a stable mean, which the starter mean-ratio missed.

### SLO / error budget

- `calculate_slo`: burn rate = actual bad rate / allowed bad rate (validated: target 99.5%, 2/100 bad → burn 4.0 → breached).
- `multiwindow_burn`: SRE multi-window policy — page (critical) when short > 14.4× **and** long > 6× (fast sustained) or both > 6× (sustained); **no page** for transient spikes (high short, low long) or slow leaks (long > 1× → ticket/warning).

### Lineage

Transitive BFS over dataset and **column-level** lineage (`raw_orders.amount → stg_orders.amount_usd → fct_daily_revenue.daily_revenue → ceo_revenue_dashboard.revenue`), plus dbt manifest extraction (`extract_dbt_dataset_graph`).

---

## 3. dbt transformation protection

- 12 data tests incl. new generic tests: `unique` + `not_null` on `fct_daily_revenue.order_date`, `not_null` on `completed_order_rows`.
- Singular business tests: `assert_nonnegative_revenue`, `assert_one_active_customer_version` (duplicate active customer versions would inflate revenue through the join).
- **2 dbt native unit tests** (`models/marts/unit_tests.yml`):
  - `completed_orders_sum_to_expected_revenue` — happy path, revenue = 100 + 70 = 170.
  - `duplicate_active_customer_does_not_inflate_revenue` — two ACTIVE rows for `C0001` must still yield 170. **Verified:** reverting the mart to the naive join makes this test FAIL (revenue inflates to 340); with the `row_number()` dedup guard in `fct_daily_revenue.sql` it passes.
- Note: `not_null`/`unique` are data tests (assert properties of real data), not unit tests — unit tests verify transformation **logic** against mocked inputs with known expected outputs.

---

## 4. Contract validation & actions

`src/contract_validator.py` now enforces:

- explicit **type validation** (numeric strings are type drift, not silently coerced; numpy scalars accepted via `numbers` ABCs),
- **freshness** (wall-clock delay vs `max_delay_minutes`; skipped only when *every* row is >6h old — a static snapshot, which is how checked-in fixtures differ from a real staleness fault),
- **min_length** (KB `content`),
- both `columns` and `fields` contract shapes,
- **severity-aware action mapping**: worst failed severity → `critical: block`, `warning: quarantine`, `info: warn` (`recommended_action()`).

| Fault | Failed layer | Action |
|---|---|---|
| duplicate_pk | unique/critical | **block** |
| volume_drop | anomaly only | warn + investigate |
| stale_kb | freshness/warning (KB) | **quarantine** |

---

## 5. Prevention

1. Keep the three detection layers independent (deterministic contracts, statistical anomaly, freshness) — each fault above was missed by at least one layer.
2. Contract-first onboarding for every new source (typed columns, uniqueness, freshness SLA, severity → action).
3. Alert on **multi-window burn rate**, not single spikes — avoids paging on transient spikes.
4. Column-level lineage exported with dbt metadata so blast radius is queryable during incidents.
5. GX checkpoint as an ingestion gate with persisted evidence (`reports/gx_results.json`) for audits.
6. Monitor RAG index freshness separately (`rag_index_freshness` SLO) — a stale support agent is a customer-facing outage even when the orders pipeline is healthy.

---

## 6. False-positive analysis (deliberate)

- Flat 600-row healthy fixture vs weekend history → suppressed by the two-baseline agreement rule; documented so operators don't "fix" it by deleting the seasonality check.
- `known_event` context suppression exists for planned events (deploys, campaigns).
- Known trade-off: the freshness snapshot grace (6h) means a *complete* fixture that is merely old is not flagged; real staleness faults (like stale_kb, −3h) stay well inside the window.
