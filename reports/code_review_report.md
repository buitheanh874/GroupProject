# Code Review Report - Traffic Signal Control Project
## Run Log - 2026-01-07 17:45
### Executive Summary
- Aligned teleport documentation with defensive implementation and ensured eval output carries all teleport/corrected KPIs.

### README vs Code Consistency Check
- Invariant A: teleport_rate uses departed uniques in code and README.
- Invariant B: arrived_corr = arrived \\ teleported; teleported_arrived = arrived ∩ teleported in code and README.
- Invariant C: corrected metrics cap teleported or not-arrived vehicles; README matches.
- Invariant D: cap_sec = teleport_time_cap_sec else episode duration (min 1s); reflected in README.

### Changes Made
- File: env/kpi.py (docstring corrected to departed-based teleport_rate and cap policy).
- File: docs/README.md (Teleport Handling Policy updated to match code, added completion/failed/throughput_corr definitions).

### Tests
- pytest -q output: 107 passed in 5.36s

### Notes
- Teleport count-only fallback remains conservative when IDs unavailable; throughput_corr uses per-step queue samples consistent with evaluator output.

# Code Review Report - Traffic Signal Control Project
## Run Log - 2026-01-07 17:20
### Executive Summary
- Hardened teleport KPIs: denominator uses departed uniques, corrected metrics penalize teleports and non-arrivals with caps, and CSV now emits defensive fields for evaluation.

### Findings
- Teleport metrics undercounted and allowed apparent KPI gains by ignoring non-arrivals; corrected now.

### Fixes Applied
- env/kpi.py: Track departed/arrived/teleported uniques, add corrected completion/failure metrics, cap teleported/non-arrived times, fix teleport_rate denominator, add throughput_corr and teleported_arrived.
- env/sumo_env.py: Add telemetry tracking per step with penalty gating, propagate teleport time cap to KPI tracker, guard reward penalty when lambda is zero.
- scripts/eval.py: Emit new teleport/corrected columns in rows and CSV.
- tests/test_kpi_tracker.py: Add coverage for teleport rate, arrived_corr exclusion, and non-arrival penalties.
- tests/test_eval_kpi_logging.py: Expect new CSV columns.
- docs/README.md: Document teleport handling policy and corrected KPI definitions.

### Corrected KPI Definition
- departed_ids: accumulated getDepartedIDList() per episode.
- arrived_ids: accumulated getArrivedIDList() per episode.
- teleported_ids: accumulated getStartingTeleportIDList() (count-only fallback when IDs unavailable).
- arrived_corr formula: |arrived_ids \ teleported_ids|.
- failed_ids definition: teleported_ids ∪ (departed_ids \ arrived_ids) with extra count-only teleports capped.
- cap_sec: teleport_time_cap_sec if set, else episode duration (last sim time), minimum 1.0s.
- teleport_rate denominator: max(1, |departed_ids|).

### Tests
- Commands:
  - pytest -q
- Tests changed:
  - tests/test_kpi_tracker.py
  - tests/test_eval_kpi_logging.py

### Verification Output
- pytest -q: 107 passed in 4.72s
- CSV header (scripts/eval.py): controller, scenario, run_id, total_reward, episode_steps, arrived_vehicles, avg_wait_time, avg_travel_time, avg_stops, avg_queue, max_wait_time, p95_wait_time, throughput, teleport_started_total, teleport_unique, teleport_rate, arrived_corr, teleported_arrived, completion_rate, failed_corr, avg_wait_time_corr, avg_travel_time_corr, p95_wait_time_corr, max_wait_time_corr, throughput_corr
