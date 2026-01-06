# CODEX Run Report

## Run 1 — 2026-01-06 17:47
### Audit reference
- Source: C:\Users\Dell\GroupProject\docs\Code_Audit_Cross_Check_Report_2026-01-06.md
- Sections used: C1, C2, C4, H4

### Goal
- Audit-fix pass for C1 queue counting, C2 transition waiting default, C4 validation deduplication, H4 downstream occupancy fail-fast.

### Changes made
- File: env/sumo_env.py
  - Change: Added optional ID check flag to downstream link validator to share logic across call sites.
  - Reason: H4 fail-fast validation; C4 single source of truth.
- File: scripts/common.py
  - Change: Reused downstream validation helper and removed duplicate state_dim checks in build_env.
  - Reason: C4 validation deduplication; H4 fail-fast direction checks.
- File: tests/test_downstream_links_validation.py
  - Change: Added whitespace/None downstream link case to ensure missing direction errors.
  - Reason: H4 downstream links must reject empty values.
- File: docs/upgrade_9_tls_plan.md
  - Change: Updated downstream occupancy mitigation and acceptance to describe fail-fast (no padding fallback).
  - Reason: H4 documentation alignment.

### Tests
- Command(s):
  - pytest -q
- Result:
  - pass (87 tests)

### Notes / Risks
- Downstream occupancy now fails during config build when enabled with missing or blank N/E/S/W links; ensure configs provide valid IDs before enabling.

### Next recommended actions
- Run a SUMO-backed smoke test for a 12D occupancy config to confirm network ID validation in practice.
- Consider refactoring duplicate logic between _step_legacy and _step_multi (audit H1).
- Review configs to ensure any occupancy-enabled scenarios supply downstream_links.

### Commit
- Commit: b59d76f6d834ba90052798c585fcb1cda1880df9

## Run 2 — 2026-01-06 18:09
### Audit reference
- Source: C:\Users\Dell\GroupProject\docs\Code_Audit_Cross_Check_Report_2026-01-06.md
- Sections used: C1, C2, C4, H4

### Goal
- Align downstream occupancy validation with audit (fail-fast, no drift), ensuring build-time and runtime share the same helper.

### Changes made
- File: env/sumo_env.py
  - Change: Tightened downstream link sanitizer (upper keys, stripped values), missing check uses None semantics, invalid mapping skips None, and runtime validation delegates to the helper.
  - Reason: H4 fail-fast validation; avoid duplicate/ drifting logic.
- File: scripts/common.py
  - Change: Continue to reuse validate_downstream_links_config at build time with validate_ids=False (structure-only) instead of bespoke checks.
  - Reason: C4 single source of truth; H4 structural enforcement.
- File: tests/test_downstream_links_validation.py
  - Change: Added coverage for validate_ids=False structure-only path and ensured whitespace/None values raise missing directions.
  - Reason: H4 test alignment with new validation rules.

### Tests
- Command(s):
  - pytest -q
- Result:
  - pass (88 tests)

### Notes / Risks
- Runtime still requires valid SUMO lane/edge IDs when occupancy enabled; configs must provide real IDs before running SUMO.

### Next recommended actions
- Consider normalizing downstream_links at config load time for YAML templates to prevent user whitespace errors.
- Run a SUMO-backed smoke test with occupancy enabled to verify network ID validation in practice.

### Commit
- Commit: b59d76f6d834ba90052798c585fcb1cda1880df9

## Run 3 — 2026-01-06 19:28
### Audit reference
- Source: C:\Users\Dell\GroupProject\docs\audit.md
- Section used: Opponent/Baseline strengthening (Hanoi fixed-time)

### Goal
- Strengthen fixed-time opponent by matching split+cycle (Hanoi-style) and wire into simulation entrypoint with tests and docs.

### Changes made
- File: controllers/fixed_time.py
  - Change: Rebuilt FixedTimeController to parse tuple/dict/object actions, match target split+cycle with penalties, and expose selected split/cycle.
  - Reason: Baseline opponent must pick closest action by split/cycle instead of static id.
- File: scripts/eval.py
  - Change: Wired fixed-time baseline to auto-select action based on scenario (70/30 unbalanced, 50/50 otherwise, cycle 90s) using env action space; legacy fixed_action_id remains respected.
  - Reason: Apply strengthened opponent in live simulation while keeping backward compatibility.
- File: tests/test_fixed_time_controller.py
  - Change: Added unit tests for tuple, dict/object parsing, cycle-aware selection, and empty-space rejection.
  - Reason: Guard the new selection logic across input shapes.
- File: docs/audit.md
  - Change: Documented strengthened Hanoi fixed-time baseline and wiring point.
  - Reason: Keep audit/backlog aware of new baseline behavior.

### Tests
- Command(s):
  - python -m compileall .
  - pytest -q
- Result:
  - pass (92 tests)

### Notes / Risks
- Fixed-time selection requires a non-empty action space; legacy fixed_action_id still overrides if provided in config.baseline.
- Cycle penalty only applies when actions expose cycle_sec; otherwise split-only selection is used.

### Commit
- Commit: 315325fa41709c88154a9eb97ff40c14b16f7218

## Run 4 — 2026-01-06 19:44
### Audit reference
- Source: C:\Users\Dell\GroupProject\docs\audit.md
- Section used: Opponent/Baseline strengthening (Hanoi fixed-time)

### Goal
- Eval wiring cleanup + traceability; ensure fixed-time auto-match not overridden and add smoke coverage.

### Changes made
- File: scripts/eval.py
  - Change: Cleaned duplicate wiring, ensured RL model load resolves args/config once, fixed-time auto-match only when controller includes fixed, and error when action space missing.
  - Reason: Prevent merge-artifact regressions and enforce correct baseline selection.
- File: tests/test_eval_wiring_smoke.py
  - Change: Added smoke tests for config resolution and action space extraction without running main.
  - Reason: Guard wiring/traceability behaviors.

### Tests
- Command(s):
  - python -m compileall .
  - pytest -q
- Result:
  - pass (95 tests)

### Notes / Risks
- Action space resolution now raises when empty; configs without actions will fail fast.
- Working tree: clean after commit.

### Commit
- Commit: HEAD
