# Audit Backlog (Base for Codex) — MDP Traffic Signal Control Project (GroupProject)

**Date:** 2026-01-06  
**Purpose:** One consolidated backlog of *what must be fixed next* (and what can be deferred), based on:
- User-provided Claude “Code Review Report - MDP Traffic Signal Control Project”
- My GitHub re-check findings (see `Github_Recheck_Findings_GroupProject_2026-01-06.md`)
- Prior audit findings files in this workspace

> Note: There are indications that some fixes exist on a local/feature branch (Codex work) but are **not** present on GitHub default branch. Therefore, this backlog explicitly separates **(A) merge/apply Codex fixes** vs **(B) new work**.

---

## 0) Baseline / Branch Hygiene (Do this first)

### 0.1 Ensure Codex fixes are merged or reapplied onto the working branch
If your working branch is GitHub default (main/master), verify whether it already includes the following *audit-compliance* fixes. If not, **port/merge them first** because they affect the correctness of training signals and config validation.

**Checklist (expected post-fix behavior):**
- `queue_count_mode="snapshot_last_step"` is **rejected** (raise `ValueError`) in both:
  - `env/mdp_metrics.py` (config dataclass / post-init validation)
  - `scripts/validation.py` (CLI/config validation)
- `include_transition_in_waiting` default is **False** in `SumoEnvConfig` and training/eval YAML configs set it explicitly to `false`.
- `downstream_links` validation: strips whitespace; rejects empty/None link IDs; rejects missing direction keys; rejects wrong direction set.

**Acceptance criteria:**
- Unit tests cover and enforce rejection of invalid configs.
- Running `scripts/validation.py` on provided configs passes.
- Running an env init with invalid config fails fast with a clear error message.

---

## 1) Critical Fixes (Blockers)

### 1.1 Reward normalization: invalid denominator can silently corrupt reward
**Source:** Claude issue #2  
**Current risk:** `compute_normalized_reward()` can fall back to `denom = 1.0` when both `t_step` and `decision_cycle_sec` are invalid, without any warning. This can silently distort training reward.

**Where:**
- `env/mdp_metrics.py` — `compute_normalized_reward(...)`

**Fix:**
- If `denom <= 0`, emit a `RuntimeWarning` (or raise `ValueError` if you want strict fail-fast).
- Add upstream validation: `decision_cycle_sec > 0` and `t_step > 0` (or `step_length_sec > 0`) at config validation time.

**Acceptance criteria / tests:**
- New test: when `t_step=0` and `decision_cycle_sec=0`, function must warn (or raise).
- No behavior change for valid configs.

---

### 1.2 Multi-TLS phase/cycle step conversion: rounding determinism + diagnostics
**Source:** Claude issue #3  
**Current state:** Multi-mode requires all TLS share the same total step count per decision cycle; code already checks this and raises if mismatch, but the root cause can be hard to diagnose when step conversion is float-based.

**Where:**
- `env/sumo_env.py` — `_step_multi()` and the helper(s) that convert seconds → steps.

**Fix options:**
1) Improve diagnostic error message: include `step_length_sec`, per-TLS interval specs, and computed steps.
2) Make seconds→steps conversion deterministic and consistent across TLS:
   - Avoid float rounding drift. Prefer `ceil(duration_sec / step_length_sec)` with a consistent rule, or use integer math when `step_length_sec` is rationalizable (e.g., Decimal).
3) Add a unit test for a non-integer `step_length_sec` (e.g., 1.1s) verifying deterministic and equal step counts across TLS.

**Acceptance criteria / tests:**
- Multi TLS with identical seconds yields identical total steps, always.
- If mismatch occurs, exception message includes enough data to pinpoint which TLS/action caused it.

---

### 1.3 Direction mapping robustness: lane-to-direction inference can be fragile
**Source:** Claude issue #1 (adjusted)  
Claude described “state ordering inconsistency”; in the repo, ordering in multi-mode is internally consistent, and there is a unit test checking state vector ordering. The *real risk* is the lane→direction mapping used to build per-direction metrics.

**Where:**
- `env/sumo_env.py` — lane grouping / direction inference logic (e.g., `_direction_lanes_by_tls`, `_infer_direction`)

**Why it matters:**
- Misclassification of lanes into N/E/S/W corrupts `q_dir`, `w_dir`, downstream occupancy and fairness.

**Fix:**
- Prefer explicit configuration:
  - Require `SumoLaneGroups.approach_lanes` / equivalent explicit per-direction lane lists for multi-mode.
- If inference is allowed:
  - Implement a more robust inference rule (based on edge geometry or SUMO net direction), not string substring heuristics.
  - Add validation: each direction has ≥1 lane (or document exceptions), and total lanes match expected controlled sets.

**Acceptance criteria / tests:**
- New test with a lane naming scheme that would break substring matching should fail validation (or infer correctly).
- Optional: property-based check that aggregated totals are consistent (e.g., sum over directions equals sum over controlled lanes).

---

## 2) Major Improvements (High ROI, not necessarily blockers)

### 2.1 Action space cycles are hardcoded (make configurable)
**Source:** Claude issue #6  
**Where:**
- `env/sumo_env.py` — `_build_action_definitions()`

**Current behavior:** cycles default to `[60, 90, 120]` in multi-mode and must stay aligned with config-driven options.

**Fix:**
- Use `config.action_table` (already exists) as the single source of truth.
- If cycles must remain defaulted, allow override via a config field (e.g., `allowed_cycles_sec`) and update tests accordingly.

**Acceptance criteria / tests:**
- If `action_table` is provided, no hardcoded cycles are used.
- Tests updated to reflect configurability (or keep a separate test for default cycles).

---

### 2.2 State normalization performance (reduce allocations)
**Source:** Claude issue #5  
**Where:**
- `env/normalization.py` — `StateNormalizer.normalize(...)`

**Fix:**
- Use in-place ops where safe (`out=` in `np.clip`, avoid redundant casts/copies).

**Acceptance criteria / tests:**
- Output matches previous implementation (numerically).
- Micro-benchmark (optional) demonstrates fewer allocations / speedup.

---

### 2.3 Reduce duplication: `_step_legacy()` vs `_step_multi()`
**Source:** (additional) cross-check report noted heavy duplication  
**Where:**  
- `env/sumo_env.py`

**Fix:**
- Extract shared logic into helper(s), e.g.:
  - `_execute_cycle_for_tls(tls_id, intervals, ...)`
  - `_accumulate_metrics(...)`
  - `_finalize_reward_and_state(...)`

**Acceptance criteria:**
- One bug fix should not require editing two separate step implementations.
- No regression in existing tests.

---

## 3) Minor / Polish (Deferable)

### 3.1 Route pool loader: simplify path resolution
**Source:** Claude issue #7  
**Where:**  
- `scripts/route_pool_loader.py`

**Fix:**  
- Consolidate `exists()` checks and make path resolution clearer.

---

### 3.2 Missing type hints and consistency improvements
**Source:** Claude issue #8, #9  
**Where:**
- `scripts/common.py` and other helper scripts

**Fix:**
- Add return type hints to public helpers.
- Standardize error messages to include context (TLS id, index, config key).

---

## 4) Items from Claude report that are likely false positives (No action unless evidence appears)

### 4.1 “Memory leak in set_route_file_pool”
Overwriting `self._route_pool` should not leak in Python (old list becomes GC-eligible).  
Only revisit if:
- You store references to the old list elsewhere, or
- There is a long-lived structure that keeps appending without reset.

---

## 5) Suggested execution plan (Codex-friendly)

1) Merge/apply audit-compliance fixes (Section 0) onto the working branch.
2) Implement reward denom warning/validation (1.1) + unit test.
3) Improve step conversion determinism and diagnostics (1.2) + unit test with non-integer step length.
4) Harden lane→direction mapping (1.3) by requiring explicit per-direction lanes in multi-mode (best) + validation tests.
5) Then proceed with major refactors and performance items (2.x), followed by polish (3.x).

---

## Opponent/Baseline strengthening (Hanoi fixed-time)
- Fixed-time baseline now matches actions by split + cycle (70/30 split for “unbalanced”, 50/50 otherwise; cycle 90s) with heavy penalty for mismatched cycles.
- Wired in `scripts/eval.py`; legacy `baseline.fixed_action_id` remains honored for backward compatibility.
- To override, set `baseline.fixed_action_id` in config to force a specific action id.

---
