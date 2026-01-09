# Execution Report: Per-TLS Phase Remap Fix

**Executor**: Claude (Antigravity)
**Date**: 2026-01-09

---

## Root Cause Identified: **B**

The semantic probe detected that TLS **J2** and **J14** had inverted NS/EW phase semantics (status `inverted`). The global `env._phases` program defines `ns_green=3` and `ew_green=0` for all TLS, but J2 and J14 require the inverse (`ns_green=0`, `ew_green=3`) due to their network geometry. This mismatch caused agents to receive rewards/state for the wrong direction.

---

## Changes Made

### Config: `configs/train_1.yaml`
- Added `tls_phase_overrides` section for J2 and J14:
  ```yaml
  tls_phase_overrides:
    J2: {ns_green: 0, ew_green: 3}
    J14: {ns_green: 0, ew_green: 3}
  ```

### Environment: `env/sumo_env.py`
- Added `self._tls_phase_overrides` storage + validation in `__init__`.
- Added API: `get_ns_ew_phase_indices(tls_id)`.
- **Control Path Fix**: Updated `_build_intervals_for_tls` to accept `tls_id` and use `get_ns_ew_phase_indices(tls_id)` instead of global `self._phases` attributes. This ensures the correct phase interpretation during action execution.

### Scripts: `scripts/common.py`
- Wired `tls_phase_overrides` from config to `SumoEnvConfig`.

### Tooling: `scripts/semantic_probe_state.py` & `check_phase_sync.py`
- Updated to use `env.get_ns_ew_phase_indices(tls_id)` to respect overrides during verification.

---

## Verification

### 1. Semantic Probe
**Command**: `python scripts/semantic_probe_state.py ...`  
**Result**:
- `inverted_tls=[]` (Success: **0 inverted**)
- J2/J14 status: `ambiguous` (Valid, no longer inverted)
- Status counts: `{'ambiguous': 15, 'consistent': 12}`

### 2. Phase Sync Check
**Command**: `python scripts/check_phase_sync.py ...`  
**Result**: `PASS` (No ordering mismatches)

---

# Execution Report: Semantic Probe State Fix

**Executor**: Claude (Antigravity)  
**Date**: 2026-01-09  

---

## Root Cause Identified: **A**

The probe script used `traci.simulationStep()` directly to advance simulation, but `env._last_state_raw` is only populated during `env.step()` calls. The fallback `_read_state_snapshot_from_lanes()` depended on direction inference that didn't reliably work with the config's lane structure.

---

## Changes Made

### Modified: `scripts/semantic_probe_state.py`

Complete rewrite to use TraCI lane metrics directly:

1. **`_get_lane_groups(env, tls_id)`**: Canonical accessor that retrieves NS/EW lane lists from `env._lanes_by_tls` (preferred) or `env._direction_lanes_by_tls` (fallback)

2. **`_snapshot_from_lanes(traci, ns_lanes, ew_lanes)`**: Computes metrics directly via TraCI:
   - `halting_ns/ew`: `traci.lane.getLastStepHaltingNumber()`
   - `waiting_ns/ew`: `traci.lane.getWaitingTime()`
   - `veh_ns/ew`: `traci.lane.getLastStepVehicleNumber()`
   - `vehicle_count`: `traci.vehicle.getIDCount()`

3. **Evidence-based skip logic**:
   - `status="skipped"` only if baseline_queue_proxy < threshold **AND** vehicle_count < 1.0
   - `status="ambiguous" reason="low_queue_signal"` if low queue but vehicles exist

4. **New CSV columns**:
   - `baseline_vehicle_count_avg`, `baseline_halting_ns_avg`, `baseline_halting_ew_avg`
   - `baseline_waiting_ns_avg`, `baseline_waiting_ew_avg`
   - `ns_hold_halting_ns_avg`, `ns_hold_halting_ew_avg`, `ew_hold_halting_ns_avg`, `ew_hold_halting_ew_avg`
   - `imp_ns`, `imp_ew`, `wrong_ns`, `wrong_ew`
   - `error_msg`, `notes`

5. **Updated stdout output** with vehicle_count summary

---

## Verification

### Command
```
python scripts/semantic_probe_state.py --config configs/train_1.yaml --out_dir reports --repeats 3 --warmup_steps 60 --baseline_steps 30 --hold_steps 40 --min_baseline_queue 5
```

### Results
```
[semantic-probe-state] expected_rows=27, written_rows=27
[semantic-probe-state] vehicle_count_avg(min/mean/max)=638.73/2119.71/2933.10
[semantic-probe-state] status_counts={'ambiguous': 12, 'consistent': 9, 'inverted': 6}
```

### Assertions
| Check | Result |
|-------|--------|
| Rows = tls_count × repeats (9×3=27) | ✓ PASS |
| vehicle_count_avg > 0 | ✓ PASS (mean=2119.71) |
| Non-zero halting sums when vehicles exist | ✓ PASS (27/27 rows) |
| No blind skips | ✓ PASS (0 skipped rows) |

---

## Key Outcome

**SUCCESS**: Semantic probe now produces non-zero baseline metrics when vehicles exist.

- **Before**: All rows skipped with `baseline_queue_below_threshold` due to reading stale zeros from `env._last_state_raw`
- **After**: TraCI lane metrics used directly; all 27 rows have non-zero halting sums (min=9.73, max=146.00)

The FAIL status in the report is **correct** - it indicates that some TLS have inverted NS/EW semantics (J14, J2), which is a network configuration issue to be addressed separately, not a measurement bug.
