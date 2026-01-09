# Claude Execution Report: Route Manifest Resolution + Phase Sync Fix

## Executive Summary
1. **Route Manifest Fix**: Resolved SUMO crash caused by manifest.txt being passed as route_file. Added automatic manifest resolution and guard.
2. **Phase Sync Fix** (from previous task): PASS/FAIL logic is now evidence-based. Status is PASS when no ordering mismatches or inverted TLS.

## Root Cause: Manifest Resolution Issue
SUMO crashed with "invalid document structure" because:
- configs set `sumo.route_file: networks/variants/train/manifest.txt`
- `load_route_pool_from_config` only set `route_pool` but didn't override `route_file`
- SUMO received manifest.txt directly (not an XML file) and crashed

## Files Changed

### Route Manifest Resolution
- **`scripts/route_pool_loader.py`**:
  - `load_route_pool_from_config` now sets `route_file` to first resolved route
  - Added `resolve_route_file_if_manifest()` for backward compatibility

- **`scripts/common.py`**:
  - Added guard in `build_env` to reject .txt files as route_file with clear error message

- **`scripts/check_phase_sync.py`**:
  - Import and call `resolve_route_file_if_manifest` before `build_env`

- **`scripts/semantic_probe_state.py`**:
  - Import and call `resolve_route_file_if_manifest` before `build_env`

### Phase Sync Status Fix (previous task)
- **`scripts/check_phase_sync.py`**:
  - Refactored status logic: FAIL only on ordering_mismatches OR inverted_tls
  - Ambiguous is now a WARNING, not a failure
  - Added `--require_semantic` and `--min_verified_fraction` flags

## Verification Commands Run

### 1. Phase Sync Check (Default Mode)
```powershell
python scripts/check_phase_sync.py --config configs/train_1.yaml --steps 300 --out_dir reports
```
**Result**: ✅ PASS
- SUMO started with resolved route: `bignet_train_seed00083.rou.xml`
- No manifest parsing error
- Status: PASS (ordering_mismatches=[], inverted_tls=[])
- Warnings: 9 ambiguous TLS (expected, informational only)

### 2. Semantic Probe
```powershell
python scripts/semantic_probe_state.py --config configs/train_1.yaml --out_dir reports --repeats 1 --warmup_steps 30 --baseline_steps 20 --hold_steps 40 --min_baseline_queue 5
```
**Result**: ✅ Script completed (SUMO started successfully)
- Status counts: 6 ambiguous, 2 inverted, 1 consistent
- Inverted TLS: J2, J3 (separate issue, may need tls_phase_overrides)
- Consistent TLS: J6
- Vehicle count: 386 - 3035 (meaningful traffic)

## Key Outputs

### phase_sync_check_report.md
```
Status: PASS
Hard Failures: None
Warnings: ambiguous_tls: ['J0', 'J1', 'J14', 'J17', 'J2', 'J3', 'J4', 'J6', 'J7']
Inverted TLS: []
Ordering mismatches: []
```

### semantic_probe_state.md
```
Status: FAIL (due to inverted TLS detection)
Inverted TLS: ['J2', 'J3']
Consistent TLS: ['J6']
Ambiguous TLS: ['J0', 'J1', 'J14', 'J17', 'J4', 'J7']
```

## Remaining Work
- J2 and J3 show inverted semantics in semantic probe - may need `tls_phase_overrides` in config

## Final Verdict
✅ Route manifest resolution is working correctly. SUMO no longer crashes with manifest.txt.
✅ Phase sync checker correctly returns PASS when no hard failures exist.
