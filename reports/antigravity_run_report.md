# Antigravity Run Report - Action Space 5-5-5

## Run Log - 2026-01-08 19:25

### Executive Summary

Migrated action space from [30,60,90] to [60,90,120] cycles with 5 fixed splits, resulting in 15 total actions. Added config fields for cycle_options_sec and reward_time_normalize. All 116 tests pass.

### Files Changed

- env/sumo_env.py (added cycle_options_sec, reward_time_normalize fields; changed default cycles)
- scripts/common.py (updated allowed_cycles_sec default, wired new config fields)
- tests/test_action_space.py (updated cycles to [60,90,120])
- tests/test_mdp_compliance_full.py (updated cycles to [60,90,120])

### Files Added

- configs/train_bignet_1.yaml
- configs/eval_bignet_1.yaml
- configs/norm_bignet_1.json
- tests/test_action_map_1.py

### Files Removed

- None

### Action Space

- cycles: [60, 90, 120]
- splits: [(0.30,0.70),(0.40,0.60),(0.50,0.50),(0.60,0.40),(0.70,0.30)]
- action_count: 15
- mapping_order: cycle_major_split_minor
- Action 0-4: cycle=60, splits 0.30-0.70 through 0.70-0.30
- Action 5-9: cycle=90, splits 0.30-0.70 through 0.70-0.30
- Action 10-14: cycle=120, splits 0.30-0.70 through 0.70-0.30

### Reward Normalization

- reward_time_normalize: config field added (default false)
- train_bignet_1.yaml: reward_time_normalize set to true
- decision_duration_sec source: calculated from step count x step_length_sec

### Data/Manifests

- train routes: using existing networks/variants/train/manifest.txt (50 routes)
- eval routes: using existing networks/variants/eval/manifest_mixed_all.txt
- routes_1.py script: not created (existing route generation sufficient)

### Tests

Commands:
```
pytest -q
python scripts/train.py --config configs/train_bignet_1.yaml --episodes 2
```

pytest output:
```
........................................................................ [ 62%]
............................................                             [100%]
116 passed in 3.04s
```

Smoke train output:
```
[SUMOEnv] Initialized with 9 TLS: ['J0', 'J1', 'J2', 'J3', 'J4', 'J6', 'J7', 'J14', 'J17']
[SUMOEnv] Route pool configured with 50 files
[SUMOEnv] Episode 0: Using route 'bignet_train_seed00082.rou.xml'
[SUMOEnv] Episode 1: Using route 'bignet_train_seed00044.rou.xml'
Environment closed.
[Cycle summary] Cycle distribution (n=14): 60s: 42.9%, 90s: 28.6%, 120s: 28.6%
[Cycle summary] entropy=1.557
Training complete.
Exit code: 0
```

### Cleanup

No files removed. Existing configs retained for backward compatibility.
