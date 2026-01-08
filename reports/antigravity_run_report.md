# Antigravity Run Report - Deadlock/Teleport Policy

## Run Log - 2026-01-08 00:45

### Executive Summary
Implemented deadlock/gridlock shaping policy as specified in `docs/8_1.md`. The new direction focuses on training RL agents to avoid traffic deadlocks rather than optimizing for teleport avoidance. This approach is more realistic and defensible for academic presentation.

Key changes:
- Added 9 new deadlock config parameters to `SumoEnvConfig`
- Implemented arrival tracking and deadlock detection logic in environment step methods
- Added early-warning shaping penalty and hard deadlock trigger with termination
- Implemented teleport-under-congestion failure rule
- Created new train/eval configs for long-horizon (1800s) episodes with teleport timeout 300s
- Added deadlock fields to KPI tracker and eval CSV output

### Files Changed
- `env/sumo_env.py` - Added deadlock config params, tracking state vars, helper methods, and penalty logic
- `env/kpi.py` - Added deadlock fields to EpisodeKpi dataclass and EpisodeKpiTracker
- `scripts/common.py` - Added loading of deadlock params from YAML
- `scripts/eval.py` - Added deadlock columns to CSV output
- `tests/test_eval_kpi_logging.py` - Updated expected keys to include deadlock columns
- `README.md` - Added section describing new deadlock approach

### Configs Added/Updated
- `configs/train_bignet_9tls_long_tele300.yaml` (NEW) - Training config with deadlock shaping enabled
- `configs/eval_bignet_9tls_long_tele300.yaml` (NEW) - Eval config with deadlock shaping disabled

### Deadlock Policy Implemented
- deadlock_early_no_arrival_sec: 30.0 (train), 0.0 (eval)
- deadlock_no_arrival_sec: 150.0 (train), 0.0 (eval)
- deadlock_queue_threshold: 20.0 (vehicle count based, train only)
- deadlock_downstream_occ_threshold: 0.85 (0-1 scale, train only)
- deadlock_active_min: 30 (minimum active vehicles to trigger, train only)
- deadlock_early_penalty_max: 5.0 (train), 0.0 (eval)
- deadlock_penalty: 100.0 (train), 0.0 (eval)
- terminate_on_deadlock: true (train), false (eval)
- teleport_failure_when_congested: true (train), false (eval)

Queue threshold choice rationale: Set to 20 vehicles as queue metrics are vehicle count based (halting number per lane). This threshold represents moderate congestion across multiple controlled lanes.

### Tests
- Commands:
  - pytest -q
- pytest output:
```
........................................................................ [ 64%]
........................................                                 [100%]
112 passed in 3.87s
```

### Verification
- All 112 tests pass including 5 new deadlock-specific tests
- CSV header will include: deadlock_triggered, deadlock_reason, deadlock_no_arrival_sec
- New tests verify:
  - Deadlock trigger with no arrivals
  - No trigger when active vehicles below minimum
  - Teleport-under-congestion failure rule
  - KPI deadlock fields
  - Eval CSV columns include deadlock fields

### New Test File Created
- `tests/test_deadlock_policy.py` - Tests for deadlock detection and CSV columns
