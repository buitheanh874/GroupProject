# GP Project Code Skeleton (M1)

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- PyYAML
- SUMO + TraCI (only required for SUMOEnv)

## Quick Start: Toy Environment

Train on toy queue environment:

python scripts/run_toy_env.py --config configs/train_toy.yaml

Plot rewards:

python analysis/plot_rewards.py --csv logs/<RUN_ID>_train_metrics.csv --window 20

## SUMO Environment (Hub-and-Spoke)

1. Provide hub-and-spoke SUMO assets under `networks/hub_spoke/` (see README there for expected TLS naming). Populate `hub_spoke.net.xml`, `hub_spoke.rou.xml`, and any additional files.
2. Use `configs/train_hub_spoke_demo.yaml` as the template:
   - Fill in `net_file`, `route_file`, `tls_ids/center_tls_id`, `lane_groups_by_tls`, `downstream_links`, and `phase_program`.
   - Keep `state_dim: 12`, `action_table: []` (auto-generates multi-cycle actions), and 12D mean/std.
3. Train: `python scripts/train.py --config configs/train_hub_spoke_demo.yaml`
4. Smoke one episode: `python scripts/run_sumo_episode.py --config configs/eval_hub_spoke.yaml --controller fixed --max_cycles 5`
5. Evaluate / cross-compare: `python scripts/eval.py --config configs/eval_hub_spoke.yaml --controller all --model_path models/hub_spoke_demo.pt --runs 5`

Only the multi-intersection direction is maintained; legacy single-node assets have been removed.

## Hub-and-Spoke (multi-TLS, shared policy)
- Config keys: `tls_ids`, `center_tls_id`, `downstream_links` (N/E/S/W edges or lanes from center), `vehicle_weights` (PCU), `state_dim` (set to 12 for multi layout), and optional `action_table` (cycle + split). `tls_id`/`action_splits` remain supported for user-supplied single-TLS configs.
- Action space default: 15 actions = 3 cycles {30,60,90}s × 5 splits; reward is `-weighted_wait/T_step` with `T_step = cycle_sec + 2*yellow_sec`.
- State (multi mode): `[q_N,q_E,q_S,q_W,w_N,w_E,w_S,w_W,occ_N,occ_E,occ_S,occ_W]`, occupancy only for `center_tls_id`, zero for satellites.
- Sample config: `configs/train_hub_spoke_demo.yaml` (points to `networks/hub_spoke/` placeholder assets).
- Action masking: `env.cycle_to_actions` maps `cycle_sec` to action ids so you can enforce synchronized cycle choices across TLS.
- Validation tips: if `tls_ids` has multiple entries, use `state_dim: 12` and provide `lane_groups_by_tls`. For 12D center occupancy, include `downstream_links` N/E/S/W (missing links now raise ValueError) or set `enable_downstream_occupancy: false` to skip occupancy.

## Action Space Configuration

### Overview

The action space defines how the agent can control traffic signal splits and cycle lengths.
Two configuration modes are supported:

### Mode 1: Fixed Cycle with Split Variations (action_splits)

Use when cycle length is fixed (e.g., always 60s):
```yaml
env:
  sumo:
    green_cycle_sec: 60
    action_splits:
      - [0.30, 0.70]  # 30% NS, 70% EW
      - [0.40, 0.60]
      - [0.50, 0.50]
      - [0.60, 0.40]
      - [0.70, 0.30]
```

This creates 5 discrete actions with different split ratios.

### Mode 2: Dynamic Cycle with Split Variations (action_table)

Use when cycle length can vary (e.g., {30, 60, 90}s):
```yaml
env:
  sumo:
    action_table:
      - {cycle_sec: 30, rho_ns: 0.30, rho_ew: 0.70}
      - {cycle_sec: 30, rho_ns: 0.50, rho_ew: 0.50}
      - {cycle_sec: 30, rho_ns: 0.70, rho_ew: 0.30}
      - {cycle_sec: 60, rho_ns: 0.30, rho_ew: 0.70}
      - {cycle_sec: 60, rho_ns: 0.50, rho_ew: 0.50}
      - {cycle_sec: 60, rho_ns: 0.70, rho_ew: 0.30}
      - {cycle_sec: 90, rho_ns: 0.30, rho_ew: 0.70}
      - {cycle_sec: 90, rho_ns: 0.50, rho_ew: 0.50}
      - {cycle_sec: 90, rho_ns: 0.70, rho_ew: 0.30}

## Teleport Handling Policy
- departed_ids from `simulation.getDepartedIDList()`, arrived_ids from `getArrivedIDList()`, teleported_ids from `getStartingTeleportIDList()`; if only counts are available, teleport_started_total increases conservatively.
- teleport_rate = teleport_unique / max(1, departed_unique); teleported_arrived = |arrived_ids ∩ teleported_ids|.
- arrived_corr = |arrived_ids \ teleported_ids|; failed_corr = |departed_ids| - arrived_corr plus any unknown count-only teleports; completion_rate = |arrived_ids| / max(1, |departed_ids|).
- cap_sec = teleport_time_cap_sec if provided, else the episode duration (at least 1s). Any teleported or not-arrived vehicle is assigned cap_sec for wait/travel; corrected averages/percentiles/max are computed on the capped list. throughput_corr reports arrived_corr per step (queue samples) using these corrected sets.
```

Each action specifies both cycle length and split ratio.

### Auto-generation (Multi-Intersection Mode)

When `state_dim: 12` and `action_table` is empty, the system auto-generates
15 actions = 3 cycles {30,60,90}s × 5 default splits:
```yaml
env:
  sumo:
    state_dim: 12
    action_table: []  # Empty = auto-generate
```

Auto-generated table:
- 3 cycles: {30, 60, 90} seconds
- 5 splits per cycle: {(0.3,0.7), (0.4,0.6), (0.5,0.5), (0.6,0.4), (0.7,0.3)}
- Total: 15 discrete actions

### Constraints

All action configurations must satisfy:
- `rho_ns + rho_ew = 1.0`
- `rho_ns >= rho_min` and `rho_ew >= rho_min` (default: 0.1)
- `g_ns >= g_min_sec` and `g_ew >= g_min_sec` (default: 5s)

Validation is performed by `scripts/validation.py:validate_action_table()`.

### Which Mode to Use?

| Scenario | Recommended Mode | Why |
|----------|------------------|-----|
| Custom single-TLS (user-supplied) | Mode 1 (action_splits) | Simpler, fewer actions |
| Multi-intersection hub-and-spoke | Mode 2 or auto-gen | Coordinate cycles across TLS |
| Research: cycle length impact | Mode 2 (action_table) | Explicit control |

### Example Configs

- Multi-intersection sample: `configs/train_hub_spoke_demo.yaml`
- Toy queue (no SUMO): `configs/train_toy.yaml`

## Normalization Stats Collection Protocol
- Collect raw states with `scripts/collect_norm_stats.py` (or `collect_normalization_stats.py`) using a fixed-action baseline for at least 50 samples; the script emits warnings if sample count is low.
- Standard deviations are clamped to `>=1e-6` to avoid divide-by-zero during normalization; clamping is reported in stdout.
- Keep `normalize_state: true` in training configs and ensure mean/std align with `state_dim`.

## Route Randomization Workflow
- Generate demand variants without SUMO by scaling an existing `.rou.xml`:
  ```
  python scripts/generate_randomized_routes.py --input networks/hub_spoke/hub_spoke.rou.xml --output-dir networks/randomized --variants 5 --seed 42 --global-range 0.7 1.3 --per-flow-noise 0.1
  ```
- The script scales all flow demand fields (probability/vehsPerHour/number) with a deterministic global factor and per-flow noise, preserving begin/end windows.

## Cross-Eval Protocol
- Compare pure vs fairness checkpoints across lambda values:
  ```
  python scripts/cross_eval_fairness.py --config configs/eval_hub_spoke.yaml --pure_ckpt models/pure.pt --fair_ckpt models/fair.pt --lambda_values 0 0.12 --output-dir results/cross_eval
  ```
- Wrapper invokes `scripts/eval.py` per lambda/policy combination and writes logs plus a summary CSV.

## Migration Notes (Audit 2026-01)
- `queue_count_mode` supports only `distinct_cycle`; `snapshot_last_step` now raises a ValueError with guidance.
- Default `include_transition_in_waiting` is `false`; set to `true` explicitly if you need transition phases to contribute to waiting rewards.
- Downstream occupancy is fail-fast: when `enable_downstream_occupancy` is true (12D state), `downstream_links` must provide N/E/S/W IDs, otherwise initialization fails.

## MDP Compliance Map
- See `docs/MDP_COMPLIANCE.md` for key spec/code pointers (reward normalization, queue counting, fairness, spillback/anti-flicker, PCU/enhanced rewards, validation, time-aware gamma).

## Hanoi Scenario Pipeline (MDP style)
1. Inspect boundaries  
   `python scripts/inspect_net_boundaries.py --net networks/hanoi/hanoi.net.xml --out configs/scenario_hanoi_candidates.json`
2. Calibrate (edit `configs/scenario_hanoi_calibration.yaml`): entry/exit edges, PCU weights, demand levels, vehicle mix, turning ratios, optional staged intervals.
3. Generate variants  
   - Train: `python scripts/generate_hanoi_route_variants.py --calib configs/scenario_hanoi_calibration.yaml --out-dir networks/variants --split train --n 50 --seed 42 --skip-router`  
   - Eval (all profiles): `python scripts/generate_hanoi_route_variants.py --calib configs/scenario_hanoi_calibration.yaml --out-dir networks/variants --split eval --n 10 --seed 42 --skip-router`
4. Wire manifests (no YAML globbing)  
   - Train config: `train.route_pool_manifest: networks/variants/train/manifest.txt`  
   - Eval config: `eval.route_pool_manifest: networks/variants/eval/manifest_low.txt` (or other profile manifests)
5. Run  
   - Train: `python scripts/train.py --config configs/train_hub_spoke_demo.yaml`  
   - Eval: `python scripts/eval.py --config configs/eval_hub_spoke.yaml --controller all --runs 3`
6. Optional calibration injection  
   - Set `scenario_calibration: configs/scenario_hanoi_calibration.yaml` to auto-populate `env.sumo.vehicle_weights` (PCU) for `use_pcu_weighted_wait` / `use_enhanced_reward` (with `reward_exponent`).  
   - User-supplied `vehicle_weights` stay unless `force_calibration_overrides: true`.
