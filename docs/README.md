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

## SUMO Environment

1. Prepare SUMO network and route files.
2. Update configs/train_sumo.yaml:
   - net_file, route_file
   - tls_id
   - lane_groups (lanes_ns_ctrl, lanes_ew_ctrl)
   - phase_program indices
   - normalization mean/std

Train:

python scripts/train.py --config configs/train_sumo.yaml

Sanity-check one episode:

python scripts/run_sumo_episode.py --config configs/eval_sumo.yaml --controller fixed --max_cycles 10

Evaluate:

python scripts/eval.py --config configs/eval_sumo.yaml --controller both --model_path models/<MODEL>.pt --runs 10

Run experiment grid (fixed vs rl, low/medium/high):

python scripts/run_experiments.py --config configs/experiments.yaml --model_path models/<MODEL>.pt

## Hub-and-Spoke (multi-TLS, shared policy)
- New config keys: `tls_ids` (list), `center_tls_id`, `downstream_links` (N/E/S/W edges or lanes from center), `vehicle_weights` (PCU), `state_dim` (set to 12 for multi layout), and optional `action_table` (cycle + split). Legacy `tls_id`/`action_splits` stay supported.
- Action space default: 15 actions = 3 cycles {30,60,90}s × 5 splits; reward is `-weighted_wait/T_step` with `T_step = cycle_sec + 2*yellow_sec`.
- State (multi mode): `[q_N,q_E,q_S,q_W,w_N,w_E,w_S,w_W,occ_N,occ_E,occ_S,occ_W]`, occupancy only for `center_tls_id`, zero for satellites.
- Sample config: `configs/train_hub_spoke_demo.yaml` (runs on `networks/BI.net.xml`, single TLS but 12D state/action-table enabled). Multi-node layouts require SUMO files in `networks/hub_spoke/` (see README there).
- Action masking: `env.cycle_to_actions` maps `cycle_sec` to action ids so you can enforce synchronized cycle choices across TLS.
- Validation tips: if `tls_ids` has multiple entries, use `state_dim: 12` and provide `lane_groups_by_tls`. For 12D center occupancy, include `downstream_links` N/E/S/W or set `enable_downstream_occupancy: false` to skip occupancy.

## Normalization Stats Collection Protocol
- Collect raw states with `scripts/collect_norm_stats.py` (or `collect_normalization_stats.py`) using a fixed-action baseline for at least 50 samples; the script emits warnings if sample count is low.
- Standard deviations are clamped to `>=1e-6` to avoid divide-by-zero during normalization; clamping is reported in stdout.
- Keep `normalize_state: true` in training configs and ensure mean/std align with `state_dim`.

## Route Randomization Workflow
- Generate demand variants without SUMO by scaling an existing `.rou.xml`:
  ```
  python scripts/generate_randomized_routes.py --input networks/BIGMAP.rou.xml --output-dir networks/randomized --variants 5 --seed 42 --global-range 0.7 1.3 --per-flow-noise 0.1
  ```
- The script scales all flow demand fields (probability/vehsPerHour/number) with a deterministic global factor and per-flow noise, preserving begin/end windows.

## Cross-Eval Protocol
- Compare pure vs fairness checkpoints across lambda values:
  ```
  python scripts/cross_eval_fairness.py --config configs/eval_sumo.yaml --pure_ckpt models/pure.pt --fair_ckpt models/fair.pt --lambda_values 0 0.12 --output-dir results/cross_eval
  ```
- Wrapper invokes `scripts/eval.py` per lambda/policy combination and writes logs plus a summary CSV.

## MDP Compliance Map
- See `docs/MDP_COMPLIANCE.md` for key spec→code pointers (reward normalization, queue counting, fairness, spillback/anti-flicker, PCU/enhanced rewards, validation, time-aware gamma).
