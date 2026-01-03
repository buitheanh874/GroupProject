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
