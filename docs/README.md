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
