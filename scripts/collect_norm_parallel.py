"""
Parallel normalization stats collection for a single demand level.
Usage: python scripts/collect_norm_parallel.py --demand 500 --episodes 9 --workers 9 --out norm_500.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from multiprocessing import Pool, cpu_count

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_env

repo_root = Path(__file__).resolve().parents[1]


def run_single_episode(args_tuple: Tuple[int, int, str, int, str]) -> List[List[float]]:
    """Run a single episode and return collected states."""
    ep_idx, demand, config_path, fixed_action, route_file = args_tuple
    
    set_global_seed(42 + ep_idx)
    
    config = load_yaml_config(config_path)
    config.setdefault("run", {})["seed"] = 42 + ep_idx
    config.setdefault("env", {}).setdefault("sumo", {})
    config["env"]["sumo"]["normalize_state"] = False
    config["env"]["sumo"]["return_raw_state"] = False
    config["env"]["sumo"]["max_sim_seconds"] = 3600
    config["env"]["sumo"]["route_file"] = route_file
    
    states = []
    
    try:
        env = build_env(config)
        state = env.reset()
        done = False
        step_count = 0
        
        while not done:
            if isinstance(state, dict):
                for tls_id, s in state.items():
                    if hasattr(s, "tolist"):
                        states.append(s.tolist())
                    elif isinstance(s, list):
                        states.append(s)
            else:
                if hasattr(state, "tolist"):
                    states.append(state.tolist())
                elif isinstance(state, list):
                    states.append(state)
            
            action_input = {tls: fixed_action for tls in sorted(state.keys())} if isinstance(state, dict) else fixed_action
            state, _, done, info = env.step(action_input)
            step_count += 1
        
        env.close()
        print(f"  [Worker {ep_idx}] Done: {step_count} steps, {len(states)} samples")
        
    except Exception as e:
        print(f"  [Worker {ep_idx}] Error: {e}")
        return []
    
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel norm collection for single demand")
    parser.add_argument("--demand", type=int, required=True, help="Demand level (e.g., 500)")
    parser.add_argument("--episodes", type=int, default=9, help="Number of episodes")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: episodes)")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file")
    parser.add_argument("--config", type=str, default="configs/train_1.yaml", help="Base config")
    parser.add_argument("--fixed-action-id", type=int, default=12)
    args = parser.parse_args()
    
    if args.workers is None:
        args.workers = min(args.episodes, cpu_count())
    
    # Load route files from manifest
    manifest_path = repo_root / f"networks/variants/train_turn801010/{args.demand}/manifest.txt"
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        route_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Parallel Norm Collection: demand={args.demand}")
    print(f"  Episodes: {args.episodes}, Workers: {args.workers}")
    print(f"  Route files available: {len(route_files)}")
    
    # Build task list
    tasks = []
    for ep in range(args.episodes):
        route_file = route_files[ep % len(route_files)]
        route_path = str(repo_root / f"networks/variants/train_turn801010/{args.demand}" / route_file)
        tasks.append((ep, args.demand, args.config, args.fixed_action_id, route_path))
    
    # Run parallel
    all_states = []
    with Pool(processes=args.workers) as pool:
        results = pool.map(run_single_episode, tasks)
    
    for states in results:
        all_states.extend(states)
    
    if not all_states:
        sys.exit("No state samples collected")
    
    # Compute stats
    arr = np.array(all_states, dtype=np.float32)
    mean = np.mean(arr, axis=0).tolist()
    std = np.std(arr, axis=0).tolist()
    std = [max(s, 1e-6) for s in std]
    
    stats = {
        "mean": mean,
        "std": std,
        "state_dim": len(mean),
        "num_samples": len(all_states),
        "demand": args.demand,
        "episodes": args.episodes,
    }
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    
    print(f"\n[Done] Saved to {out_path}")
    print(f"  State dim: {stats['state_dim']}")
    print(f"  Total samples: {stats['num_samples']}")


if __name__ == "__main__":
    main()
