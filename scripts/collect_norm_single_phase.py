"""
Collect normalization statistics for a single demand level.
Usage: python scripts/collect_norm_single_phase.py --demand 1000 --episodes 10 --out norm_1000.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_env

repo_root = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect norm stats for single demand level")
    parser.add_argument("--demand", type=int, required=True, help="Demand level (e.g., 1000)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file")
    parser.add_argument("--config", type=str, default="configs/train_1.yaml", help="Base config file")
    parser.add_argument("--fixed-action-id", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-sim-seconds",
        type=int,
        default=3600,
        help="Override max_sim_seconds for collection (default: 3600)",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    
    config = load_yaml_config(args.config)
    config.setdefault("run", {})["seed"] = args.seed
    config.setdefault("env", {}).setdefault("sumo", {})
    config["env"]["sumo"]["normalize_state"] = False
    config["env"]["sumo"]["return_raw_state"] = False
    if args.max_sim_seconds is not None:
        config["env"]["sumo"]["max_sim_seconds"] = int(args.max_sim_seconds)
    # Force single-route mode to avoid pulling from any route_pool in the base config
    config["env"]["sumo"]["route_pool"] = []
    
    # Set route file from manifest
    manifest_path = repo_root / f"networks/variants/train_turn801010/{args.demand}/manifest.txt"
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        route_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Collecting norm stats for demand={args.demand}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Route files available: {len(route_files)}")
    
    all_states: List[List[float]] = []
    fixed_action = args.fixed_action_id
    
    for ep in range(args.episodes):
        route_file = route_files[ep % len(route_files)]
        route_path = repo_root / f"networks/variants/train_turn801010/{args.demand}" / route_file
        config["env"]["sumo"]["route_file"] = str(route_path)
        
        env = build_env(config)
        
        try:
            state = env.reset()
        except Exception as e:
            print(f"  Episode {ep + 1}: reset failed - {e}")
            env.close()
            continue
        
        done = False
        step_count = 0
        while not done:
            if isinstance(state, dict):
                for tls_id, s in state.items():
                    if hasattr(s, "tolist"):
                        all_states.append(s.tolist())
                    elif isinstance(s, list):
                        all_states.append(s)
            else:
                if hasattr(state, "tolist"):
                    all_states.append(state.tolist())
                elif isinstance(state, list):
                    all_states.append(state)
            
            try:
                action_input = {tls: fixed_action for tls in sorted(state.keys())} if isinstance(state, dict) else fixed_action
                state, _, done, info = env.step(action_input)
                step_count += 1
            except Exception as e:
                print(f"  Episode {ep + 1}: step failed - {e}")
                break
        
        env.close()
        print(f"  Episode {ep + 1}/{args.episodes} done ({step_count} steps, {len(all_states)} samples)")
    
    if not all_states:
        sys.exit("No state samples collected")
    
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
    print(f"  Samples: {stats['num_samples']}")


if __name__ == "__main__":
    main()
