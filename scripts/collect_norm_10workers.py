#!/usr/bin/env python
"""
Parallel Normalization Collection with Fixed Workers

Runs exactly N workers in parallel, each collecting M episodes.
All workers use the same demand level (800 veh/hr/lane).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker_collect(
    worker_id: int,
    config: Dict[str, Any],
    num_episodes: int,
    base_port: int,
    seed: int,
    max_sim_seconds: int = 1500,
) -> Tuple[int, List[List[float]]]:
    """Single worker collecting normalization stats."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    
    from rl.utils import set_global_seed
    from scripts.common import build_env
    from scripts.route_pool_loader import load_route_pool_from_config
    
    repo_root = Path(__file__).resolve().parents[1]
    set_global_seed(seed)
    
    # Configure worker
    config = config.copy()
    config.setdefault("run", {})["seed"] = seed
    config.setdefault("env", {}).setdefault("sumo", {})
    config["env"]["sumo"]["normalize_state"] = False
    config["env"]["sumo"]["worker_id"] = worker_id
    config["env"]["sumo"]["base_port"] = base_port
    config["env"]["sumo"]["max_sim_seconds"] = max_sim_seconds  # Override duration
    config["env"]["sumo"]["sumo_extra_args"] = ["--no-warnings", "true"]
    
    print(f"[Worker {worker_id}] Starting {num_episodes} episodes on port {base_port} (max {max_sim_seconds}s)...")
    
    collected_states: List[List[float]] = []
    
    try:
        route_pool = load_route_pool_from_config(config, split="train", project_root=repo_root)
        env = build_env(config)
        if route_pool and hasattr(env, "set_route_file_pool"):
            env.set_route_file_pool(route_pool)
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to create env: {e}")
        return worker_id, []
    
    for ep in range(num_episodes):
        try:
            state = env.reset()
        except Exception as e:
            print(f"[Worker {worker_id}] Episode {ep + 1}: reset failed - {e}")
            continue
        
        done = False
        step_count = 0
        
        while not done:
            # Collect states
            if isinstance(state, dict):
                for tls_id, s in state.items():
                    if hasattr(s, "tolist"):
                        collected_states.append(s.tolist())
                    elif isinstance(s, list):
                        collected_states.append(s)
            else:
                if hasattr(state, "tolist"):
                    collected_states.append(state.tolist())
                elif isinstance(state, list):
                    collected_states.append(state)
            
            # Fixed action (action 12 = 90s cycle, 50/50 split)
            try:
                action = 12  # Fixed action for normalization
                if isinstance(state, dict):
                    action_input = {tls: action for tls in sorted(state.keys())}
                else:
                    action_input = action
                state, _, done, _ = env.step(action_input)
                step_count += 1
            except Exception as e:
                print(f"[Worker {worker_id}] Step error: {e}")
                break
        
        print(f"[Worker {worker_id}] Episode {ep + 1}/{num_episodes} done ({step_count} steps, {len(collected_states)} samples)")
    
    try:
        env.close()
    except Exception:
        pass
    
    print(f"[Worker {worker_id}] Complete: {len(collected_states)} total samples")
    return worker_id, collected_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel norm collection with fixed workers")
    parser.add_argument("--config", type=str, default="configs/train_1.yaml")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--episodes", type=int, default=4, help="Episodes per worker")
    parser.add_argument("--max-sim", type=int, default=1500, help="Max simulation seconds per episode")
    parser.add_argument("--out", type=str, default="configs/norm_curriculum_v5.json")
    parser.add_argument("--base-port", type=int, default=9100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    from rl.utils import load_yaml_config
    
    config = load_yaml_config(args.config)
    total_episodes = args.workers * args.episodes
    
    print(f"Parallel Normalization Collection")
    print(f"  Workers: {args.workers}")
    print(f"  Episodes per worker: {args.episodes}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Max sim seconds: {args.max_sim}")
    print(f"  Base port: {args.base_port}")
    print("-" * 50)
    
    # Create worker tasks
    tasks = []
    for i in range(args.workers):
        worker_port = args.base_port + i
        worker_seed = args.seed + i * 100
        tasks.append((i, config.copy(), args.episodes, worker_port, worker_seed, args.max_sim))
    
    mp.set_start_method("spawn", force=True)
    all_states: List[List[float]] = []
    
    with mp.Pool(processes=args.workers) as pool:
        results = pool.starmap(worker_collect, tasks)
        
        for worker_id, worker_states in results:
            if worker_states:
                all_states.extend(worker_states)
                print(f"Merged {len(worker_states)} samples from Worker {worker_id}")
            else:
                print(f"Warning: No samples from Worker {worker_id}")
    
    if not all_states:
        sys.exit("No state samples collected!")
    
    print("\nComputing statistics...")
    import numpy as np
    arr = np.array(all_states, dtype=np.float32)
    mean = np.mean(arr, axis=0).tolist()
    std = np.std(arr, axis=0).tolist()
    std = [max(s, 1e-6) for s in std]  # Prevent division by zero
    
    stats = {
        "mean": mean,
        "std": std,
        "state_dim": len(mean),
        "num_samples": len(all_states),
        "num_workers": args.workers,
        "episodes_per_worker": args.episodes,
    }
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    
    print(f"\n[Done] Saved to {out_path}")
    print(f"  State dim: {stats['state_dim']}")
    print(f"  Samples: {stats['num_samples']}")
    print(f"  Mean: {mean[:3]}... (showing first 3)")
    print(f"  Std: {std[:3]}... (showing first 3)")


if __name__ == "__main__":
    main()
