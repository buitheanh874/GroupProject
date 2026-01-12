from __future__ import annotations

import argparse
import json
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def collect_phase_stats(
    phase_idx: int,
    phase_config: Dict[str, Any],
    manifest: str,
    num_episodes: int,
    fixed_action_id: int,
    base_port: int,
    seed: int,
    result_queue: Queue,
) -> None:
    import numpy as np
    import torch
    torch.set_num_threads(1)
    
    from rl.utils import set_global_seed
    from scripts.common import build_env
    from scripts.route_pool_loader import load_route_pool_from_config
    
    repo_root = Path(__file__).resolve().parents[1]
    
    set_global_seed(seed)
    
    phase_config = phase_config.copy()
    phase_config.setdefault("run", {})["seed"] = seed
    phase_config.setdefault("env", {}).setdefault("sumo", {})
    phase_config["env"]["sumo"]["normalize_state"] = False
    phase_config["env"]["sumo"]["return_raw_state"] = False
    phase_config["env"]["sumo"]["worker_id"] = phase_idx
    phase_config["env"]["sumo"]["base_port"] = base_port
    phase_config.setdefault("baseline", {})["fixed_action_id"] = fixed_action_id
    phase_config.setdefault("train", {})["route_pool_manifest"] = manifest
    
    try:
        route_pool = load_route_pool_from_config(phase_config, split="train", project_root=repo_root)
        env = build_env(phase_config)
        if route_pool and hasattr(env, "set_route_file_pool"):
            try:
                env.set_route_file_pool(route_pool)
            except Exception:
                pass
    except Exception as e:
        print(f"[Phase {phase_idx}] Failed to create env: {e}")
        result_queue.put((phase_idx, []))
        return
    
    phase_states: List[List[float]] = []
    
    for ep in range(num_episodes):
        try:
            state = env.reset()
        except Exception as e:
            print(f"[Phase {phase_idx}] Episode {ep + 1}: reset failed - {e}")
            continue
        
        done = False
        step_count = 0
        while not done:
            if isinstance(state, dict):
                for tls_id, s in state.items():
                    if hasattr(s, "tolist"):
                        phase_states.append(s.tolist())
                    elif isinstance(s, list):
                        phase_states.append(s)
            else:
                if hasattr(state, "tolist"):
                    phase_states.append(state.tolist())
                elif isinstance(state, list):
                    phase_states.append(state)
            
            try:
                action_input = {tls: fixed_action_id for tls in sorted(state.keys())} if isinstance(state, dict) else fixed_action_id
                state, _, done, info = env.step(action_input)
                step_count += 1
            except Exception as e:
                break
        
        print(f"[Phase {phase_idx}] Episode {ep + 1}/{num_episodes} done ({step_count} steps)")
    
    try:
        env.close()
    except Exception:
        pass
    
    print(f"[Phase {phase_idx}] Complete: {len(phase_states)} samples")
    result_queue.put((phase_idx, phase_states))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--total-episodes", type=int, default=40)
    parser.add_argument("--out", type=str, default="configs/norm_curriculum_v4.json")
    parser.add_argument("--fixed-action-id", type=int, default=12)
    parser.add_argument("--base-port", type=int, default=8900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    from rl.utils import load_yaml_config
    
    config = load_yaml_config(args.config)
    curriculum_cfg = config.get("curriculum", {})
    phases = curriculum_cfg.get("phases", [])
    
    if not phases:
        sys.exit("No curriculum phases found in config")
    
    train_episodes = [phase.get("episodes", 100) for phase in phases]
    total_train = sum(train_episodes)
    norm_episodes = [max(1, int(args.total_episodes * ep / total_train)) for ep in train_episodes]
    
    diff = args.total_episodes - sum(norm_episodes)
    if diff != 0:
        max_idx = norm_episodes.index(max(norm_episodes))
        norm_episodes[max_idx] += diff
    
    print(f"Parallel norm: {len(phases)} phases, {args.total_episodes} total episodes")
    print(f"Distribution: {norm_episodes}")
    print(f"Base port: {args.base_port}")
    print()
    
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    
    processes: List[Process] = []
    for i, phase in enumerate(phases):
        manifest = phase.get("route_pool_manifest", "")
        phase_seed = args.seed + i * 1000
        
        p = Process(
            target=collect_phase_stats,
            args=(
                i,
                config.copy(),
                manifest,
                norm_episodes[i],
                args.fixed_action_id,
                args.base_port,
                phase_seed,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)
        print(f"Started phase {i}: {phase.get('name', f'phase{i}')} ({norm_episodes[i]} eps, port {args.base_port + i})")
    
    all_raw_states: List[List[float]] = []
    results_collected = 0
    
    while results_collected < len(phases):
        try:
            phase_idx, phase_states = result_queue.get(timeout=600)
            all_raw_states.extend(phase_states)
            results_collected += 1
            print(f"Received results from phase {phase_idx}: {len(phase_states)} samples")
        except Exception as e:
            print(f"Timeout or error waiting for results: {e}")
            break
    
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    
    if not all_raw_states:
        sys.exit("No state samples collected")
    
    import numpy as np
    arr = np.array(all_raw_states, dtype=np.float32)
    mean = np.mean(arr, axis=0).tolist()
    std = np.std(arr, axis=0).tolist()
    std = [max(s, 1e-6) for s in std]
    
    stats = {
        "mean": mean,
        "std": std,
        "state_dim": len(mean),
        "num_samples": len(all_raw_states),
        "num_phases": len(phases),
        "episodes_per_phase": norm_episodes,
    }
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    
    print(f"\n[Done] Saved to {out_path}")
    print(f"  State dim: {stats['state_dim']}")
    print(f"  Samples: {stats['num_samples']}")


if __name__ == "__main__":
    main()
