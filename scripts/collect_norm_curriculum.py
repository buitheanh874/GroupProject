"""
Collect normalization statistics across ALL curriculum phases.
Usage: python scripts/collect_norm_curriculum.py --config configs/train_1.yaml --total-episodes 100 --out configs/norm_curriculum.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_env
from scripts.route_pool_loader import load_route_pool_from_config

repo_root = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect norm stats across curriculum phases")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--total-episodes", type=int, default=100, help="Total episodes across all phases")
    parser.add_argument("--out", type=str, default="configs/norm_curriculum.json")
    parser.add_argument("--fixed-action-id", type=int, default=12)
    parser.add_argument("--max-cycles", type=int, default=None, help="Override max_cycles from config")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    print(f"Collecting norm stats from {len(phases)} phases")
    print(f"Total episodes: {args.total_episodes}")
    print(f"Distribution: {norm_episodes} (based on training ratio)")

    all_raw_states: List[List[float]] = []
    set_global_seed(args.seed)

    for phase_idx, phase in enumerate(phases):
        phase_name = phase.get("name", f"phase{phase_idx}")
        manifest = phase.get("route_pool_manifest", "")
        phase_eps = norm_episodes[phase_idx]
        print(f"\n[Phase {phase_idx + 1}/{len(phases)}] {phase_name} ({phase_eps} episodes)")
        print(f"  Manifest: {manifest}")

        phase_config = load_yaml_config(args.config)
        phase_config.setdefault("run", {})["seed"] = args.seed + phase_idx * 1000
        phase_config.setdefault("env", {}).setdefault("sumo", {})
        phase_config["env"]["sumo"]["normalize_state"] = False
        phase_config["env"]["sumo"]["return_raw_state"] = False
        if args.max_cycles is not None:
            phase_config["env"]["sumo"]["max_cycles"] = args.max_cycles
        phase_config.setdefault("baseline", {})["fixed_action_id"] = args.fixed_action_id
        phase_config.setdefault("train", {})["route_pool_manifest"] = manifest

        route_pool = load_route_pool_from_config(phase_config, split="train", project_root=repo_root)
        env = build_env(phase_config)
        if route_pool and hasattr(env, "set_route_file_pool"):
            try:
                env.set_route_file_pool(route_pool)
            except Exception:
                pass

        fixed_action = args.fixed_action_id
        phase_states: List[List[float]] = []

        for ep in range(phase_eps):
            try:
                state = env.reset()
            except Exception as e:
                print(f"    Episode {ep + 1}: reset failed - {e}")
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
                    action_input = {tls: fixed_action for tls in sorted(state.keys())} if isinstance(state, dict) else fixed_action
                    state, _, done, info = env.step(action_input)
                    step_count += 1
                except Exception as e:
                    print(f"    Episode {ep + 1}: step failed - {e}")
                    break

            if (ep + 1) % 5 == 0 or ep == phase_eps - 1:
                print(f"    Progress: {ep + 1}/{phase_eps} episodes")

        env.close()
        all_raw_states.extend(phase_states)
        print(f"  Collected {len(phase_states)} state samples")

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
        "episodes_per_phase": args.episodes_per_phase,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n[Done] Saved to {out_path}")
    print(f"  State dim: {stats['state_dim']}")
    print(f"  Samples: {stats['num_samples']}")


if __name__ == "__main__":
    main()
