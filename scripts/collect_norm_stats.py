from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np

from rl.utils import ensure_dir, load_yaml_config, set_global_seed
from scripts.common import build_env


def try_vec(x: Any, expected_dim: Optional[int]) -> Optional[List[float]]:
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if expected_dim is not None and arr.size != int(expected_dim):
            return None
        return [float(v) for v in arr.tolist()]
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_sumo.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="configs/norm_stats.json")
    parser.add_argument("--max-cycles", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config.setdefault("run", {})
    config["run"]["seed"] = int(args.seed)
    config.setdefault("env", {})
    config["env"].setdefault("sumo", {})
    config["env"]["sumo"]["normalize_state"] = False
    config["env"]["sumo"]["return_raw_state"] = False

    if args.max_cycles is not None:
        config["env"]["sumo"]["max_cycles"] = int(args.max_cycles)
    else:
        if config["env"]["sumo"].get("max_cycles", 0) <= 0:
            config["env"]["sumo"]["max_cycles"] = 20

    set_global_seed(int(args.seed))

    env = build_env(config)
    fixed_action_id = int(config.get("baseline", {}).get("fixed_action_id", 2))

    raw_states: List[List[float]] = []
    expected_dim: Optional[int] = None

    try:
        for episode in range(int(args.episodes)):
            if hasattr(env, "set_seed"):
                env.set_seed(int(args.seed + episode))

            state = env.reset()
            expected_dim = int(getattr(env, "state_dim", len(state))) if expected_dim is None else expected_dim

            if isinstance(state, dict):
                for obs in state.values():
                    vec = try_vec(obs, expected_dim)
                    if vec is not None:
                        raw_states.append(vec)
            else:
                vec = try_vec(state, expected_dim)
                if vec is not None:
                    raw_states.append(vec)

            done = False
            while not done:
                action_input = {tls: int(fixed_action_id) for tls in sorted(state.keys())} if isinstance(state, dict) else int(fixed_action_id)
                next_state, _, done, info = env.step(action_input)

                if isinstance(info, dict) and "state_raw" in info:
                    raw_info = info["state_raw"]
                    if isinstance(raw_info, list):
                        vec = try_vec(raw_info, expected_dim)
                        if vec is not None:
                            raw_states.append(vec)
                    elif isinstance(raw_info, dict):
                        for val in raw_info.values():
                            vec = try_vec(val, expected_dim)
                            if vec is not None:
                                raw_states.append(vec)

                if isinstance(next_state, dict):
                    for obs in next_state.values():
                        vec = try_vec(obs, expected_dim)
                        if vec is not None:
                            raw_states.append(vec)
                else:
                    vec = try_vec(next_state, expected_dim)
                    if vec is not None:
                        raw_states.append(vec)

    finally:
        env.close()

    if len(raw_states) == 0:
        sys.exit("No raw states collected. Check SUMO configuration and lane grouping.")

    data = np.asarray(raw_states, dtype=np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    if len(raw_states) < 50:
        print(f"[WARN] Only collected {len(raw_states)} samples; normalization stats may be noisy.")
    if np.any(std < 1e-6):
        print("[WARN] Standard deviation contained near-zero values; clamping to avoid divide-by-zero.")
    std = np.maximum(std, 1e-6)

    output_path = Path(args.out)
    ensure_dir(str(output_path.parent))

    payload = {
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "num_samples": len(raw_states),
        "state_dim": int(expected_dim) if expected_dim is not None else int(data.shape[1]),
        "feature_names": [],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Collected {len(raw_states)} states from {args.episodes} episodes")
    print(f"Mean: {payload['mean']}")
    print(f"Std:  {payload['std']}")
    print(f"Saved normalization stats to {output_path}")


if __name__ == "__main__":
    main()
