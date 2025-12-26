from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np

from rl.utils import ensure_dir, load_yaml_config, set_global_seed
from scripts.common import build_env


def try_vec4(x) -> Optional[List[float]]:
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size != 4:
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

    try:
        for episode in range(int(args.episodes)):
            if hasattr(env, "set_seed"):
                env.set_seed(int(args.seed + episode))

            state = env.reset()
            vec = try_vec4(state)
            if vec is not None:
                raw_states.append(vec)

            done = False
            while not done:
                next_state, _, done, info = env.step(int(fixed_action_id))
                
                vec = try_vec4(next_state)
                if vec is not None:
                    raw_states.append(vec)

    finally:
        env.close()

    if len(raw_states) == 0:
        sys.exit("No raw states collected. Check SUMO configuration and lane grouping.")

    data = np.asarray(raw_states, dtype=np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.maximum(std, 1e-6)

    output_path = Path(args.out)
    ensure_dir(str(output_path.parent))

    payload = {
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "num_samples": len(raw_states),
        "state_dim": 4,
        "feature_names": ["q_NS", "q_EW", "w_NS", "w_EW"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Collected {len(raw_states)} states from {args.episodes} episodes")
    print(f"Mean: {payload['mean']}")
    print(f"Std:  {payload['std']}")
    print(f"Saved normalization stats to {output_path}")


if __name__ == "__main__":
    main()