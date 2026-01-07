from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional

from scripts.repo_root import find_repo_root
from pathlib import Path
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


def append_raw_states(
    raw_states: List[List[float]],
    raw_source: Any,
    expected_dim: Optional[int],
) -> Optional[int]:
    if raw_source is None:
        return expected_dim
    if isinstance(raw_source, dict):
        for val in raw_source.values():
            vec = try_vec(val, expected_dim)
            if vec is None:
                continue
            if expected_dim is None:
                expected_dim = len(vec)
            raw_states.append(vec)
        return expected_dim
    vec = try_vec(raw_source, expected_dim)
    if vec is None:
        return expected_dim
    if expected_dim is None:
        expected_dim = len(vec)
    raw_states.append(vec)
    return expected_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_hub_spoke_demo.yaml")
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
            if expected_dim is None:
                state_dim_value = getattr(env, "state_dim", None)
                expected_dim = int(state_dim_value) if state_dim_value is not None else None

            raw_state = env.get_last_state_raw() if hasattr(env, "get_last_state_raw") else None
            if raw_state is None:
                raw_state = state
            expected_dim = append_raw_states(raw_states, raw_state, expected_dim)

            done = False
            while not done:
                action_input = {tls: int(fixed_action_id) for tls in sorted(state.keys())} if isinstance(state, dict) else int(fixed_action_id)
                next_state, _, done, info = env.step(action_input)

                raw_info = info.get("state_raw") if isinstance(info, dict) else None
                if raw_info is not None:
                    expected_dim = append_raw_states(raw_states, raw_info, expected_dim)
                else:
                    expected_dim = append_raw_states(raw_states, next_state, expected_dim)
                state = next_state

    finally:
        env.close()

    if len(raw_states) == 0:
        sys.exit("No raw states collected. Check SUMO configuration and lane grouping.")

    if len(raw_states) < 50:
        sys.exit(
            f"ERROR: Insufficient samples for normalization statistics.\n"
            f"  Collected: {len(raw_states)} samples\n"
            f"  Required: 50+ samples\n"
            f"  Solution: Increase --episodes or --max-cycles\n"
            f"  Recommended: --episodes 10 or more"
        )

    data = np.asarray(raw_states, dtype=np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    if np.any(std < 1e-6):
        problematic = [i for i, s in enumerate(std) if s < 1e-6]
        print(
            f"[WARN] Standard deviation near-zero for features: {problematic}\n"
            f"  This may indicate:\n"
            f"    - Constant values (no variance)\n"
            f"    - Insufficient traffic in simulation\n"
            f"  Clamping to 1e-6 to avoid divide-by-zero"
        )

    std = np.maximum(std, 1e-6)

    if len(raw_states) < 100:
        print(
            f"[WARN] Sample count ({len(raw_states)}) is below recommended (100+).\n"
            f"  Normalization statistics may be less robust."
        )

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
