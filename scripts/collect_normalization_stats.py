from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rl.utils import load_yaml_config, save_yaml_config, set_global_seed
from scripts.common import build_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    set_global_seed(seed)

    env = build_env(config)

    fixed_action_id = int(config.get("baseline", {}).get("fixed_action_id", 2))

    raw_states: List[List[float]] = []
    expected_dim = int(getattr(env, "state_dim", 4))

    for episode in range(int(args.episodes)):
        if hasattr(env, "set_seed"):
            env.set_seed(int(seed + episode))

        state = env.reset()
        if isinstance(state, dict):
            for value in state.values():
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                if arr.size == expected_dim:
                    raw_states.append([float(x) for x in arr.tolist()])
        else:
            arr = np.asarray(state, dtype=np.float32).reshape(-1)
            if arr.size == expected_dim:
                raw_states.append([float(x) for x in arr.tolist()])
        done = False

        while not done:
            action_input = {tls: int(fixed_action_id) for tls in sorted(state.keys())} if isinstance(state, dict) else int(fixed_action_id)
            next_state, reward, done, info = env.step(action_input)
            if isinstance(info, dict) and "state_raw" in info:
                raw = info["state_raw"]
                if isinstance(raw, list) and len(raw) == expected_dim:
                    raw_states.append([float(x) for x in raw])
                elif isinstance(raw, dict):
                    for value in raw.values():
                        if isinstance(value, list) and len(value) == expected_dim:
                            raw_states.append([float(x) for x in value])
            elif isinstance(next_state, dict):
                for value in next_state.values():
                    arr = np.asarray(value, dtype=np.float32).reshape(-1)
                    if arr.size == expected_dim:
                        raw_states.append([float(x) for x in arr.tolist()])
            else:
                arr = np.asarray(next_state, dtype=np.float32).reshape(-1)
                if arr.size == expected_dim:
                    raw_states.append([float(x) for x in arr.tolist()])
            state = next_state

    env.close()

    if len(raw_states) <= 0:
        raise RuntimeError("No raw states collected. Check SUMO configuration and lane grouping.")

    data = np.asarray(raw_states, dtype=np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.maximum(std, 1e-6)

    config["normalization"] = {
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
    }

    save_yaml_config(config, args.out)
    print(f"Saved updated config with normalization stats to: {str(args.out)}")


if __name__ == "__main__":
    main()
