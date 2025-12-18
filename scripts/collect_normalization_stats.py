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

    for episode in range(int(args.episodes)):
        if hasattr(env, "set_seed"):
            env.set_seed(int(seed + episode))

        state = env.reset()
        done = False

        while not done:
            next_state, reward, done, info = env.step(int(fixed_action_id))
            if isinstance(info, dict) and "state_raw" in info:
                raw = info["state_raw"]
                if isinstance(raw, list) and len(raw) == 4:
                    raw_states.append([float(x) for x in raw])
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
