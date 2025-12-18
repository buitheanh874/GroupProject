from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_agent, build_env, format_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--controller", type=str, choices=["fixed", "rl"], default="fixed")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--max_cycles", type=int, default=10)
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    set_global_seed(seed)

    env = build_env(config)

    agent = None
    if str(args.controller).lower() == "rl":
        agent, _ = build_agent(config, env)
        model_path = str(args.model_path).strip()
        if model_path == "":
            raise ValueError("model_path is required when controller is rl")
        agent.load_model(model_path)
        agent.to_eval_mode()

    fixed_action_id = int(config.get("baseline", {}).get("fixed_action_id", 2))

    if hasattr(env, "set_seed"):
        env.set_seed(seed)

    state = env.reset()
    done = False

    cycle = 0
    while not done and cycle < int(args.max_cycles):
        if str(args.controller).lower() == "fixed":
            action_id = int(fixed_action_id)
        else:
            action_id = int(agent.select_action(state=state, epsilon=0.0))

        next_state, reward, done, info = env.step(action_id)

        state_raw = info.get("state_raw", None) if isinstance(info, dict) else None
        raw_text = str(state_raw) if state_raw is not None else "N/A"

        print(f"Cycle={int(cycle)} | Action={int(action_id)} | Reward={float(reward):.3f} | StateNorm={format_state(next_state)} | StateRaw={raw_text}")

        state = next_state
        cycle += 1

    if hasattr(env, "episode_kpi"):
        kpi = env.episode_kpi()
        print(f"Episode KPI: {kpi}")

    env.close()


if __name__ == "__main__":
    main()
