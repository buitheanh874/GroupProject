from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config.setdefault("run", {})
    config["run"]["seed"] = int(args.seed)

    set_global_seed(int(args.seed))

    env = build_env(config)

    fixed_action_id = int(config.get("baseline", {}).get("fixed_action_id", 2))

    if hasattr(env, "set_seed"):
        env.set_seed(int(args.seed))

    state = env.reset()
    done = False

    try:
        for cycle in range(int(args.cycles)):
            if done:
                break

            action_id = int(fixed_action_id)
            next_state, reward, done, info = env.step(action_id)

            g_ns = info.get("g_ns") if isinstance(info, dict) else None
            g_ew = info.get("g_ew") if isinstance(info, dict) else None
            decision_sec = info.get("decision_cycle_sec") if isinstance(info, dict) else None
            decision_steps = info.get("decision_steps") if isinstance(info, dict) else None
            waiting_total = info.get("waiting_total") if isinstance(info, dict) else None
            state_raw = info.get("state_raw") if isinstance(info, dict) else None
            state_norm = info.get("state_norm") if isinstance(info, dict) else None

            print(
                f"cycle={cycle+1} action={action_id} g_ns={g_ns} g_ew={g_ew} decision_steps={decision_steps} "
                f"decision_sec={decision_sec} state_raw={state_raw} state_norm={state_norm} "
                f"W={waiting_total} reward={reward}"
            )

            state = next_state
    finally:
        env.close()


if __name__ == "__main__":
    main()
