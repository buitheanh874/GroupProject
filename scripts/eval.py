from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rl.utils import ensure_dir, generate_run_id, load_yaml_config, set_global_seed
from scripts.common import build_agent, build_env


def run_episode(env: Any, controller: str, fixed_action_id: int, agent: Optional[Any] = None) -> Dict[str, Any]:
    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    last_info: Dict[str, Any] = {}

    while not done:
        if controller == "fixed":
            action_id = int(fixed_action_id)
        else:
            action_id = int(agent.select_action(state=state, epsilon=0.0))

        next_state, reward, done, info = env.step(action_id)

        total_reward += float(reward)
        step_count += 1
        state = next_state

        if isinstance(info, dict):
            last_info = info

    kpi = {}
    if isinstance(last_info, dict):
        kpi = last_info.get("episode_kpi", {})

    if hasattr(env, "episode_kpi") and len(kpi) <= 0:
        kpi = env.episode_kpi()

    result: Dict[str, Any] = {
        "total_reward": float(total_reward),
        "episode_steps": int(step_count),
        "arrived_vehicles": int(kpi.get("arrived_vehicles", 0)),
        "avg_wait_time": float(kpi.get("avg_wait_time", 0.0)),
        "avg_travel_time": float(kpi.get("avg_travel_time", 0.0)),
        "avg_stops": float(kpi.get("avg_stops", 0.0)),
        "avg_queue": float(kpi.get("avg_queue", 0.0)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--controller", type=str, choices=["fixed", "rl", "both"], default="both")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    set_global_seed(seed)

    env = build_env(config)

    controller_arg = str(args.controller).lower().strip()
    controllers: List[str]
    if controller_arg == "both":
        controllers = ["fixed", "rl"]
    else:
        controllers = [controller_arg]

    agent = None
    if "rl" in controllers:
        agent, _ = build_agent(config, env)
        model_path = str(args.model_path).strip()
        if model_path == "":
            raise ValueError("model_path is required for RL evaluation")
        agent.load_model(model_path)
        agent.to_eval_mode()

    fixed_action_id = int(config.get("baseline", {}).get("fixed_action_id", 2))

    logging_cfg = config.get("logging", {})
    results_dir = ensure_dir(str(logging_cfg.get("results_dir", "results")))

    scenario_name = str(config.get("scenario", {}).get("name", "")).strip()
    run_id = generate_run_id(prefix="eval")

    results_path = os.path.join(results_dir, f"{run_id}_results.csv")

    with open(results_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "controller",
                "scenario",
                "run_id",
                "total_reward",
                "episode_steps",
                "arrived_vehicles",
                "avg_wait_time",
                "avg_travel_time",
                "avg_stops",
                "avg_queue",
            ],
        )
        writer.writeheader()

        for controller in controllers:
            for run_index in range(int(args.runs)):
                if hasattr(env, "set_seed"):
                    env.set_seed(int(seed + run_index))

                result = run_episode(env=env, controller=controller, fixed_action_id=fixed_action_id, agent=agent)

                row = {
                    "controller": str(controller),
                    "scenario": str(scenario_name),
                    "run_id": int(run_index),
                    **result,
                }
                writer.writerow(row)
                csv_file.flush()

                print(
                    f"Controller={str(controller)} | Run={int(run_index)} | Reward={float(result['total_reward']):.3f} | AvgWait={float(result['avg_wait_time']):.3f} | Arrived={int(result['arrived_vehicles'])}"
                )

    env.close()
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
