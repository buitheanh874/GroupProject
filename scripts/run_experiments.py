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


def run_episode(env: Any, controller: str, fixed_action_id: int, agent: Optional[Any]) -> Dict[str, Any]:
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

        state = next_state
        total_reward += float(reward)
        step_count += 1

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
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    set_global_seed(seed)

    env = build_env(config)

    fixed_action_id = int(config.get("baseline", {}).get("fixed_action_id", 2))

    scenario_map = config.get("scenario_map", {})
    scenarios = [str(x) for x in config.get("experiments", {}).get("scenarios", [])]
    if len(scenarios) <= 0:
        scenarios = [str(x) for x in scenario_map.keys()]

    controllers = [str(x).lower() for x in config.get("experiments", {}).get("controllers", ["fixed", "rl"])]
    runs_per_setting = int(config.get("experiments", {}).get("runs_per_setting", 10))

    agent: Optional[Any] = None
    if "rl" in controllers:
        agent, _ = build_agent(config, env)
        model_path_value = str(args.model_path).strip()
        if model_path_value == "":
            model_path_value = str(config.get("runtime", {}).get("model_path", "")).strip()
        if model_path_value == "":
            raise ValueError("model_path is required for RL experiments")
        agent.load_model(model_path_value)
        agent.to_eval_mode()

    logging_cfg = config.get("logging", {})
    results_dir = ensure_dir(str(logging_cfg.get("results_dir", "results")))
    run_id = generate_run_id(prefix="experiments")
    results_path = os.path.join(results_dir, f"{run_id}_experiments.csv")

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

        for scenario_index, scenario in enumerate(scenarios):
            route_file = scenario_map.get(str(scenario), None)
            if route_file is not None and hasattr(env, "set_route_file"):
                env.set_route_file(str(route_file))

            for controller in controllers:
                for run_index in range(int(runs_per_setting)):
                    run_seed = int(seed + scenario_index * 10000 + run_index)
                    if hasattr(env, "set_seed"):
                        env.set_seed(run_seed)

                    result = run_episode(env=env, controller=str(controller), fixed_action_id=fixed_action_id, agent=agent)

                    row = {
                        "controller": str(controller),
                        "scenario": str(scenario),
                        "run_id": int(run_index),
                        **result,
                    }
                    writer.writerow(row)
                    csv_file.flush()

                    print(
                        f"Scenario={str(scenario)} | Controller={str(controller)} | Run={int(run_index)} | Reward={float(result['total_reward']):.3f} | AvgWait={float(result['avg_wait_time']):.3f}"
                    )

    env.close()
    print(f"Saved experiments to: {results_path}")


if __name__ == "__main__":
    main()
