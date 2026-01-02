from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rl.utils import ensure_dir, generate_run_id, load_yaml_config, set_global_seed
from scripts.common import build_agent, build_env
from controllers.max_pressure import MaxPressureSplitController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--controller", type=str, choices=["fixed", "rl", "max_pressure", "all"], default="all")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    set_global_seed(seed)

    env = build_env(config)

    controller_arg = str(args.controller).lower().strip()
    if controller_arg == "all":
        controllers = ["fixed", "max_pressure", "rl"]
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

    max_pressure_controller = None
    if "max_pressure" in controllers:
        sumo_cfg = config.get("env", {}).get("sumo", {})
        lane_cfg = sumo_cfg.get("lane_groups", {})
        
        lanes_ns = [str(x) for x in lane_cfg.get("lanes_ns_ctrl", [])]
        lanes_ew = [str(x) for x in lane_cfg.get("lanes_ew_ctrl", [])]
        
        splits_raw = sumo_cfg.get("action_splits", [])
        splits_ns = [float(x[0]) for x in splits_raw] if len(splits_raw) > 0 else []
        
        if len(lanes_ns) > 0 and len(lanes_ew) > 0 and len(splits_ns) > 0 and getattr(env, "state_dim", 4) == 4:
            max_pressure_controller = MaxPressureSplitController(
                lanes_ns=lanes_ns,
                lanes_ew=lanes_ew,
                splits_ns=splits_ns,
            )

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
                "max_wait_time",
                "p95_wait_time",
            ],
        )
        writer.writeheader()

        for controller in controllers:
            for run_index in range(int(args.runs)):
                if hasattr(env, "set_seed"):
                    env.set_seed(int(seed + run_index))

                state = env.reset()
                done = False
                total_reward = 0.0
                step_count = 0
                last_info = {}

                while not done:
                    if isinstance(state, dict):
                        tls_ids_sorted = sorted(state.keys())
                        center_id = None
                        if hasattr(env, "center_tls_id"):
                            center_candidate = getattr(env, "center_tls_id")
                            if isinstance(center_candidate, str) and center_candidate in tls_ids_sorted:
                                center_id = center_candidate
                        if center_id is None:
                            center_id = tls_ids_sorted[0]

                        if controller == "fixed":
                            actions = {tls: int(fixed_action_id) for tls in tls_ids_sorted}
                        elif controller == "max_pressure":
                            actions = {tls: int(fixed_action_id) for tls in tls_ids_sorted}
                        else:
                            center_action = int(agent.select_action(state=state[center_id], epsilon=0.0))
                            allowed_ids = None
                            if hasattr(env, "cycle_to_actions"):
                                for _, ids in env.cycle_to_actions.items():
                                    if center_action in ids:
                                        allowed_ids = [int(x) for x in ids]
                                        break
                                if allowed_ids is None:
                                    for _, ids in env.cycle_to_actions.items():
                                        if int(fixed_action_id) in ids:
                                            allowed_ids = [int(x) for x in ids]
                                            break
                            actions = {tls: int(agent.select_action(state=state[tls], epsilon=0.0, allowed_action_ids=allowed_ids)) for tls in tls_ids_sorted}

                        next_state, rewards, done, info = env.step(actions)
                        reward_values = list(rewards.values()) if isinstance(rewards, dict) else [float(rewards)]
                        total_reward += float(np.mean(reward_values))
                        step_count += 1
                        state = next_state
                        if isinstance(info, dict):
                            last_info = info
                    else:
                        if controller == "fixed":
                            action_id = int(fixed_action_id)
                        elif controller == "max_pressure":
                            if max_pressure_controller is not None and hasattr(env, "get_last_state_raw"):
                                state_raw = env.get_last_state_raw()
                                if state_raw is not None:
                                    action_id = max_pressure_controller.select_action(state_raw)
                                else:
                                    action_id = int(fixed_action_id)
                            else:
                                action_id = int(fixed_action_id)
                        else:
                            action_id = int(agent.select_action(state=state, epsilon=0.0))

                        next_state, reward, done, info = env.step(action_id)
                        total_reward += float(reward)
                        step_count += 1
                        state = next_state
                        if isinstance(info, dict):
                            last_info = info

                kpi = last_info.get("episode_kpi", {}) if last_info else {}
                if hasattr(env, "episode_kpi") and len(kpi) <= 0:
                    kpi = env.episode_kpi()

                row = {
                    "controller": str(controller),
                    "scenario": str(scenario_name),
                    "run_id": int(run_index),
                    "total_reward": float(total_reward),
                    "episode_steps": int(step_count),
                    "arrived_vehicles": int(kpi.get("arrived_vehicles", 0)),
                    "avg_wait_time": float(kpi.get("avg_wait_time", 0.0)),
                    "avg_travel_time": float(kpi.get("avg_travel_time", 0.0)),
                    "avg_stops": float(kpi.get("avg_stops", 0.0)),
                    "avg_queue": float(kpi.get("avg_queue", 0.0)),
                    "max_wait_time": float(kpi.get("max_wait_time", 0.0)),
                    "p95_wait_time": float(kpi.get("p95_wait_time", 0.0)),
                }
                writer.writerow(row)
                csv_file.flush()

                print(
                    f"Controller={controller} | Run={run_index} | "
                    f"Reward={total_reward:.3f} | AvgWait={kpi.get('avg_wait_time', 0):.3f} | "
                    f"MaxWait={kpi.get('max_wait_time', 0):.3f} | "
                    f"Arrived={kpi.get('arrived_vehicles', 0)}"
                )

    env.close()
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
