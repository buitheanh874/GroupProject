from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_agent, build_env


def run_simulation(controller: str, scenario: str, config_path: str, model_path: str = "") -> Dict[str, Any]:
    config = load_yaml_config(config_path)

    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    set_global_seed(seed)

    env = build_env(config)

    scenario_map = config.get("scenario_map", {})
    scenario_key = str(scenario).strip()

    if scenario_key in scenario_map and hasattr(env, "set_route_file"):
        env.set_route_file(str(scenario_map[scenario_key]))

    if hasattr(env, "set_seed"):
        env.set_seed(seed)

    agent: Optional[Any] = None
    controller_key = str(controller).strip().lower()

    if controller_key == "rl":
        agent, _ = build_agent(config, env)
        model_path_value = str(model_path).strip()
        if model_path_value == "":
            model_path_value = str(config.get("runtime", {}).get("model_path", "")).strip()
        if model_path_value == "":
            raise ValueError("model_path is required for RL controller")
        agent.load_model(model_path_value)
        agent.to_eval_mode()

    fixed_action_id = int(config.get("baseline", {}).get("fixed_action_id", 2))

    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    last_info: Dict[str, Any] = {}

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

            if controller_key == "fixed":
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
            total_reward += float(sum(reward_values) / len(reward_values))
            step_count += 1
            state = next_state
        else:
            if controller_key == "fixed":
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

    env.close()

    result: Dict[str, Any] = {
        "controller": controller_key,
        "scenario": scenario_key,
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
    parser.add_argument("--controller", type=str, choices=["fixed", "rl"], default="fixed")
    parser.add_argument("--scenario", type=str, default="default")
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()

    result = run_simulation(
        controller=str(args.controller),
        scenario=str(args.scenario),
        config_path=str(args.config),
        model_path=str(args.model_path),
    )
    print(result)


if __name__ == "__main__":
    main()
