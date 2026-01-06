from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rl.utils import ensure_dir, generate_run_id, load_yaml_config, set_global_seed
from scripts.common import build_agent, build_env, resolve_allowed_action_ids
from controllers.fixed_time import FixedTimeController, FixedTimeControllerConfig
from controllers.max_pressure import MaxPressureSplitController, select_action_from_defs
from scripts.route_pool_loader import load_route_pool_from_config
from scripts.scenario_config_bridge import apply_calibration_overrides
from scripts.config_normalization import normalize_action_table_schema


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--controller", type=str, choices=["fixed", "rl", "max_pressure", "all"], default="all")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--runs", type=int, default=10)
    return parser.parse_args(argv)


def build_eval_row(
    controller: str,
    scenario: str,
    run_id: int,
    total_reward: float,
    episode_steps: int,
    kpi: Dict[str, Any],
) -> Dict[str, Any]:
    arrived = int(kpi.get("arrived_vehicles", 0))
    throughput = float(arrived) / float(max(1, episode_steps))
    return {
        "controller": str(controller),
        "scenario": str(scenario),
        "run_id": int(run_id),
        "total_reward": float(total_reward),
        "episode_steps": int(episode_steps),
        "arrived_vehicles": arrived,
        "avg_wait_time": float(kpi.get("avg_wait_time", 0.0)),
        "avg_travel_time": float(kpi.get("avg_travel_time", 0.0)),
        "avg_stops": float(kpi.get("avg_stops", 0.0)),
        "avg_queue": float(kpi.get("avg_queue", 0.0)),
        "max_wait_time": float(kpi.get("max_wait_time", 0.0)),
        "p95_wait_time": float(kpi.get("p95_wait_time", 0.0)),
        "throughput": throughput,
    }


def build_failed_row(
    controller: str,
    scenario: str,
    run_id: int
) -> Dict[str, Any]:
    return {
        "controller": str(controller),
        "scenario": str(scenario),
        "run_id": int(run_id),
        "total_reward": -99999.0,
        "episode_steps": 0,
        "arrived_vehicles": 0,
        "avg_wait_time": 9999.0,
        "avg_travel_time": 0.0,
        "avg_stops": 0.0,
        "avg_queue": 999.0,
        "max_wait_time": 9999.0,
        "p95_wait_time": 9999.0,
        "throughput": 0.0,
    }


def _resolve_fixed_time_config(config_path: str, scenario_name: str) -> FixedTimeControllerConfig:
    label = f"{config_path} {scenario_name}".lower()
    is_unbalanced = "unbalanced" in label
    target_split = (0.7, 0.3) if is_unbalanced else (0.5, 0.5)
    return FixedTimeControllerConfig(target_split=target_split, target_cycle_sec=90)


def _resolve_action_space(env: Any, config: Dict[str, Any]) -> List[Any]:
    if hasattr(env, "_action_defs"):
        defs = getattr(env, "_action_defs")
        if isinstance(defs, (list, tuple)) and len(defs) > 0:
            return list(defs)

    action_table_root = config.get("action_table", [])
    if isinstance(action_table_root, list) and len(action_table_root) > 0:
        return action_table_root

    sumo_cfg = config.get("env", {}).get("sumo", {})
    action_table = sumo_cfg.get("action_table", [])
    if isinstance(action_table, list) and len(action_table) > 0:
        return action_table

    action_splits = sumo_cfg.get("action_splits", [])
    if isinstance(action_splits, list) and len(action_splits) > 0:
        return action_splits

    return []


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    config = apply_calibration_overrides(config, project_root=repo_root)
    config = normalize_action_table_schema(config)
    route_pool = load_route_pool_from_config(config, split="eval", project_root=repo_root)
    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    set_global_seed(seed)

    env = build_env(config)
    
    if hasattr(env, "route_pool_index"):
        env.route_pool_index = 0
    
    if route_pool and hasattr(env, "set_route_file_pool"):
        try:
            env.set_route_file_pool(route_pool)
        except Exception:
            pass

    scenario_name = str(config.get("scenario", {}).get("name", "")).strip()

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
            model_path = str(config.get("eval", {}).get("model_path", "")).strip()
            if model_path == "":
                raise ValueError("model_path is required for RL evaluation")
            agent.load_model(model_path)
            agent.to_eval_mode()

    baseline_cfg = config.get("baseline", {})
    use_legacy_fixed = "fixed_action_id" in baseline_cfg
    fixed_action_id = int(baseline_cfg.get("fixed_action_id")) if use_legacy_fixed else None
    fixed_fallback_warned = False

    needs_fixed_baseline = any(ctrl in controllers for ctrl in ["fixed", "max_pressure"])
    fixed_time_controller = None
    fixed_target_config = _resolve_fixed_time_config(args.config, scenario_name)
    if not use_legacy_fixed and needs_fixed_baseline:
        action_space = _resolve_action_space(env, config)
        fixed_time_controller = FixedTimeController(action_space=action_space, config=fixed_target_config)
        fixed_action_id = fixed_time_controller.act()
        print(
            "[FixedTime] Using matched action | "
            f"target_split={fixed_target_config.target_split} "
            f"target_cycle={fixed_target_config.target_cycle_sec} | "
            f"selected_action_id={fixed_action_id} "
            f"selected_split={fixed_time_controller.selected_split} "
            f"selected_cycle={fixed_time_controller.selected_cycle_sec}"
        )
    elif use_legacy_fixed and needs_fixed_baseline:
        print(f"[FixedTime] Using legacy fixed_action_id={fixed_action_id} from config.baseline")
    if fixed_action_id is None:
        fixed_action_id = 0

    max_pressure_controller = None
    if "max_pressure" in controllers:
        try:
            sumo_cfg = config.get("env", {}).get("sumo", {})
            lane_cfg = sumo_cfg.get("lane_groups_by_tls", {})
            if not lane_cfg:
                 lane_cfg = sumo_cfg.get("lane_groups", {})
            
            lanes_ns = []
            lanes_ew = []
            splits_ns = [] 
            
            pass 
        except Exception:
            pass

    logging_cfg = config.get("logging", {})
    results_dir = ensure_dir(str(logging_cfg.get("results_dir", "results")))

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
                "throughput",
            ],
        )
        writer.writeheader()

        for controller in controllers:
            if controller == "max_pressure":
                try:
                    sumo_cfg = config.get("env", {}).get("sumo", {})
                    pass 
                except Exception:
                    pass

            for run_index in range(int(args.runs)):
                if hasattr(env, "set_seed"):
                    env.set_seed(int(seed + run_index))

                try:
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

                            action_defs = getattr(env, "_action_defs", [])
                            
                            if controller == "fixed":
                                candidate_action = int(fixed_action_id)
                                allowed_ids = resolve_allowed_action_ids(env, target_action=candidate_action, fallback_action=candidate_action)
                                action_value = candidate_action
                                if allowed_ids:
                                    allowed_ints = [int(x) for x in allowed_ids]
                                    if action_value not in allowed_ints:
                                        fallback_value = int(allowed_ints[0])
                                        if not fixed_fallback_warned:
                                            print(f"[WARN] fixed_action_id={action_value} not in allowed cycle bucket; using {fallback_value} instead.")
                                            fixed_fallback_warned = True
                                        action_value = fallback_value
                                actions = {tls: action_value for tls in tls_ids_sorted}

                            elif controller == "max_pressure":
                                allowed_ids = resolve_allowed_action_ids(env, target_action=None, fallback_action=int(fixed_action_id))
                                default_action = int(allowed_ids[0]) if allowed_ids not in (None, []) else int(fixed_action_id)
                                actions = {}
                                for tls in tls_ids_sorted:
                                    act = select_action_from_defs(
                                        state_raw=state[tls],
                                        action_defs=action_defs,
                                        allowed_action_ids=allowed_ids,
                                        default_action_id=default_action,
                                    )
                                    actions[tls] = int(act)
                            
                            else:
                                if agent:
                                    actions = {}
                                    for tls in tls_ids_sorted:
                                        act = int(agent.select_action(state=state[tls], epsilon=0.0))
                                        actions[tls] = act
                                else:
                                    actions = {tls: 0 for tls in tls_ids_sorted}

                            next_state, rewards, done, info = env.step(actions)
                            reward_values = list(rewards.values()) if isinstance(rewards, dict) else [float(rewards)]
                            total_reward += float(np.mean(reward_values))
                        
                        else:
                            if controller == "fixed":
                                action_id = int(fixed_action_id)
                            elif controller == "max_pressure":
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

                    row = build_eval_row(
                        controller=controller,
                        scenario=scenario_name,
                        run_id=run_index,
                        total_reward=total_reward,
                        episode_steps=step_count,
                        kpi=kpi,
                    )
                    writer.writerow(row)
                    csv_file.flush()

                    print(
                        f"Controller={controller} | Run={run_index} | "
                        f"Reward={total_reward:.3f} | AvgWait={kpi.get('avg_wait_time', 0):.3f} | "
                        f"Arrived={kpi.get('arrived_vehicles', 0)} | Throughput={row['throughput']:.3f}"
                    )

                except Exception as e:
                    print(f"[CRASH RECOVERY] Controller={controller} Run={run_index} failed: {e}")
                    
                    row = build_failed_row(controller, scenario_name, run_index)
                    writer.writerow(row)
                    csv_file.flush()
                    
                    try:
                        env.close()
                    except:
                        pass
                    
                    try:
                        env = build_env(config)
                        if route_pool and hasattr(env, "set_route_file_pool"):
                            env.set_route_file_pool(route_pool)
                        if hasattr(env, "route_pool_index"):
                            env.route_pool_index = run_index + 1
                    except Exception as rebuild_err:
                        print(f"Failed to rebuild env: {rebuild_err}")
                        break

    env.close()
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
