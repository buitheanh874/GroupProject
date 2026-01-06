from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np

from scripts.repo_root import find_repo_root

project_root = find_repo_root(__file__)
sys.path.insert(0, str(project_root))

from rl.cycle_tracker import CycleDistributionTracker
from rl.utils import ensure_dir, generate_run_id, linear_epsilon, load_yaml_config, save_yaml_config, set_global_seed
from scripts.common import build_agent, build_env, resolve_allowed_action_ids
from scripts.route_pool_loader import load_route_pool_from_config
from scripts.scenario_config_bridge import apply_calibration_overrides
from scripts.config_normalization import normalize_action_table_schema


def run_training(config: Dict[str, Any]) -> str:
    config = apply_calibration_overrides(config, project_root=project_root)
    config = normalize_action_table_schema(config)
    train_cfg = config.get("train", {})
    route_pool = load_route_pool_from_config(config, split="train", project_root=project_root)

    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    set_global_seed(seed)

    env = build_env(config)
    if route_pool and hasattr(env, "set_route_file_pool"):
        try:
            env.set_route_file_pool(route_pool)
        except Exception:
            pass

    agent, _ = build_agent(config, env)
    agent.to_train_mode()

    run_name = str(run_cfg.get("run_name", "train"))
    run_id = generate_run_id(prefix=run_name)

    logging_cfg = config.get("logging", {})
    log_dir = ensure_dir(str(logging_cfg.get("log_dir", "logs")))
    model_dir = ensure_dir(str(logging_cfg.get("model_dir", "models")))
    ensure_dir(str(logging_cfg.get("results_dir", "results")))

    config_copy_path = os.path.join(log_dir, f"{run_id}_config.yaml")
    save_yaml_config(config, config_copy_path)

    metrics_path = os.path.join(log_dir, f"{run_id}_train_metrics.csv")

    episodes = int(train_cfg.get("episodes", 200))
    save_every_episodes = int(train_cfg.get("save_every_episodes", 50))
    print_every_episodes = int(train_cfg.get("print_every_episodes", 10))

    exploration_cfg = config.get("exploration", {})
    eps_start = float(exploration_cfg.get("eps_start", 1.0))
    eps_end = float(exploration_cfg.get("eps_end", 0.05))
    eps_decay_steps = int(exploration_cfg.get("eps_decay_steps", 50000))

    allowed_cycles = []
    if hasattr(env, "_action_defs"):
        allowed_cycles = sorted(set(int(a.cycle_sec) for a in env._action_defs))
    cycle_tracker = CycleDistributionTracker(allowed_cycles) if len(allowed_cycles) > 0 else None
    log_cycle_every = int(train_cfg.get("log_cycle_distribution_every", 10))

    best_reward = -float("inf")
    global_step = 0

    try:
        with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "episode",
                "episode_reward",
                "avg_loss",
                "episode_steps",
                "global_step",
                "num_tls",
                "epsilon_end",
                "arrived_vehicles",
                "avg_wait_time",
                "avg_travel_time",
                "avg_stops",
                "avg_queue",
                "decision_cycle_sec",
                "decision_steps",
                "waiting_total",
            ]
            
            if len(allowed_cycles) > 0:
                for cycle in allowed_cycles:
                    fieldnames.append(f"cycle_{cycle}_count")
            if cycle_tracker is not None:
                for cycle in allowed_cycles:
                    fieldnames.append(f"cycle_{cycle}_pct")
                fieldnames.append("cycle_entropy")
            
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for episode in range(1, int(episodes) + 1):
                if hasattr(env, "set_seed"):
                    env.set_seed(int(seed + episode))

                state = env.reset()
                episode_cycle_counts = Counter({cycle: 0 for cycle in allowed_cycles})

                done = False
                episode_reward = 0.0
                episode_steps = 0
                last_epsilon = float(eps_start)

                losses = []
                info: Dict[str, Any] = {}

                while not done:
                    epsilon = linear_epsilon(
                        global_step=global_step,
                        eps_start=eps_start,
                        eps_end=eps_end,
                        decay_steps=eps_decay_steps,
                    )
                    last_epsilon = float(epsilon)

                    if isinstance(state, dict):
                        tls_ids_sorted = sorted(state.keys())
                        center_id = None
                        if hasattr(env, "center_tls_id"):
                            center_id_candidate = getattr(env, "center_tls_id")
                            if isinstance(center_id_candidate, str) and center_id_candidate in tls_ids_sorted:
                                center_id = center_id_candidate
                        if center_id is None:
                            center_id = tls_ids_sorted[0]

                        center_action = agent.select_action(state=state[center_id], epsilon=epsilon)
                        allowed_ids = resolve_allowed_action_ids(
                            env=env,
                            target_action=center_action,
                            fallback_action=int(config.get("baseline", {}).get("fixed_action_id", 2)),
                        )

                        actions: Dict[str, int] = {}
                        for tls_id in tls_ids_sorted:
                            actions[str(tls_id)] = agent.select_action(
                                state=state[tls_id],
                                epsilon=epsilon,
                                allowed_action_ids=allowed_ids,
                            )

                        next_state, rewards, done, info = env.step(actions)

                        step_rewards = list(rewards.values()) if isinstance(rewards, dict) else [float(rewards)]
                        step_reward = float(np.mean(step_rewards))

                        t_step_value = None
                        if isinstance(info, dict):
                            t_step_value = info.get("t_step") or info.get("decision_cycle_sec")
                            if t_step_value is not None:
                                t_step_value = float(t_step_value)
                        gamma_value = agent.compute_gamma(t_step_value)

                        for tls_id in tls_ids_sorted:
                            action_id = actions[tls_id]
                            next_obs = next_state.get(tls_id) if isinstance(next_state, dict) else next_state
                            reward_value = rewards.get(tls_id, 0.0) if isinstance(rewards, dict) else rewards
                            agent.store_transition(state[tls_id], action_id, reward_value, next_obs, done, gamma=gamma_value)

                        loss_value = agent.update()
                        if loss_value is not None:
                            losses.append(float(loss_value))

                        state = next_state
                        episode_reward += float(step_reward)
                        episode_steps += 1
                        global_step += len(actions)
                    else:
                        action_id = agent.select_action(state=state, epsilon=epsilon)
                        next_state, reward, done, info = env.step(action_id)

                        gamma_value = agent.compute_gamma(info.get("t_step") if isinstance(info, dict) else None)
                        agent.store_transition(state, action_id, reward, next_state, done, gamma=gamma_value)
                        loss_value = agent.update()
                        if loss_value is not None:
                            losses.append(float(loss_value))

                        state = next_state
                        episode_reward += float(reward)
                        episode_steps += 1
                        global_step += 1
                    
                    if isinstance(info, dict):
                        cycle_key = info.get("cycle_sec")
                        if cycle_key is None:
                            cycle_key = info.get("green_cycle_sec")
                        if cycle_key is not None:
                            if cycle_tracker is not None:
                                cycle_tracker.record(int(cycle_key))
                            if cycle_key in episode_cycle_counts:
                                episode_cycle_counts[int(cycle_key)] += 1

                avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

                kpi = {}
                if isinstance(info, dict):
                    kpi = info.get("episode_kpi", {}) if done else {}
                    
                    if cycle_tracker is not None:
                        cycle_key = info.get("cycle_sec")
                        if cycle_key is None:
                            cycle_key = info.get("green_cycle_sec")
                        if cycle_key is not None:
                            cycle_tracker.record(int(cycle_key))

                row: Dict[str, Any] = {
                    "episode": int(episode),
                    "episode_reward": float(episode_reward),
                    "avg_loss": float(avg_loss),
                    "episode_steps": int(episode_steps),
                    "global_step": int(global_step),
                    "num_tls": int(len(state) if isinstance(state, dict) else 1),
                    "epsilon_end": float(last_epsilon),
                    "arrived_vehicles": int(kpi.get("arrived_vehicles", 0)),
                    "avg_wait_time": float(kpi.get("avg_wait_time", 0.0)),
                    "avg_travel_time": float(kpi.get("avg_travel_time", 0.0)),
                    "avg_stops": float(kpi.get("avg_stops", 0.0)),
                    "avg_queue": float(kpi.get("avg_queue", 0.0)),
                    "decision_cycle_sec": float(info.get("decision_cycle_sec", 0.0)) if isinstance(info, dict) else 0.0,
                    "decision_steps": int(info.get("decision_steps", 0)) if isinstance(info, dict) else 0,
                    "waiting_total": float(
                        info.get("waiting_total", info.get("total_wait_reward", info.get("total_weighted_wait", 0.0))) if isinstance(info, dict) else 0.0
                    ),
                }
                
                if len(allowed_cycles) > 0:
                    for cycle in allowed_cycles:
                        row[f"cycle_{cycle}_count"] = int(episode_cycle_counts.get(cycle, 0))
                if cycle_tracker is not None:
                    cycle_dist = cycle_tracker.get_distribution()
                    for cycle in allowed_cycles:
                        row[f"cycle_{cycle}_pct"] = float(cycle_dist.get(cycle, 0.0))
                    row["cycle_entropy"] = float(cycle_tracker.get_entropy())

                writer.writerow(row)
                csv_file.flush()

                losses.clear()
                del row

                should_save_periodic = int(save_every_episodes) > 0 and (int(episode) % int(save_every_episodes) == 0)
                is_best = float(episode_reward) > float(best_reward)

                if is_best:
                    best_reward = float(episode_reward)

                if should_save_periodic:
                    model_path = os.path.join(model_dir, f"{run_id}_episode_{int(episode)}.pt")
                    agent.save_model(model_path)

                if is_best:
                    best_model_path = os.path.join(model_dir, f"{run_id}_best.pt")
                    agent.save_model(best_model_path)

                if int(print_every_episodes) > 0 and (int(episode) % int(print_every_episodes) == 0):
                    print(
                        f"Episode {int(episode)}/{int(episodes)} | Reward={float(episode_reward):.3f} | AvgLoss={float(avg_loss):.6f} | Epsilon={float(last_epsilon):.4f}"
                    )
                    if cycle_tracker is not None and (int(episode) % int(log_cycle_every) == 0):
                        print(f"  {cycle_tracker.get_summary_str()}")
                        print(f"  Cycle entropy: {cycle_tracker.get_entropy():.3f}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        print("Environment closed.")

    if cycle_tracker is not None:
        print(f"[Cycle summary] {cycle_tracker.get_summary_str()}")
        print(f"[Cycle summary] entropy={cycle_tracker.get_entropy():.3f}")

    return metrics_path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_hub_spoke_demo.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)

    if args.seed is not None:
        config.setdefault("run", {})
        config["run"]["seed"] = int(args.seed)
    if args.run_name is not None:
        config.setdefault("run", {})
        config["run"]["run_name"] = str(args.run_name)
    if args.episodes is not None:
        config.setdefault("train", {})
        config["train"]["episodes"] = int(args.episodes)
    if args.log_dir is not None or args.model_dir is not None or args.results_dir is not None:
        config.setdefault("logging", {})
        if args.log_dir is not None:
            config["logging"]["log_dir"] = str(args.log_dir)
        if args.model_dir is not None:
            config["logging"]["model_dir"] = str(args.model_dir)
        if args.results_dir is not None:
            config["logging"]["results_dir"] = str(args.results_dir)

    try:
        metrics_path = run_training(config)
        print(f"Training complete. Metrics: {metrics_path}")
    except KeyboardInterrupt:
        print("Training interrupted.")
        sys.exit(1)
    except Exception as exc:
        print(f"Training failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
