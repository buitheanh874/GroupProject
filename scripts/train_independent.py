"""
Independent Multi-Agent Training Script

This script trains 9 independent DQN agents (one per traffic light) with:
- No parameter sharing
- No communication between agents
- Separate Q-networks and replay buffers for each agent
- Each agent only observes its local state

This is a baseline for comparison with the centralized multi-agent approach.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import time
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

if __package__ in (None, ""):
    script_dir = Path(__file__).resolve().parent
    project_root_hint = script_dir.parent
    if str(project_root_hint) not in sys.path:
        sys.path.insert(0, str(project_root_hint))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

from scripts.repo_root import find_repo_root

project_root = find_repo_root(__file__)
sys.path.insert(0, str(project_root))

from rl.utils import ensure_dir, generate_run_id, linear_epsilon, load_yaml_config, save_yaml_config, set_global_seed
from scripts.common import build_agent, build_env
from scripts.route_pool_loader import load_route_pool_from_config, validate_route_file_nonempty
from scripts.scenario_config_bridge import apply_calibration_overrides
from scripts.config_normalization import normalize_action_table_schema


def run_independent_training(config: Dict[str, Any]) -> str:
    """Train 9 independent agents with no communication."""
    
    config = apply_calibration_overrides(config, project_root=project_root)
    config = normalize_action_table_schema(config)
    train_cfg = config.get("train", {})
    
    # Curriculum setup (same as train.py)
    curriculum_cfg = config.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    curriculum_phases = curriculum_cfg.get("phases", [])
    
    if curriculum_enabled and len(curriculum_phases) > 0:
        print("[Curriculum] Enabled with {} phases".format(len(curriculum_phases)))
        for i, phase in enumerate(curriculum_phases):
            print(f"  Phase {i+1}: {phase.get('name', 'unnamed')} - {phase.get('episodes', 0)} episodes")
    else:
        curriculum_enabled = False
        print("[Curriculum] Disabled - using single manifest training")
    
    phase_schedule: List[Dict[str, Any]] = []
    if curriculum_enabled:
        for phase in curriculum_phases:
            phase_schedule.append({
                "name": phase.get("name", "phase"),
                "episodes": phase.get("episodes", 100),
                "manifest": phase.get("route_pool_manifest", ""),
                "max_sim_seconds": phase.get("max_sim_seconds", 1800),
            })
    else:
        default_episodes = int(train_cfg.get("episodes", 200))
        phase_schedule.append({
            "name": "default",
            "episodes": default_episodes,
            "manifest": train_cfg.get("route_pool_manifest", ""),
            "max_sim_seconds": config.get("env", {}).get("sumo", {}).get("max_sim_seconds", 1800),
        })
    
    total_episodes = sum(p["episodes"] for p in phase_schedule)
    
    print("[Independent Multi-Agent] Training 9 independent agents")
    print("[Independent Multi-Agent] No parameter sharing, no communication")
    print(f"[Training] Total episodes: {total_episodes}")
    
    route_pool = load_route_pool_from_config(config, split="train", project_root=project_root)
    sumo_cfg = config.get("env", {}).get("sumo", {})
    route_file = sumo_cfg.get("route_file")
    if not route_pool and route_file:
        validate_route_file_nonempty(Path(route_file))

    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    set_global_seed(seed)

    # Build environment
    env = build_env(config)
    if route_pool and hasattr(env, "set_route_file_pool"):
        try:
            env.set_route_file_pool(route_pool)
        except Exception:
            pass

    # Get TLS IDs
    tls_ids = []
    if hasattr(env, "_tls_ids"):
        tls_ids = env._tls_ids
    else:
        raise ValueError("Environment must have _tls_ids attribute for multi-agent training")
    
    num_agents = len(tls_ids)
    print(f"[Independent Multi-Agent] Number of agents: {num_agents}")
    print(f"[Independent Multi-Agent] TLS IDs: {tls_ids}")
    
    # Create independent agents - one per TLS
    agents = {}
    for tls_id in tls_ids:
        agent, _ = build_agent(config, env)
        agent.to_train_mode()
        agents[tls_id] = agent
        print(f"  Created independent agent for {tls_id}")

    run_name = str(run_cfg.get("run_name", "train_independent"))
    run_id = generate_run_id(prefix=run_name)

    logging_cfg = config.get("logging", {})
    log_dir = ensure_dir(str(logging_cfg.get("log_dir", "logs")))
    model_dir = ensure_dir(str(logging_cfg.get("model_dir", "models")))
    ensure_dir(str(logging_cfg.get("results_dir", "results")))

    config_copy_path = os.path.join(log_dir, f"{run_id}_config.yaml")
    save_yaml_config(config, config_copy_path)

    metrics_path = os.path.join(log_dir, f"{run_id}_train_metrics.csv")

    save_every_episodes = int(train_cfg.get("save_every_episodes", 30))
    print_every_episodes = int(train_cfg.get("print_every_episodes", 5))

    exploration_cfg = config.get("exploration", {})
    eps_start = float(exploration_cfg.get("eps_start", 1.0))
    eps_end = float(exploration_cfg.get("eps_end", 0.05))
    eps_decay_steps = int(exploration_cfg.get("eps_decay_steps", 45000))

    best_reward = -float("inf")
    global_step = 0
    
    current_phase_idx = 0
    phase_episode_count = 0
    current_phase = phase_schedule[0]
    
    def switch_to_phase(phase_idx: int) -> None:
        """Switch to a new curriculum phase by loading its route manifest."""
        nonlocal current_phase, phase_episode_count
        if phase_idx >= len(phase_schedule):
            return
        current_phase = phase_schedule[phase_idx]
        phase_episode_count = 0
        manifest_path = current_phase["manifest"]
        max_sim = current_phase.get("max_sim_seconds", 1800)
        
        # Update max_sim_seconds
        if hasattr(env, "set_max_sim_seconds"):
            env.set_max_sim_seconds(int(max_sim))
        
        if manifest_path and hasattr(env, "set_route_file_pool"):
            try:
                from scripts.route_pool_loader import _load_manifest, _resolve_path
                manifest_full = _resolve_path(manifest_path, project_root, project_root)
                if manifest_full.exists():
                    routes = _load_manifest(manifest_full, project_root)
                    env.set_route_file_pool(routes)
                    print(f"[Curriculum] Phase {phase_idx+1}/{len(phase_schedule)}: {current_phase['name']}")
                    print(f"  Manifest: {manifest_path} ({len(routes)} routes)")
                    print(f"  Episodes: {current_phase['episodes']}, Max sim: {max_sim}s")
            except Exception as e:
                print(f"[Curriculum] Warning: Failed to load manifest for phase {phase_idx}: {e}")
    
    switch_to_phase(0)

    try:
        with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "episode",
                "episode_reward",
                "avg_loss",
                "episode_steps",
                "global_step",
                "num_agents",
                "epsilon_end",
                "arrived_vehicles",
                "avg_wait_time",
                "avg_travel_time",
                "avg_stops",
                "avg_queue",
            ]
            
            # Add per-agent loss columns
            for tls_id in tls_ids:
                fieldnames.append(f"loss_{tls_id}")
            
            if curriculum_enabled:
                fieldnames.append("phase_name")
                fieldnames.append("phase_episode")
            
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for episode in range(1, total_episodes + 1):
                # Handle curriculum phase transitions
                if curriculum_enabled:
                    phase_episode_count += 1
                    if phase_episode_count > current_phase["episodes"] and current_phase_idx < len(phase_schedule) - 1:
                        current_phase_idx += 1
                        switch_to_phase(current_phase_idx)
                        phase_episode_count = 1
                        # Save phase checkpoint for all agents
                        for tls_id in tls_ids:
                            phase_path = os.path.join(model_dir, f"{run_id}_{tls_id}_phase{current_phase_idx}.pt")
                            agents[tls_id].save_checkpoint(phase_path, {
                                "episode": episode,
                                "global_step": global_step,
                                "phase_idx": current_phase_idx,
                                "best_reward": best_reward,
                                "tls_id": tls_id,
                            })
                        print(f"[Curriculum] Saved phase checkpoints for all {num_agents} agents")
                
                if hasattr(env, "set_seed"):
                    env.set_seed(int(seed + episode))

                # Reset environment
                reset_ok = False
                for _retry in range(3):
                    try:
                        state = env.reset()
                        reset_ok = True
                        break
                    except Exception as reset_err:
                        print(f"[WARN] Episode {episode}: reset failed (attempt {_retry+1}/3): {reset_err}")
                        try:
                            env.close()
                        except Exception:
                            pass
                        time.sleep(2)
                
                if not reset_ok:
                    print(f"[ERROR] Episode {episode}: reset failed after 3 attempts, skipping")
                    continue

                if not isinstance(state, dict):
                    raise ValueError("State must be a dictionary for multi-agent training")

                done = False
                episode_reward = 0.0
                episode_steps = 0
                last_epsilon = float(eps_start)

                # Track losses per agent
                losses_per_agent = {tls_id: [] for tls_id in tls_ids}
                info: Dict[str, Any] = {}

                while not done:
                    epsilon = linear_epsilon(
                        global_step=global_step,
                        eps_start=eps_start,
                        eps_end=eps_end,
                        decay_steps=eps_decay_steps,
                    )
                    last_epsilon = float(epsilon)

                    # Each agent selects action independently based on local observation
                    actions: Dict[str, int] = {}
                    for tls_id in tls_ids:
                        local_state = state[tls_id]
                        action = agents[tls_id].select_action(state=local_state, epsilon=epsilon)
                        actions[tls_id] = action

                    # Execute actions in environment
                    try:
                        next_state, rewards, done, info = env.step(actions)
                    except Exception as step_err:
                        print(f"[WARN] Episode {episode}: step failed at step {episode_steps}: {step_err}")
                        done = True
                        next_state = state
                        rewards = {tls_id: 0.0 for tls_id in tls_ids}
                        info = {}
                        try:
                            env.close()
                        except Exception:
                            pass

                    # Compute gamma
                    t_step_value = None
                    if isinstance(info, dict):
                        t_step_value = info.get("t_step") or info.get("decision_cycle_sec")
                        if t_step_value is not None:
                            t_step_value = float(t_step_value)
                    
                    # Each agent stores transition and updates independently
                    for tls_id in tls_ids:
                        agent = agents[tls_id]
                        action_id = actions[tls_id]
                        local_state = state[tls_id]
                        local_next_state = next_state.get(tls_id) if isinstance(next_state, dict) else next_state
                        local_reward = rewards.get(tls_id, 0.0) if isinstance(rewards, dict) else rewards
                        
                        gamma_value = agent.compute_gamma(t_step_value)
                        agent.store_transition(local_state, action_id, local_reward, local_next_state, done, gamma=gamma_value)
                        
                        # Update agent independently
                        metrics = agent.update()
                        if metrics is not None:
                            loss_value = metrics.get('loss', 0.0) if isinstance(metrics, dict) else float(metrics)
                            losses_per_agent[tls_id].append(float(loss_value))
                            if len(losses_per_agent[tls_id]) > 500:
                                losses_per_agent[tls_id].pop(0)

                    # Compute average reward across all agents
                    step_rewards = list(rewards.values()) if isinstance(rewards, dict) else [float(rewards)]
                    step_reward = float(np.mean(step_rewards))

                    state = next_state
                    episode_reward += float(step_reward)
                    episode_steps += 1
                    global_step += num_agents  # Each agent takes one step

                # Compute average loss across all agents
                all_losses = []
                for tls_id in tls_ids:
                    all_losses.extend(losses_per_agent[tls_id])
                avg_loss = float(np.mean(all_losses)) if len(all_losses) > 0 else 0.0

                # Get KPI
                kpi = {}
                if isinstance(info, dict):
                    kpi = info.get("episode_kpi", {}) if done else {}

                # Write metrics
                row: Dict[str, Any] = {
                    "episode": int(episode),
                    "episode_reward": float(episode_reward),
                    "avg_loss": float(avg_loss),
                    "episode_steps": int(episode_steps),
                    "global_step": int(global_step),
                    "num_agents": int(num_agents),
                    "epsilon_end": float(last_epsilon),
                    "arrived_vehicles": int(kpi.get("arrived_vehicles", 0)),
                    "avg_wait_time": float(kpi.get("avg_wait_time", 0.0)),
                    "avg_travel_time": float(kpi.get("avg_travel_time", 0.0)),
                    "avg_stops": float(kpi.get("avg_stops", 0.0)),
                    "avg_queue": float(kpi.get("avg_queue", 0.0)),
                }
                
                # Add per-agent losses
                for tls_id in tls_ids:
                    agent_losses = losses_per_agent[tls_id]
                    row[f"loss_{tls_id}"] = float(np.mean(agent_losses)) if len(agent_losses) > 0 else 0.0
                
                if curriculum_enabled:
                    row["phase_name"] = current_phase["name"]
                    row["phase_episode"] = phase_episode_count

                writer.writerow(row)
                csv_file.flush()

                # Clear losses
                for tls_id in tls_ids:
                    losses_per_agent[tls_id].clear()

                # Save checkpoints
                should_save_periodic = int(save_every_episodes) > 0 and (int(episode) % int(save_every_episodes) == 0)
                is_best = float(episode_reward) > float(best_reward)

                if is_best:
                    best_reward = float(episode_reward)

                if should_save_periodic:
                    # Save all agents
                    for tls_id in tls_ids:
                        model_path = os.path.join(model_dir, f"{run_id}_{tls_id}_episode_{int(episode)}.pt")
                        agents[tls_id].save_checkpoint(model_path, {
                            "episode": episode,
                            "global_step": global_step,
                            "phase_idx": current_phase_idx,
                            "best_reward": best_reward,
                            "tls_id": tls_id,
                        })

                if is_best:
                    # Save best models for all agents
                    for tls_id in tls_ids:
                        best_model_path = os.path.join(model_dir, f"{run_id}_{tls_id}_best.pt")
                        agents[tls_id].save_checkpoint(best_model_path, {
                            "episode": episode,
                            "global_step": global_step,
                            "phase_idx": current_phase_idx,
                            "best_reward": best_reward,
                            "tls_id": tls_id,
                        })

                if int(print_every_episodes) > 0 and (int(episode) % int(print_every_episodes) == 0):
                    phase_info = f" | Phase: {current_phase['name']}" if curriculum_enabled else ""
                    print(
                        f"Episode {int(episode)}/{int(total_episodes)} | Reward={float(episode_reward):.3f} | "
                        f"AvgLoss={float(avg_loss):.6f} | Epsilon={float(last_epsilon):.4f}{phase_info}"
                    )

    except (KeyboardInterrupt, Exception) as e:
        # Save crash checkpoints for all agents
        try:
            for tls_id in tls_ids:
                crash_path = os.path.join(model_dir, f"{run_id}_{tls_id}_crash_ep{episode}.pt")
                agents[tls_id].save_checkpoint(crash_path, {
                    "episode": episode,
                    "global_step": global_step,
                    "phase_idx": current_phase_idx,
                    "best_reward": best_reward,
                    "tls_id": tls_id,
                })
            print(f"\n[Crash Recovery] Saved checkpoints for all {num_agents} agents")
        except Exception as save_err:
            print(f"\n[Crash Recovery] Failed to save checkpoints: {save_err}")
        
        if isinstance(e, KeyboardInterrupt):
            print("\n[Training] Interrupted by user (Ctrl+C)")
        else:
            print(f"\n[Training] Failed with error: {e}")
            raise
    finally:
        try:
            env.close()
        except Exception:
            pass
        print("Environment closed.")

    return metrics_path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train independent multi-agent system (no communication)")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
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
        metrics_path = run_independent_training(config)
        print(f"Training complete. Metrics: {metrics_path}")
    except KeyboardInterrupt:
        print("Training interrupted. Check model_dir for crash checkpoints.")
        sys.exit(1)
    except Exception as exc:
        print(f"Training failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
