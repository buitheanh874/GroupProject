from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

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


def _select_route_for_demand(demand: int, seed: int) -> str:
    """Select deterministic route from manifest for a given demand/seed."""
    manifest_path = project_root / f"networks/variants/train_turn801010/{demand}/manifest.txt"
    if not manifest_path.exists():
        return ""
    try:
        with open(manifest_path, "r") as f:
            routes = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except Exception:
        return ""
    if not routes:
        return ""
    route_file = routes[(int(seed) - 42) % len(routes)]
    full_path = manifest_path.parent / route_file
    return str(full_path) if full_path.exists() else ""


def _append_csv_row(path: str, fieldnames: list[str], row: Dict[str, Any]) -> None:
    """Append a row to CSV, writing header if file is new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_smoke_eval_episode(
    agent,
    base_config: Dict[str, Any],
    demand: int,
    seed: int,
    horizon_sec: int,
    warmup_sec: int,
    log_path: str,
    episode_idx: int,
    phase_name: str,
) -> Dict[str, Any]:
    """Run a quick greedy evaluation with the current agent."""
    from scripts.common import build_env, resolve_allowed_action_ids
    from scripts.route_pool_loader import load_route_pool_from_config
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("env", {}).setdefault("sumo", {})
    cfg.setdefault("run", {})
    cfg["run"]["seed"] = int(seed)
    cfg["env"]["sumo"]["max_sim_seconds"] = int(horizon_sec)
    cfg["env"]["sumo"]["warmup_sec"] = int(warmup_sec)

    route_file = _select_route_for_demand(demand, seed)
    if route_file:
        cfg["env"]["sumo"]["route_file"] = route_file
    else:
        # Fallback: use existing route pool if configured
        try:
            pool = load_route_pool_from_config(cfg, split="train", project_root=project_root)
            if pool:
                cfg["env"]["sumo"]["route_file"] = pool[seed % len(pool)]
        except Exception:
            pass

    env = build_env(cfg)
    agent.to_eval_mode()
    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    last_info: Dict[str, Any] = {}

    while not done:
        allowed_action_ids = resolve_allowed_action_ids(env, target_action=None, fallback_action=None)
        if isinstance(state, dict):
            actions: Dict[str, int] = {}
            for tls_id, tls_state in state.items():
                allowed_ids = None
                if isinstance(allowed_action_ids, dict):
                    allowed_ids = allowed_action_ids.get(tls_id)
                actions[tls_id] = agent.select_action(
                    state=np.asarray(tls_state, dtype=np.float32),
                    epsilon=0.0,
                    allowed_action_ids=allowed_ids,
                )
        else:
            actions = agent.select_action(state=np.asarray(state, dtype=np.float32), epsilon=0.0)

        next_state, rewards, done, info = env.step(actions)
        reward_values = list(rewards.values()) if isinstance(rewards, dict) else [float(rewards)]
        total_reward += float(np.mean(reward_values))
        step_count += 1
        state = next_state
        last_info = info if isinstance(info, dict) else {}

        if step_count >= int(horizon_sec * 2):
            break

    kpi = env.episode_kpi() if hasattr(env, "episode_kpi") else {}
    if not kpi and isinstance(last_info, dict):
        kpi = last_info.get("episode_kpi", {})

    n_present_end = 0
    if hasattr(env, "_traci") and env._traci is not None:
        try:
            n_present_end = int(env._traci.vehicle.getIDCount())
        except Exception:
            n_present_end = 0

    env.close()
    agent.to_train_mode()

    row = {
        "episode": int(episode_idx),
        "phase_name": phase_name,
        "demand": int(demand),
        "seed": int(seed),
        "horizon_sec": int(horizon_sec),
        "avg_wait_time_corr": float(kpi.get("avg_wait_time_corr", 0.0)),
        "throughput_corr": float(kpi.get("throughput_corr", 0.0)),
        "completion_rate": float(kpi.get("completion_rate", 0.0)),
        "teleport_rate": float(kpi.get("teleport_rate", 0.0)),
        "n_present_end": int(n_present_end),
        "route_file": os.path.basename(route_file) if route_file else "",
        "total_reward": float(total_reward),
        "episode_steps": int(step_count),
        "timestamp": time.time(),
    }
    _append_csv_row(
        log_path,
        fieldnames=[
            "episode",
            "phase_name",
            "demand",
            "seed",
            "horizon_sec",
            "avg_wait_time_corr",
            "throughput_corr",
            "completion_rate",
            "teleport_rate",
            "n_present_end",
            "route_file",
            "total_reward",
            "episode_steps",
            "timestamp",
        ],
        row=row,
    )
    return row


from rl.cycle_tracker import CycleDistributionTracker
from rl.utils import ensure_dir, generate_run_id, linear_epsilon, load_yaml_config, save_yaml_config, set_global_seed
from scripts.common import build_agent, build_env, resolve_allowed_action_ids
from scripts.route_pool_loader import load_route_pool_from_config, validate_route_file_nonempty
from scripts.scenario_config_bridge import apply_calibration_overrides
from scripts.config_normalization import normalize_action_table_schema


def run_training(config: Dict[str, Any], resume_path: Optional[str] = None, start_episode_override: Optional[int] = None) -> str:
    config = apply_calibration_overrides(config, project_root=project_root)
    config = normalize_action_table_schema(config)
    train_cfg = config.get("train", {})
    smoke_eval_every = int(train_cfg.get("smoke_eval_every", 0))
    smoke_eval_demand = int(train_cfg.get("smoke_eval_demand", 750))
    smoke_eval_horizon = int(train_cfg.get("smoke_eval_horizon_sec", 750))
    curriculum_hist_every = int(train_cfg.get("curriculum_hist_every", 0))
    
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
    
    phase_schedule = []
    if curriculum_enabled:
        for phase in curriculum_phases:
            phase_schedule.append({
                "name": phase.get("name", "phase"),
                "episodes": phase.get("episodes", 100),
                "manifest": phase.get("route_pool_manifest", ""),
            })
    else:
        default_episodes = int(train_cfg.get("episodes", 200))
        phase_schedule.append({
            "name": "default",
            "episodes": default_episodes,
            "manifest": train_cfg.get("route_pool_manifest", ""),
        })
    
    total_episodes = sum(p["episodes"] for p in phase_schedule)
    print(f"[Training] Total episodes: {total_episodes}")
    
    route_pool = load_route_pool_from_config(config, split="train", project_root=project_root)
    sumo_cfg = config.get("env", {}).get("sumo", {})
    route_file = sumo_cfg.get("route_file")
    if not route_pool and route_file:
        validate_route_file_nonempty(Path(route_file))

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

    start_episode = 1
    resume_global_step = 0
    resume_phase_idx = 0
    resume_best_reward = -float("inf")
    
    if resume_path:
        print(f"[Resume] Loading checkpoint: {resume_path}")
        extra_state = agent.load_checkpoint(resume_path)
        start_episode = extra_state.get("episode", 0) + 1
        resume_global_step = extra_state.get("global_step", 0)
        resume_phase_idx = extra_state.get("phase_idx", 0)
        resume_best_reward = extra_state.get("best_reward", -float("inf"))
        print(f"[Resume] Starting from episode {start_episode}")
        print(f"[Resume] Global step: {resume_global_step}, Phase: {resume_phase_idx}")
        print(f"[Resume] Best reward: {resume_best_reward:.2f}")
    
    # Override start episode if explicitly provided (useful for resuming from parallel)
    if start_episode_override is not None:
        start_episode = start_episode_override
        print(f"[Override] Start episode set to: {start_episode}")

    run_name = str(run_cfg.get("run_name", "train"))
    run_id = generate_run_id(prefix=run_name)

    logging_cfg = config.get("logging", {})
    log_dir = ensure_dir(str(logging_cfg.get("log_dir", "logs")))
    model_dir = ensure_dir(str(logging_cfg.get("model_dir", "models")))
    ensure_dir(str(logging_cfg.get("results_dir", "results")))

    config_copy_path = os.path.join(log_dir, f"{run_id}_config.yaml")

    # [Reproducibility] Try to get git commit hash
    try:
        import subprocess
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        config["meta"] = config.get("meta", {})
        config["meta"]["git_commit"] = git_hash
    except Exception:
        pass
        
    save_yaml_config(config, config_copy_path)

    metrics_path = os.path.join(log_dir, f"{run_id}_train_metrics.csv")
    smoke_eval_log_path = os.path.join(log_dir, f"{run_id}_smoke_eval.csv")
    curriculum_stats_path = os.path.join(log_dir, f"{run_id}_curriculum_stats.jsonl")

    episodes = total_episodes
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
    episode_to_phase: Dict[int, int] = {}
    episode_uid_base = int(train_cfg.get("episode_uid_base", 0))

    best_reward = resume_best_reward
    global_step = resume_global_step
    
    current_phase_idx = resume_phase_idx
    phase_episode_count = 0
    current_phase = phase_schedule[min(current_phase_idx, len(phase_schedule) - 1)]
    
    def switch_to_phase(phase_idx: int) -> None:
        """Switch to a new curriculum phase by loading its route manifest."""
        nonlocal current_phase, phase_episode_count
        if phase_idx >= len(phase_schedule):
            return
        current_phase = phase_schedule[phase_idx]
        phase_episode_count = 0
        manifest_path = current_phase["manifest"]
        if manifest_path and hasattr(env, "set_route_file_pool"):
            try:
                from scripts.route_pool_loader import _load_manifest, _resolve_path
                manifest_full = _resolve_path(manifest_path, project_root, project_root)
                if manifest_full.exists():
                    routes = _load_manifest(manifest_full, project_root)
                    env.set_route_file_pool(routes)
                    print(f"[Curriculum] Phase {phase_idx+1}/{len(phase_schedule)}: {current_phase['name']}")
                    print(f"  Manifest: {manifest_path} ({len(routes)} routes)")
                    print(f"  Episodes: {current_phase['episodes']}")
            except Exception as e:
                print(f"[Curriculum] Warning: Failed to load manifest for phase {phase_idx}: {e}")
    
    switch_to_phase(0)

    try:
        with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "episode",
                "episode_uid",
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
                "decision_steps",
                "waiting_total",
                "env_seed",
                "route_file",
            ]
            
            if len(allowed_cycles) > 0:
                for cycle in allowed_cycles:
                    fieldnames.append(f"cycle_{cycle}_count")
            if cycle_tracker is not None:
                for cycle in allowed_cycles:
                    fieldnames.append(f"cycle_{cycle}_pct")
                fieldnames.append("cycle_entropy")
            
            if curriculum_enabled:
                fieldnames.append("phase_name")
                fieldnames.append("phase_episode")
            
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for episode in range(start_episode, int(episodes) + 1):
                episode_uid = int(episode_uid_base + episode)
                if curriculum_enabled:
                    phase_episode_count += 1
                    if phase_episode_count > current_phase["episodes"] and current_phase_idx < len(phase_schedule) - 1:
                        current_phase_idx += 1
                        switch_to_phase(current_phase_idx)
                        phase_episode_count = 1
                        phase_checkpoint_path = os.path.join(model_dir, f"{run_id}_phase{current_phase_idx}.pt")
                        agent.save_checkpoint(phase_checkpoint_path, {
                            "episode": episode,
                            "global_step": global_step,
                            "phase_idx": current_phase_idx,
                            "best_reward": best_reward,
                        })
                        print(f"[Curriculum] Saved checkpoint: {phase_checkpoint_path}")
                
                if hasattr(env, "set_seed"):
                    env.set_seed(int(seed + episode))

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

                        try:
                            next_state, rewards, done, info = env.step(actions)
                        except Exception as step_err:
                            print(f"[WARN] Episode {episode}: step failed at step {episode_steps}: {step_err}")
                            done = True
                            next_state = state
                            rewards = {tls_id: 0.0 for tls_id in tls_ids_sorted}
                            info = {}
                            try:
                                env.close()
                            except Exception:
                                pass

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
                            agent.store_transition(state[tls_id], action_id, reward_value, next_obs, done, gamma=gamma_value, episode_uid=episode_uid)

                        metrics = agent.update()
                        if metrics is not None:
                            loss_value = metrics.get('loss', 0.0) if isinstance(metrics, dict) else float(metrics)
                            losses.append(float(loss_value))
                            if len(losses) > 500:
                                losses.pop(0)

                        state = next_state
                        episode_reward += float(step_reward)
                        episode_steps += 1
                        global_step += len(actions)
                    else:
                        action_id = agent.select_action(state=state, epsilon=epsilon)
                        try:
                            next_state, reward, done, info = env.step(action_id)
                        except Exception as step_err:
                            print(f"[WARN] Episode {episode}: step failed at step {episode_steps}: {step_err}")
                            done = True
                            next_state = state
                            reward = 0.0
                            info = {}
                            try:
                                env.close()
                            except Exception:
                                pass

                        gamma_value = agent.compute_gamma(info.get("t_step") if isinstance(info, dict) else None)
                        agent.store_transition(state, action_id, reward, next_state, done, gamma=gamma_value, episode_uid=episode_uid)
                        metrics = agent.update()
                        if metrics is not None:
                            loss_value = metrics.get('loss', 0.0) if isinstance(metrics, dict) else float(metrics)
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
                    "episode_uid": int(episode_uid),
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
                    "env_seed": int(seed + episode) if hasattr(env, "set_seed") else 0,
                    "route_file": str(os.path.basename(env._route_file)) if hasattr(env, "_route_file") and env._route_file else "",
                }
                
                if len(allowed_cycles) > 0:
                    for cycle in allowed_cycles:
                        row[f"cycle_{cycle}_count"] = int(episode_cycle_counts.get(cycle, 0))
                if cycle_tracker is not None:
                    cycle_dist = cycle_tracker.get_distribution()
                    for cycle in allowed_cycles:
                        row[f"cycle_{cycle}_pct"] = float(cycle_dist.get(cycle, 0.0))
                    row["cycle_entropy"] = float(cycle_tracker.get_entropy())
                
                if curriculum_enabled:
                    row["phase_name"] = current_phase["name"]
                    row["phase_episode"] = phase_episode_count
                    episode_to_phase[int(episode_uid)] = int(current_phase_idx)
                else:
                    episode_to_phase[int(episode_uid)] = 0

                writer.writerow(row)
                csv_file.flush()

                if curriculum_hist_every > 0 and (episode % curriculum_hist_every == 0) and len(agent.replay_buffer) > 0:
                    buffer_hist = agent.replay_buffer.get_phase_histogram(episode_to_phase)
                    sampled_hist: Dict[int, int] = {}
                    try:
                        sample_size = min(256, len(agent.replay_buffer))
                        batch = agent.replay_buffer.sample(batch_size=sample_size, device=torch.device("cpu"))
                        ep_uids = batch.episode_uids.cpu().numpy().reshape(-1)
                        for uid in ep_uids:
                            phase_val = episode_to_phase.get(int(uid), -1)
                            sampled_hist[phase_val] = sampled_hist.get(phase_val, 0) + 1
                    except Exception as hist_err:
                        sampled_hist = {"error": str(hist_err)}
                    hist_entry = {
                        "timestamp": time.time(),
                        "episode": int(episode),
                        "episode_uid": int(episode_uid),
                        "global_step": int(global_step),
                        "buffer_size": len(agent.replay_buffer),
                        "buffer_phase_histogram": buffer_hist,
                        "sampled_batch_phase_histogram": sampled_hist,
                        "phase_idx": int(current_phase_idx),
                        "phase_name": current_phase.get("name", "default"),
                    }
                    with open(curriculum_stats_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(hist_entry) + "\n")

                losses.clear()
                del row

                if smoke_eval_every > 0 and (episode % smoke_eval_every == 0):
                    warmup_eval = int(config.get("env", {}).get("sumo", {}).get("warmup_sec", 300))
                    try:
                        run_smoke_eval_episode(
                            agent=agent,
                            base_config=config,
                            demand=smoke_eval_demand,
                            seed=int(seed + episode + 999),
                            horizon_sec=smoke_eval_horizon,
                            warmup_sec=warmup_eval,
                            log_path=smoke_eval_log_path,
                            episode_idx=int(episode),
                            phase_name=current_phase.get("name", "default"),
                        )
                    except Exception as smoke_err:
                        print(f"[SmokeEval] Failed at episode {episode}: {smoke_err}")

                should_save_periodic = int(save_every_episodes) > 0 and (int(episode) % int(save_every_episodes) == 0)
                is_best = float(episode_reward) > float(best_reward)

                if is_best:
                    best_reward = float(episode_reward)

                if should_save_periodic:
                    model_path = os.path.join(model_dir, f"{run_id}_episode_{int(episode)}.pt")
                    agent.save_checkpoint(model_path, {
                        "episode": episode,
                        "global_step": global_step,
                        "phase_idx": current_phase_idx,
                        "best_reward": best_reward,
                    })

                if is_best:
                    best_model_path = os.path.join(model_dir, f"{run_id}_best.pt")
                    agent.save_checkpoint(best_model_path, {
                        "episode": episode,
                        "global_step": global_step,
                        "phase_idx": current_phase_idx,
                        "best_reward": best_reward,
                    })

                if int(print_every_episodes) > 0 and (int(episode) % int(print_every_episodes) == 0):
                    print(
                        f"Episode {int(episode)}/{int(episodes)} | Reward={float(episode_reward):.3f} | AvgLoss={float(avg_loss):.6f} | Epsilon={float(last_epsilon):.4f}"
                    )
                    if cycle_tracker is not None and (int(episode) % int(log_cycle_every) == 0):
                        print(f"  {cycle_tracker.get_summary_str()}")
                        print(f"  Cycle entropy: {cycle_tracker.get_entropy():.3f}")
    except (KeyboardInterrupt, Exception) as e:

        try:
            crash_path = os.path.join(model_dir, f"{run_id}_crash_ep{episode}.pt")
            agent.save_checkpoint(crash_path, {
                "episode": episode,
                "global_step": global_step,
                "phase_idx": current_phase_idx,
                "best_reward": best_reward,
            })
            print(f"\n[Crash Recovery] Saved checkpoint: {crash_path}")
            print(f"[Crash Recovery] Resume with: python scripts/train.py --config <config> --resume {crash_path}")
        except Exception as save_err:
            print(f"\n[Crash Recovery] Failed to save checkpoint: {save_err}")
        
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

    if cycle_tracker is not None:
        print(f"[Cycle summary] {cycle_tracker.get_summary_str()}")
        print(f"[Cycle summary] entropy={cycle_tracker.get_entropy():.3f}")

    return metrics_path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--start-episode", type=int, default=None,
                        help="Override starting episode number (for resuming from parallel)")
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
        metrics_path = run_training(config, resume_path=args.resume, start_episode_override=args.start_episode)
        print(f"Training complete. Metrics: {metrics_path}")
    except KeyboardInterrupt:
        print("Training interrupted. Check model_dir for crash checkpoint.")
        sys.exit(1)
    except Exception as exc:
        print(f"Training failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
