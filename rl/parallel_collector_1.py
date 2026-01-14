from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from multiprocessing import Queue, Event
from queue import Empty, Full
import time

import numpy as np
import torch
torch.set_num_threads(1)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from rl.dueling_dqn import DuelingDQN


def collector_process(
    worker_id: int,
    config: Dict[str, Any],
    experience_queue: Queue,
    weight_queue: Queue,
    stop_event: Event,
) -> None:
    from scripts.common import build_env
    from scripts.route_pool_loader import load_route_pool_from_config
    from rl.utils import set_global_seed
    
    import random
    
    parallel_cfg = config.get("parallel", {})
    base_port = int(parallel_cfg.get("base_port", 8813))
    base_seed = int(parallel_cfg.get("base_seed", 42))
    chunk_size = int(parallel_cfg.get("chunk_size", 256))
    epsilon_base = float(parallel_cfg.get("epsilon_base", 0.2))
    epsilon_delta = float(parallel_cfg.get("epsilon_worker_delta", 0.02))
    reset_max_retries = int(parallel_cfg.get("reset_max_retries", 3))
    reset_backoff_base_sec = float(parallel_cfg.get("reset_backoff_base_sec", 1.0))
    reset_backoff_cap_sec = float(parallel_cfg.get("reset_backoff_cap_sec", 8.0))
    
    sumo_cfg = config.get("env", {}).get("sumo", {})
    tls_ids = sumo_cfg.get("tls_ids", [])
    num_tls = len(tls_ids) if tls_ids else 1
    
    worker_seed = base_seed + worker_id
    worker_port = base_port + worker_id
    epsilon = min(1.0, max(0.0, epsilon_base + epsilon_delta * worker_id))
    
    set_global_seed(worker_seed)
    env_config = config.copy()
    env_config.setdefault("env", {}).setdefault("sumo", {})
    env_config["env"]["sumo"]["worker_id"] = worker_id
    env_config["env"]["sumo"]["base_port"] = worker_port
    env_config["run"] = env_config.get("run", {}).copy()
    env_config["run"]["seed"] = worker_seed
    curriculum_cfg = config.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    phases = curriculum_cfg.get("phases", [])
    
    phase_info = []
    if curriculum_enabled and phases:
        for i, phase in enumerate(phases):
            phase_info.append({
                "name": phase.get("name", f"phase{i}"),
                "episodes": phase.get("episodes", 100),
                "manifest": phase.get("route_pool_manifest", ""),
                "max_sim_seconds": phase.get("max_sim_seconds", 3600),  
            })
        print(f"[Worker {worker_id}] Curriculum enabled: {len(phases)} phases")
    else:
        print(f"[Worker {worker_id}] Curriculum disabled, using single route file")
    
    try:
        env = build_env(env_config)
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to build env: {e}")
        stop_event.set()
        return

    current_phase_idx = 0
    route_pool = None
    
    if curriculum_enabled and phase_info:
        phase = phase_info[0]
        try:
            temp_config = env_config.copy()
            temp_config.setdefault("train", {})["route_pool_manifest"] = phase["manifest"]
            route_pool = load_route_pool_from_config(temp_config, split="train", project_root=project_root)
            if route_pool and hasattr(env, "set_route_file_pool"):
                env.set_route_file_pool(route_pool)
            if hasattr(env, "set_max_sim_seconds"):
                env.set_max_sim_seconds(phase["max_sim_seconds"])
            print(f"[Worker {worker_id}] Phase 0 ({phase['name']}): {len(route_pool)} routes, max_sim={phase['max_sim_seconds']}s")
        except Exception as e:
            print(f"[Worker {worker_id}] Failed to load route pool: {e}")
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    hidden_dims = config.get("agent", {}).get("hidden_dims", [192, 192])
    
    policy_net = DuelingDQN(state_dim, action_dim, hidden_dims)
    policy_net.eval()
    
    local_step = 0
    local_buffer: List[Tuple] = []
    global_steps_in_buffer = 0
    episode_count = 0
    phase_episode_count = 0
    reset_fail_count = 0
    step_error_count = 0

    if curriculum_enabled and phase_info:
        total_episodes = sum(p["episodes"] for p in phase_info)
        num_workers = int(parallel_cfg.get("num_actors", 2))
        worker_episodes_per_phase = [max(1, p["episodes"] // num_workers) for p in phase_info]

        resume_episode = int(parallel_cfg.get("resume_episode", 0))
        if resume_episode > 0:
            episode_count = resume_episode
            cumulative = 0
            for idx, p in enumerate(phase_info):
                phase_eps = worker_episodes_per_phase[idx]
                if cumulative + phase_eps > resume_episode:
                    current_phase_idx = idx
                    phase_episode_count = resume_episode - cumulative
                    break
                cumulative += phase_eps
            else:
                current_phase_idx = len(phase_info) - 1
                phase_episode_count = 0
            
            if current_phase_idx > 0:
                phase = phase_info[current_phase_idx]
                try:
                    temp_config = env_config.copy()
                    temp_config.setdefault("train", {})["route_pool_manifest"] = phase["manifest"]
                    route_pool = load_route_pool_from_config(temp_config, split="train", project_root=project_root)
                    if route_pool and hasattr(env, "set_route_file_pool"):
                        env.set_route_file_pool(route_pool)
                    if hasattr(env, "set_max_sim_seconds"):
                        env.set_max_sim_seconds(phase["max_sim_seconds"])
                except Exception as e:
                    print(f"[Worker {worker_id}] Failed to load resume phase route pool: {e}")
            
            print(f"[Worker {worker_id}] Resuming from Episode {resume_episode}, Phase {current_phase_idx} ({phase_info[current_phase_idx]['name']}), Ep in phase: {phase_episode_count}")
    
    try:
        while not stop_event.is_set():
            _drain_and_load_weights(policy_net, weight_queue)
            
            state = None
            for attempt in range(reset_max_retries):
                try:
                    state = env.reset()
                    break
                except Exception as e:
                    print(f"[Worker {worker_id}] Reset attempt {attempt+1}/{reset_max_retries} failed: {e}")
                    try:
                        env.close()
                    except Exception:
                        pass
                    if attempt < reset_max_retries - 1:
                        backoff = min(reset_backoff_cap_sec, reset_backoff_base_sec * (2 ** attempt))
                        jitter = random.uniform(0, backoff)
                        time.sleep(jitter)
                    else:
                        reset_fail_count += 1
            
            if state is None:
                print(f"[Worker {worker_id}] All reset attempts failed, retrying...")
                continue
            
            if curriculum_enabled and phase_info:
                phase_name = phase_info[current_phase_idx]["name"]
                print(f"[Worker {worker_id}] Episode {episode_count} | Phase {current_phase_idx} ({phase_name}) | Ep in phase: {phase_episode_count}")
            else:
                print(f"[Worker {worker_id}] Episode {episode_count}")
            
            done = False
            episode_reward = 0.0
            episode_steps = 0
            
            while not done and not stop_event.is_set():
                if isinstance(state, dict):
                    first_key = list(state.keys())[0]
                    first_state = state[first_key]
                    action_id = _select_action(policy_net, first_state, action_dim, epsilon)
                    actions = {tls_id: action_id for tls_id in state.keys()}
                else:
                    action_id = _select_action(policy_net, state, action_dim, epsilon)
                    actions = action_id
                
                try:
                    next_state, reward, done, info = env.step(actions)
                except Exception as e:
                    print(f"[Worker {worker_id}] Step error: {e}")
                    step_error_count += 1
                    try:
                        env.close()
                    except Exception:
                        pass
                    episode_count += 1
                    break

                if isinstance(reward, dict):
                    episode_reward += sum(reward.values())
                else:
                    episode_reward += reward
                episode_steps += 1
                
                if isinstance(state, dict):
                    for tls_id in state.keys():
                        transition = (
                            state[tls_id].copy(),
                            actions[tls_id],
                            reward.get(tls_id, 0.0),
                            next_state[tls_id].copy(),
                            done,
                        )
                        local_buffer.append(transition)
                    global_steps_in_buffer += 1
                else:
                    transition = (state.copy(), actions, reward, next_state.copy(), done)
                    local_buffer.append(transition)
                    global_steps_in_buffer += 1
                
                state = next_state
                local_step += 1
                
                if len(local_buffer) >= chunk_size:
                    _send_chunk_with_metadata(experience_queue, local_buffer, global_steps_in_buffer)
                    local_buffer = []
                    global_steps_in_buffer = 0
            
            if len(local_buffer) > 0:
                _send_chunk_with_metadata(experience_queue, local_buffer, global_steps_in_buffer)
                local_buffer = []
                global_steps_in_buffer = 0
            
            episode_count += 1
            phase_episode_count += 1
            print(f"[Worker {worker_id}] Episode {episode_count} done | Steps: {episode_steps} | Reward: {episode_reward:.2f}")

            if curriculum_enabled and phase_info and current_phase_idx < len(phase_info) - 1:
                target_eps = worker_episodes_per_phase[current_phase_idx]
                if phase_episode_count >= target_eps:
                    current_phase_idx += 1
                    phase_episode_count = 0
                    phase = phase_info[current_phase_idx]

                    try:
                        temp_config = env_config.copy()
                        temp_config.setdefault("train", {})["route_pool_manifest"] = phase["manifest"]
                        route_pool = load_route_pool_from_config(temp_config, split="train", project_root=project_root)
                        if route_pool and hasattr(env, "set_route_file_pool"):
                            env.set_route_file_pool(route_pool)
                        if hasattr(env, "set_max_sim_seconds"):
                            env.set_max_sim_seconds(phase["max_sim_seconds"])
                        print(f"[Worker {worker_id}] === Switched to Phase {current_phase_idx} ({phase['name']}): {len(route_pool) if route_pool else 0} routes, max_sim={phase['max_sim_seconds']}s ===")
                    except Exception as e:
                        print(f"[Worker {worker_id}] Failed to switch phase: {e}")
                
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        stop_event.set()
    finally:
        try:
            env.close()
        except Exception:
            pass
        print(f"[Worker {worker_id}] Finished. Total episodes: {episode_count}")


def _select_action(policy_net: DuelingDQN, state: np.ndarray, action_dim: int, epsilon: float) -> int:
    if np.random.random() < epsilon:
        return np.random.randint(action_dim)
    
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())


def _drain_and_load_weights(policy_net: DuelingDQN, weight_queue: Queue) -> None:
    latest_weights = None
    
    while True:
        try:
            weights = weight_queue.get_nowait()
            latest_weights = weights
        except Empty:
            break
    
    if latest_weights is not None:
        policy_net.load_state_dict(latest_weights)


def _send_chunk_with_metadata(queue: Queue, buffer: List[Tuple], global_steps: int) -> None:
    chunk_data = {
        "transitions": list(buffer),
        "global_steps": global_steps,
    }
    try:
        queue.put_nowait(chunk_data)
    except Full:
        pass
