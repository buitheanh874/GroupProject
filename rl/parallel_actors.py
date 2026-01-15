from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Queue, Event
from queue import Empty, Full
import time

import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from rl.dueling_dqn import DuelingDQN


def actor_process(
    worker_id: int,
    config: Dict[str, Any],
    experience_queue: Queue,
    weight_queue: Queue,
    stop_event: Event,
    base_port: int = 8800,
    chunk_size: int = 128,
) -> None:
    from scripts.common import build_env, build_agent
    from rl.utils import set_global_seed
    
    seed = config.get("run", {}).get("seed", 42) + worker_id * 1000
    set_global_seed(seed)
    
    env_config = config.copy()
    env_config.setdefault("env", {}).setdefault("sumo", {})
    env_config["env"]["sumo"]["worker_id"] = worker_id
    env_config["env"]["sumo"]["base_port"] = base_port
    
    env = build_env(env_config)
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    hidden_dims = config.get("agent", {}).get("hidden_dims", [192, 192])
    
    policy_net = DuelingDQN(state_dim, action_dim, hidden_dims)
    policy_net.eval()
    
    epsilon_start = config.get("exploration", {}).get("eps_start", 1.0)
    epsilon_end = config.get("exploration", {}).get("eps_end", 0.05)
    epsilon_decay_steps = config.get("exploration", {}).get("eps_decay_steps", 50000)
    
    local_step = 0
    weight_version = -1
    local_buffer: List[Tuple] = []
    
    try:
        while not stop_event.is_set():
            _sync_weights(policy_net, weight_queue, weight_version)
            
            try:
                state = env.reset()
            except Exception as e:
                print(f"[Actor {worker_id}] Reset failed: {e}")
                time.sleep(1.0)
                continue
            
            done = False
            episode_step = 0
            
            while not done and not stop_event.is_set():
                epsilon = _compute_epsilon(
                    local_step, epsilon_start, epsilon_end, epsilon_decay_steps
                )
                
                if isinstance(state, dict):
                    actions = {}
                    for tls_id, s in state.items():
                        actions[tls_id] = _select_action(policy_net, s, action_dim, epsilon)
                else:
                    actions = _select_action(policy_net, state, action_dim, epsilon)
                
                try:
                    next_state, reward, done, info = env.step(actions)
                except Exception as e:
                    print(f"[Actor {worker_id}] Step failed: {e}")
                    break
                
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
                else:
                    transition = (state.copy(), actions, reward, next_state.copy(), done)
                    local_buffer.append(transition)
                
                state = next_state
                local_step += 1
                episode_step += 1
                
                if len(local_buffer) >= chunk_size:
                    _send_chunk(experience_queue, local_buffer, worker_id)
                    local_buffer = []
                
                if episode_step % 10 == 0:
                    weight_version = _sync_weights(policy_net, weight_queue, weight_version)
            
            if len(local_buffer) > 0:
                _send_chunk(experience_queue, local_buffer, worker_id)
                local_buffer = []
                
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        print(f"[Actor {worker_id}] Stopped after {local_step} steps")


def _select_action(
    policy_net: DuelingDQN, 
    state: np.ndarray, 
    action_dim: int, 
    epsilon: float
) -> int:
    if np.random.random() < epsilon:
        return np.random.randint(action_dim)
    
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())


def _compute_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    progress = min(1.0, float(step) / float(decay_steps))
    return start + (end - start) * progress


def _sync_weights(
    policy_net: DuelingDQN, 
    weight_queue: Queue, 
    current_version: int
) -> int:
    latest_weights = None
    latest_version = current_version
    
    while True:
        try:
            weights, version = weight_queue.get_nowait()
            if version > latest_version:
                latest_weights = weights
                latest_version = version
        except Empty:
            break
    
    if latest_weights is not None:
        policy_net.load_state_dict(latest_weights)
    
    return latest_version


def _send_chunk(queue: Queue, buffer: List[Tuple], worker_id: int) -> None:
    chunk = list(buffer)
    try:
        queue.put_nowait(chunk)
    except Full:
        pass
