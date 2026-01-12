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
    from rl.utils import set_global_seed
    
    parallel_cfg = config.get("parallel", {})
    base_port = int(parallel_cfg.get("base_port", 8813))
    base_seed = int(parallel_cfg.get("base_seed", 42))
    chunk_size = int(parallel_cfg.get("chunk_size", 256))
    epsilon_base = float(parallel_cfg.get("epsilon_base", 0.2))
    epsilon_delta = float(parallel_cfg.get("epsilon_worker_delta", 0.02))
    
    worker_seed = base_seed + worker_id
    worker_port = base_port + worker_id
    epsilon = min(1.0, max(0.0, epsilon_base + epsilon_delta * worker_id))
    
    set_global_seed(worker_seed)
    
    env_config = config.copy()
    env_config.setdefault("env", {}).setdefault("sumo", {})
    env_config["env"]["sumo"]["worker_id"] = worker_id
    env_config["env"]["sumo"]["base_port"] = base_port
    env_config["run"] = env_config.get("run", {}).copy()
    env_config["run"]["seed"] = worker_seed
    
    try:
        env = build_env(env_config)
    except Exception as e:
        stop_event.set()
        return
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    hidden_dims = config.get("agent", {}).get("hidden_dims", [192, 192])
    
    policy_net = DuelingDQN(state_dim, action_dim, hidden_dims)
    policy_net.eval()
    
    local_step = 0
    local_buffer: List[Tuple] = []
    
    try:
        while not stop_event.is_set():
            _drain_and_load_weights(policy_net, weight_queue)
            
            try:
                state = env.reset()
            except Exception:
                time.sleep(1.0)
                continue
            
            done = False
            
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
                except Exception:
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
                
                if len(local_buffer) >= chunk_size:
                    _send_chunk(experience_queue, local_buffer)
                    local_buffer = []
            
            if len(local_buffer) > 0:
                _send_chunk(experience_queue, local_buffer)
                local_buffer = []
                
    except Exception:
        stop_event.set()
    finally:
        try:
            env.close()
        except Exception:
            pass


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


def _send_chunk(queue: Queue, buffer: List[Tuple]) -> None:
    chunk = list(buffer)
    try:
        queue.put_nowait(chunk)
    except Full:
        pass
