"""
Optimized parallel collector with performance instrumentation.

Changes from original parallel_collector_1.py:
1. IntervalLogger: Log every N seconds instead of per-step
2. Worker0 verbose: Only worker rank=0 logs details
3. TransitionCounters: Transition-level no-drop accounting
4. TimingBreakdown: Instrumentation for bottleneck analysis
5. Sentinel-based shutdown: Clean termination protocol
6. Packed transitions: Numpy array serialization (optional)

All changes are SAFE - do not affect MDP semantics or algorithm.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from multiprocessing import Queue, Event, Value
from queue import Empty, Full
import time

import numpy as np
import torch
torch.set_num_threads(1)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from rl.dueling_dqn import DuelingDQN
from rl.perf_utils import (
    PerformanceConfig,
    TransitionCounters,
    TimingBreakdown,
    IntervalLogger,
    SENTINEL_CHUNK,
    is_sentinel,
)


def compute_epsilon(
    global_step: int,
    eps_start: float,
    eps_end: float,
    warmup_steps: int,
    decay_steps: int,
    worker_multiplier: float = 1.0,
) -> float:
    """Compute epsilon with warmup, linear decay, and per-worker multiplier."""
    if global_step < warmup_steps:
        eps_base = eps_start
    else:
        progress = min(1.0, (global_step - warmup_steps) / max(1, decay_steps))
        eps_base = eps_start - progress * (eps_start - eps_end)
    
    eps_worker = eps_base * worker_multiplier
    return max(0.0, min(1.0, eps_worker))


def collector_process_optimized(
    worker_id: int,
    config: Dict[str, Any],
    experience_queue: Queue,
    weight_queue: Queue,
    stop_event: Event,
    global_step_counter: Value = None,
    counters_produced: Value = None,  # Shared counter for produced_transitions
) -> None:
    """
    Optimized collector process with performance instrumentation.
    
    Key changes:
    - IntervalLogger for reduced I/O
    - Worker0 verbose only
    - Transition-level counters
    - Timing breakdown
    - Sentinel-based shutdown
    """
    from scripts.common import build_env
    from scripts.route_pool_loader import load_route_pool_from_config
    from rl.utils import set_global_seed
    import random
    
    # Load performance config
    perf_cfg_dict = config.get("performance", {})
    perf_config = PerformanceConfig.from_dict(perf_cfg_dict)
    
    # Determine verbosity
    is_verbose = (worker_id == 0) or not perf_config.is_enabled("worker0_verbose_only")
    
    # Initialize logging
    interval_sec = perf_config.interval_logging_sec if perf_config.enable_all_optimizations else 0.0
    interval_logger = IntervalLogger(interval_sec=interval_sec, enabled=True)
    
    # Initialize instrumentation
    timing = TimingBreakdown()
    local_counters = TransitionCounters()
    
    def log_msg(msg: str, force: bool = False) -> None:
        """Log message respecting verbosity and interval."""
        if force or (is_verbose and interval_logger.should_log()):
            print(msg)
    
    parallel_cfg = config.get("parallel", {})
    exploration_cfg = config.get("exploration", {})
    
    base_port = int(parallel_cfg.get("base_port", 8813))
    base_seed = int(parallel_cfg.get("base_seed", 42))
    chunk_size = int(parallel_cfg.get("chunk_size", 256))
    reset_max_retries = int(parallel_cfg.get("reset_max_retries", 3))
    reset_backoff_base_sec = float(parallel_cfg.get("reset_backoff_base_sec", 1.0))
    reset_backoff_cap_sec = float(parallel_cfg.get("reset_backoff_cap_sec", 8.0))
    
    # Epsilon schedule parameters
    eps_start = float(exploration_cfg.get("eps_start", 0.60))
    eps_end = float(exploration_cfg.get("eps_end", 0.05))
    warmup_steps = int(exploration_cfg.get("warmup_global_steps", 8000))
    decay_steps = int(exploration_cfg.get("eps_decay_steps", 60000))
    
    # Per-worker diversity multipliers
    worker_multipliers = parallel_cfg.get("epsilon_worker_multipliers", [0.85, 0.95, 1.05, 1.15])
    if isinstance(worker_multipliers, list) and len(worker_multipliers) > 0:
        worker_multiplier = worker_multipliers[worker_id % len(worker_multipliers)]
    else:
        worker_multiplier = 1.0
    
    sumo_cfg = config.get("env", {}).get("sumo", {})
    tls_ids = sumo_cfg.get("tls_ids", [])
    num_tls = len(tls_ids) if tls_ids else 1
    
    worker_seed = base_seed + worker_id
    worker_port = base_port + worker_id
    
    log_msg(f"[Worker {worker_id}] Starting (verbose={is_verbose}, interval={interval_sec}s)", force=True)
    log_msg(f"[Worker {worker_id}] Epsilon: start={eps_start:.2f}, end={eps_end:.2f}, multiplier={worker_multiplier:.2f}", force=True)
    
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
        log_msg(f"[Worker {worker_id}] Curriculum: {len(phases)} phases", force=True)
    
    try:
        timing.start("env_build")
        env = build_env(env_config)
        timing.stop("env_build")
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
            log_msg(f"[Worker {worker_id}] Phase 0 ({phase['name']}): {len(route_pool)} routes", force=True)
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
    local_episode_counter = 0
    current_episode_uid = worker_id * 1_000_000 + local_episode_counter
    
    # Logging accumulators
    cumulative_worker_steps = 0
    episode_start_time = time.time()

    # Each worker runs FULL episode count (same as parallel_collector_1.py)
    # NOT divided among workers
    if curriculum_enabled and phase_info:
        worker_episodes_per_phase = [p["episodes"] for p in phase_info]
        worker_total_episodes = sum(worker_episodes_per_phase)
        log_msg(f"[Worker {worker_id}] Assigned {worker_total_episodes} episodes (phases: {worker_episodes_per_phase})", force=True)
    else:
        worker_episodes_per_phase = []

    try:
        while not stop_event.is_set():
            _drain_and_load_weights(policy_net, weight_queue)
            
            # Reset with timing
            timing.start("env_reset")
            state = None
            for attempt in range(reset_max_retries):
                try:
                    state = env.reset()
                    break
                except Exception as e:
                    log_msg(f"[Worker {worker_id}] Reset attempt {attempt+1} failed: {e}")
                    try:
                        env.close()
                    except Exception:
                        pass
                    if attempt < reset_max_retries - 1:
                        backoff = min(reset_backoff_cap_sec, reset_backoff_base_sec * (2 ** attempt))
                        jitter = random.uniform(0, backoff)
                        time.sleep(jitter)
            timing.stop("env_reset")
            
            if state is None:
                log_msg(f"[Worker {worker_id}] All reset attempts failed")
                continue
            
            if is_verbose and curriculum_enabled and phase_info:
                log_msg(f"[Worker {worker_id}] Ep {episode_count} | Phase {current_phase_idx}")
            
            done = False
            episode_reward = 0.0
            episode_steps = 0
            episode_start_time = time.time()
            
            while not done and not stop_event.is_set():
                # Get global step
                if global_step_counter is not None:
                    current_global_step = global_step_counter.value
                else:
                    current_global_step = local_step
                
                # Compute epsilon
                epsilon = compute_epsilon(
                    global_step=current_global_step,
                    eps_start=eps_start,
                    eps_end=eps_end,
                    warmup_steps=warmup_steps,
                    decay_steps=decay_steps,
                    worker_multiplier=worker_multiplier,
                )
                
                # Action selection with timing
                timing.start("action_select")
                if isinstance(state, dict):
                    first_key = list(state.keys())[0]
                    first_state = state[first_key]
                    action_id = _select_action(policy_net, first_state, action_dim, epsilon)
                    actions = {tls_id: action_id for tls_id in state.keys()}
                else:
                    action_id = _select_action(policy_net, state, action_dim, epsilon)
                    actions = action_id
                timing.stop("action_select")
                
                # Environment step with timing
                timing.start("env_step")
                try:
                    next_state, reward, done, info = env.step(actions)
                except Exception as e:
                    print(f"[Worker {worker_id}] Step error: {e}")
                    try:
                        env.close()
                    except Exception:
                        pass
                    episode_count += 1
                    break
                timing.stop("env_step")

                if isinstance(reward, dict):
                    episode_reward += sum(reward.values())
                else:
                    episode_reward += reward
                episode_steps += 1
                
                # Build transitions with timing
                timing.start("transition_build")
                if isinstance(state, dict):
                    for tls_id in state.keys():
                        transition = (
                            state[tls_id].copy(),
                            actions[tls_id],
                            reward.get(tls_id, 0.0),
                            next_state[tls_id].copy(),
                            done,
                            current_episode_uid,
                        )
                        local_buffer.append(transition)
                    global_steps_in_buffer += 1
                else:
                    transition = (state.copy(), actions, reward, next_state.copy(), done, current_episode_uid)
                    local_buffer.append(transition)
                    global_steps_in_buffer += 1
                timing.stop("transition_build")
                
                # Increment shared global counter
                if global_step_counter is not None:
                    with global_step_counter.get_lock():
                        global_step_counter.value += 1
                
                state = next_state
                local_step += 1
                
                # Send chunk if buffer full
                if len(local_buffer) >= chunk_size:
                    timing.start("queue_put")
                    _send_chunk_blocking(
                        experience_queue, local_buffer, global_steps_in_buffer,
                        phase_idx=current_phase_idx,
                        episode_uid=current_episode_uid,
                        counters=local_counters,
                        counters_shared=counters_produced,
                    )
                    timing.stop("queue_put")
                    local_buffer = []
                    global_steps_in_buffer = 0
            
            # Send remaining transitions
            if len(local_buffer) > 0:
                timing.start("queue_put")
                _send_chunk_blocking(
                    experience_queue, local_buffer, global_steps_in_buffer,
                    phase_idx=current_phase_idx,
                    episode_uid=current_episode_uid,
                    counters=local_counters,
                    counters_shared=counters_produced,
                )
                timing.stop("queue_put")
                local_buffer = []
                global_steps_in_buffer = 0
            
            episode_count += 1
            phase_episode_count += 1
            local_episode_counter += 1
            current_episode_uid = worker_id * 1_000_000 + local_episode_counter
            cumulative_worker_steps += episode_steps
            
            # Interval logging
            if is_verbose and interval_logger.should_log():
                ep_time = time.time() - episode_start_time
                log_msg(f"[Worker {worker_id}] Ep {episode_count} | steps={episode_steps} | "
                        f"reward={episode_reward:.2f} | ε={epsilon:.3f} | "
                        f"produced={local_counters.produced_transitions}")
            
            # Phase transition
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
                        log_msg(f"[Worker {worker_id}] === Phase {current_phase_idx} ({phase['name']}) ===", force=True)
                    except Exception as e:
                        print(f"[Worker {worker_id}] Failed to switch phase: {e}")
            
            # Check completion
            elif curriculum_enabled and phase_info and current_phase_idx == len(phase_info) - 1:
                target_eps = worker_episodes_per_phase[current_phase_idx]
                if phase_episode_count >= target_eps:
                    log_msg(f"[Worker {worker_id}] Completed all {episode_count} episodes", force=True)
                    break
                
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        stop_event.set()
    finally:
        # Send sentinel for clean shutdown
        try:
            sentinel = SENTINEL_CHUNK(worker_id)
            experience_queue.put(sentinel, block=True, timeout=5.0)
            log_msg(f"[Worker {worker_id}] Sent sentinel", force=True)
        except Exception as e:
            print(f"[Worker {worker_id}] Failed to send sentinel: {e}")
        
        try:
            env.close()
        except Exception:
            pass
        
        # Final summary
        log_msg(f"[Worker {worker_id}] === SUMMARY ===", force=True)
        log_msg(f"[Worker {worker_id}]   Episodes: {episode_count}", force=True)
        log_msg(f"[Worker {worker_id}]   Transitions: {local_counters.produced_transitions}", force=True)
        log_msg(f"[Worker {worker_id}]   Timing: {timing.summary_str()}", force=True)


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


def _send_chunk_blocking(
    queue: Queue, 
    buffer: List[Tuple], 
    global_steps: int,
    phase_idx: int = -1,
    episode_uid: int = -1,
    counters: Optional[TransitionCounters] = None,
    counters_shared: Optional[Value] = None,
) -> None:
    """
    Send chunk with BLOCKING put (no drop) and update counters.
    
    Counter increment happens AFTER successful put.
    """
    transition_count = len(buffer)
    chunk_data = {
        "transitions": list(buffer),
        "global_steps": global_steps,
        "phase_idx": phase_idx,
        "episode_uid": episode_uid,
        "count": transition_count,
    }
    
    # Blocking put - will wait if queue is full
    queue.put(chunk_data, block=True)
    
    # Update counters AFTER successful put
    if counters is not None:
        counters.record_produced(transition_count)
    
    if counters_shared is not None:
        with counters_shared.get_lock():
            counters_shared.value += transition_count
