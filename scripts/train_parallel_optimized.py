"""
Optimized train_parallel with performance instrumentation.

Key changes from train_parallel.py:
1. Sentinel-based shutdown (no empty() reliance)
2. Transition-level counters (produced_transitions == consumed_transitions)
3. push_batch for replay buffer (optional)
4. Timing breakdown instrumentation
5. Failure-safe termination on exception

All changes are SAFE - do not affect MDP semantics or algorithm.
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import multiprocessing as mp
from multiprocessing import Queue, Event, Process, Value
from pathlib import Path
from typing import Any, Dict, List
from queue import Empty, Full
import time

import numpy as np
import torch

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from rl.utils import load_yaml_config
from rl.perf_utils import (
    PerformanceConfig,
    TransitionCounters,
    TimingBreakdown,
    IntervalLogger,
    is_sentinel,
    compute_throughput,
)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Optimized parallel training with performance instrumentation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-episode", type=int, default=0)
    parser.add_argument("--episode-offset", type=int, default=0)
    args = parser.parse_args(argv)
    
    config = load_yaml_config(args.config)
    parallel_cfg = config.get("parallel", {})
    perf_cfg = config.get("performance", {})
    
    perf_config = PerformanceConfig.from_dict(perf_cfg)
    
    if args.resume_episode > 0:
        parallel_cfg["resume_episode"] = args.resume_episode
    if args.episode_offset > 0:
        parallel_cfg["episode_offset"] = args.episode_offset
    config["parallel"] = parallel_cfg
    
    if not parallel_cfg.get("enabled", False):
        print("parallel.enabled is false. Use scripts/train.py for standard training.")
        return 0
    
    num_actors = int(parallel_cfg.get("num_actors", 4))
    queue_max_chunks = int(parallel_cfg.get("queue_max_chunks", 200))
    sync_every_updates = int(parallel_cfg.get("sync_every_updates", 100))
    
    # Use performance config for queue size if optimizations enabled
    queue_maxsize = perf_config.queue_maxsize if perf_config.enable_all_optimizations else queue_max_chunks
    
    print("=" * 50)
    print("OPTIMIZED Parallel Training")
    print("=" * 50)
    print(f"Performance optimizations: {'ENABLED' if perf_config.enable_all_optimizations else 'DISABLED'}")
    print(f"  - queue_maxsize: {queue_maxsize}")
    print(f"  - interval_logging: {perf_config.interval_logging_sec}s")
    print(f"  - worker0_verbose_only: {perf_config.worker0_verbose_only}")
    print(f"  - use_packed_transitions: {perf_config.use_packed_transitions}")
    print(f"  - use_batch_replay_push: {perf_config.use_batch_replay_push}")
    print(f"Actors: {num_actors}")
    print()
    
    if args.dry_run:
        print("Dry run complete. No processes spawned.")
        return 0
    
    mp.set_start_method("spawn", force=True)
    
    experience_queue = Queue(maxsize=queue_maxsize)
    weight_queues = [Queue(maxsize=1) for _ in range(num_actors)]
    stop_event = Event()
    global_step_counter = Value('l', 0)
    
    # Shared counter for produced_transitions (workers increment after put)
    produced_transitions_counter = Value('l', 0)
    
    # Use optimized collector if enabled
    if perf_config.enable_all_optimizations:
        from rl.parallel_collector_optimized import collector_process_optimized as collector_fn
    else:
        from rl.parallel_collector_1 import collector_process as collector_fn
    
    actors: List[Process] = []
    start_time = time.time()
    
    for i in range(num_actors):
        if perf_config.enable_all_optimizations:
            p = Process(
                target=collector_fn,
                args=(i, config, experience_queue, weight_queues[i], stop_event, 
                      global_step_counter, produced_transitions_counter),
                daemon=True,
            )
        else:
            p = Process(
                target=collector_fn,
                args=(i, config, experience_queue, weight_queues[i], stop_event, global_step_counter),
                daemon=True,
            )
        p.start()
        actors.append(p)
        print(f"Started collector {i}")
    
    print("Starting optimized learner...")
    
    consumed_transitions = _run_learner_optimized(
        config=config,
        experience_queue=experience_queue,
        weight_queues=weight_queues,
        stop_event=stop_event,
        sync_every_updates=sync_every_updates,
        resume_path=args.resume,
        global_step_counter=global_step_counter,
        num_actors=num_actors,
        perf_config=perf_config,
    )
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    # Signal workers to stop
    stop_event.set()
    
    # Wait for workers
    for i, p in enumerate(actors):
        p.join(timeout=10.0)
        if p.is_alive():
            print(f"[WARN] Worker {i} did not terminate, forcing...")
            p.terminate()
    
    # Verify no-drop
    produced_transitions = produced_transitions_counter.value
    no_drop_passed = (produced_transitions == consumed_transitions)
    
    print("=" * 50)
    print("TRAINING COMPLETE - NO-DROP VERIFICATION")
    print("=" * 50)
    print(f"  produced_transitions: {produced_transitions}")
    print(f"  consumed_transitions: {consumed_transitions}")
    print(f"  No-drop: {'PASS' if no_drop_passed else 'FAIL'}")
    print()
    
    # Throughput metrics
    throughput = compute_throughput(consumed_transitions, wall_time)
    print(f"  decision_steps/sec: {throughput['decision_steps_per_sec']:.2f}")
    print(f"  wall_time: {wall_time:.2f}s")
    print()
    
    if not no_drop_passed:
        print("[ERROR] Transition drop detected! This is a critical bug.")
        return 1
    
    print("Done")
    return 0


def _run_learner_optimized(
    config: Dict[str, Any],
    experience_queue: Queue,
    weight_queues: List[Queue],
    stop_event: Event,
    sync_every_updates: int,
    resume_path: str = None,
    global_step_counter: Value = None,
    num_actors: int = 1,
    perf_config: PerformanceConfig = None,
) -> int:
    """
    Optimized learner with sentinel-based shutdown and transition counters.
    
    Returns consumed_transitions for no-drop verification.
    """
    from rl.agent import AgentConfig, DQNAgent
    
    if perf_config is None:
        perf_config = PerformanceConfig()
    
    agent_cfg = config.get("agent", {})
    parallel_cfg = config.get("parallel", {})
    train_cfg = config.get("train", {})
    sumo_cfg = config.get("env", {}).get("sumo", {})
    logging_cfg = config.get("logging", {})
    
    state_dim = sumo_cfg.get("state_dim", 12)
    tls_ids = sumo_cfg.get("tls_ids", [])
    num_tls = len(tls_ids) if tls_ids else 1
    
    action_table = sumo_cfg.get("action_table", [])
    if action_table:
        action_dim = len(action_table)
    else:
        action_splits = sumo_cfg.get("action_splits", [])
        cycle_opts = sumo_cfg.get("cycle_options_sec", [60])
        action_dim = max(1, len(action_splits) * len(cycle_opts) if action_splits else 1)
    
    hidden_dims = agent_cfg.get("hidden_dims", [192, 192])
    gamma = agent_cfg.get("gamma", 0.99)
    lr = agent_cfg.get("learning_rate", 0.0003)
    batch_size = agent_cfg.get("batch_size", 256)
    buffer_size = agent_cfg.get("replay_buffer_size", 200000)
    target_update_freq = agent_cfg.get("target_update_freq", 3000)
    clip_grad_norm = agent_cfg.get("clip_grad_norm", 10.0)
    seed = config.get("run", {}).get("seed", 42)
    use_huber_loss = agent_cfg.get("use_huber_loss", True)
    learning_starts = int(agent_cfg.get("learning_starts", 5000))
    train_freq = int(agent_cfg.get("train_freq", 4))
    max_update_time_ms = float(parallel_cfg.get("max_update_time_ms", 50.0))
    
    log_dir = str(logging_cfg.get("log_dir", "logs"))
    model_dir = str(logging_cfg.get("model_dir", "models/parallel"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    agent_config = AgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        gamma=gamma,
        learning_rate=lr,
        batch_size=batch_size,
        replay_buffer_size=buffer_size,
        target_update_freq=target_update_freq,
        clip_grad_norm=clip_grad_norm,
        seed=seed,
        use_huber_loss=use_huber_loss,
    )
    
    device = torch.device("cpu")
    agent = DQNAgent(agent_config, device=device)
    agent.to_train_mode()
    
    # Instrumentation
    timing = TimingBreakdown()
    counters = TransitionCounters()
    interval_logger = IntervalLogger(
        interval_sec=perf_config.interval_logging_sec if perf_config.enable_all_optimizations else 0,
        enabled=True
    )
    
    learner_updates = 0
    agent_transitions_total = 0
    global_env_steps_total = 0
    pending_transitions = 0
    learning_started = False
    
    checkpoint_interval_sec = float(parallel_cfg.get("checkpoint_interval_sec", 300.0))
    last_checkpoint_time = time.time()
    recent_losses = []
    
    # Resume handling
    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Loading: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        agent.online_net.load_state_dict(checkpoint["online_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        learner_updates = checkpoint.get("learner_updates", 0)
        agent_transitions_total = checkpoint.get("agent_transitions_total", 0)
        global_env_steps_total = checkpoint.get("global_env_steps_total", 0)
        if global_step_counter is not None:
            resumed_steps = checkpoint.get("global_decision_steps", global_env_steps_total)
            with global_step_counter.get_lock():
                global_step_counter.value = resumed_steps
        if agent_transitions_total >= learning_starts:
            learning_started = True
        print(f"[Resume] learner_updates={learner_updates}")
    
    # Sentinel tracking
    sentinels_received = 0
    
    # Calculate target episodes for completion detection
    curriculum_cfg = config.get("curriculum", {})
    if curriculum_cfg.get("enabled", False) and curriculum_cfg.get("phases"):
        total_target_episodes = sum(p.get("episodes", 0) for p in curriculum_cfg["phases"])
    else:
        total_target_episodes = int(train_cfg.get("episodes", 100))
    
    print(f"Learner config: learning_starts={learning_starts}, train_freq={train_freq}")
    print(f"Waiting for {num_actors} workers (sentinels required: {num_actors})")
    
    episode_to_phase: Dict[int, int] = {}
    
    try:
        while not stop_event.is_set():
            # Receive chunk with timeout
            timing.start("queue_get")
            try:
                chunk_data = experience_queue.get(timeout=0.5)
            except Empty:
                chunk_data = None
            timing.stop("queue_get")
            
            if chunk_data is None:
                # Check if all workers sent sentinels
                if sentinels_received >= num_actors:
                    print(f"[Learner] All {num_actors} sentinels received. Stopping.")
                    break
                continue
            
            # Check for sentinel
            if is_sentinel(chunk_data):
                worker_id = chunk_data.get("worker_id", -1)
                sentinels_received += 1
                print(f"[Learner] Sentinel from worker {worker_id} ({sentinels_received}/{num_actors})")
                continue
            
            # Process chunk
            timing.start("chunk_process")
            
            if isinstance(chunk_data, dict):
                transitions = chunk_data.get("transitions", [])
                chunk_global_steps = chunk_data.get("global_steps", 0)
                chunk_phase_idx = chunk_data.get("phase_idx", -1)
                chunk_episode_uid = chunk_data.get("episode_uid", -1)
                chunk_count = chunk_data.get("count", len(transitions))
                
                if chunk_episode_uid >= 0 and chunk_phase_idx >= 0:
                    episode_to_phase[chunk_episode_uid] = chunk_phase_idx
            else:
                transitions = chunk_data
                chunk_global_steps = len(chunk_data) // num_tls
                chunk_count = len(transitions)
            
            # Update consumed counters AFTER successful get
            counters.record_consumed(chunk_count)
            
            # Add to replay buffer
            timing.start("replay_add")
            
            if perf_config.is_enabled("use_batch_replay_push") and len(transitions) > 0:
                # Batch push
                try:
                    n = len(transitions)
                    states_arr = np.array([t[0] for t in transitions], dtype=np.float32)
                    actions_arr = np.array([t[1] for t in transitions], dtype=np.int64)
                    rewards_arr = np.array([t[2] for t in transitions], dtype=np.float32)
                    next_states_arr = np.array([t[3] for t in transitions], dtype=np.float32)
                    dones_arr = np.array([float(t[4]) for t in transitions], dtype=np.float32)
                    gammas_arr = np.ones(n, dtype=np.float32) * gamma
                    uids_arr = np.array([t[5] if len(t) > 5 else -1 for t in transitions], dtype=np.int64)
                    
                    agent.replay_buffer.push_batch(
                        states=states_arr,
                        actions=actions_arr,
                        rewards=rewards_arr,
                        next_states=next_states_arr,
                        dones=dones_arr,
                        gammas=gammas_arr,
                        episode_uids=uids_arr,
                    )
                except Exception as e:
                    print(f"[WARN] push_batch failed, falling back to push(): {e}")
                    for transition in transitions:
                        if len(transition) == 6:
                            s, a, r, ns, d, ep_uid = transition
                        else:
                            s, a, r, ns, d = transition
                            ep_uid = -1
                        agent.replay_buffer.push(
                            state=np.asarray(s, dtype=np.float32),
                            action=int(a),
                            reward=float(r),
                            next_state=np.asarray(ns, dtype=np.float32),
                            done=bool(d),
                            episode_uid=int(ep_uid),
                        )
            else:
                # Standard push
                for transition in transitions:
                    if len(transition) == 6:
                        s, a, r, ns, d, ep_uid = transition
                    else:
                        s, a, r, ns, d = transition
                        ep_uid = -1
                    agent.replay_buffer.push(
                        state=np.asarray(s, dtype=np.float32),
                        action=int(a),
                        reward=float(r),
                        next_state=np.asarray(ns, dtype=np.float32),
                        done=bool(d),
                        episode_uid=int(ep_uid),
                    )
            
            timing.stop("replay_add")
            timing.stop("chunk_process")
            
            agent_transitions_total += chunk_count
            global_env_steps_total += chunk_global_steps
            pending_transitions += chunk_count
            
            # Learning
            if not learning_started:
                if agent_transitions_total >= learning_starts:
                    learning_started = True
                    print(f"[Learner] Learning started at {agent_transitions_total} transitions")
                continue
            
            timing.start("learner_update")
            update_start = time.perf_counter()
            
            while pending_transitions >= train_freq:
                elapsed_ms = (time.perf_counter() - update_start) * 1000
                if elapsed_ms > max_update_time_ms:
                    break
                
                metrics = agent.update()
                if metrics is not None:
                    learner_updates += 1
                    loss_value = metrics.get('loss', 0.0) if isinstance(metrics, dict) else float(metrics)
                    recent_losses.append(loss_value)
                pending_transitions -= train_freq
                
                if learner_updates % sync_every_updates == 0:
                    _broadcast_weights(agent, weight_queues)
            
            timing.stop("learner_update")
            
            # Interval logging
            if interval_logger.should_log() and learner_updates > 0:
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                print(f"[Learner] Updates: {learner_updates} | Trans: {agent_transitions_total} | "
                      f"Consumed: {counters.consumed_transitions} | Loss: {avg_loss:.4f}")
                recent_losses.clear()
            
            # Checkpoint
            now = time.time()
            if now - last_checkpoint_time >= checkpoint_interval_sec and learner_updates > 0:
                current_global_steps = global_step_counter.value if global_step_counter else global_env_steps_total
                ckpt_path = os.path.join(model_dir, f"parallel_opt_ckpt_step{learner_updates}.pt")
                agent.save_checkpoint(ckpt_path, {
                    "learner_updates": learner_updates,
                    "agent_transitions_total": agent_transitions_total,
                    "global_env_steps_total": global_env_steps_total,
                    "global_decision_steps": current_global_steps,
                    "consumed_transitions": counters.consumed_transitions,
                })
                print(f"[Checkpoint] Saved: {ckpt_path}")
                last_checkpoint_time = now
                
    except KeyboardInterrupt:
        print("[Learner] Interrupted by user")
    except Exception as e:
        print(f"[Learner] Fatal error: {e}")
        stop_event.set()  # Failure-safe: trigger shutdown
        raise
    finally:
        # Save final model
        final_global_steps = global_step_counter.value if global_step_counter else global_env_steps_total
        final_path = os.path.join(model_dir, f"parallel_opt_final_step{learner_updates}.pt")
        agent.save_checkpoint(final_path, {
            "learner_updates": learner_updates,
            "agent_transitions_total": agent_transitions_total,
            "global_env_steps_total": global_env_steps_total,
            "global_decision_steps": final_global_steps,
            "consumed_transitions": counters.consumed_transitions,
        })
        print(f"[Learner] Saved: {final_path}")
        print(timing.summary_str())
    
    return counters.consumed_transitions


def _broadcast_weights(agent, weight_queues: List[Queue]) -> None:
    from rl.agent import DQNAgent
    weights = {k: v.cpu().clone() for k, v in agent.online_net.state_dict().items()}
    
    for wq in weight_queues:
        while True:
            try:
                wq.get_nowait()
            except Empty:
                break
        try:
            wq.put_nowait(weights)
        except Full:
            pass


if __name__ == "__main__":
    sys.exit(main())
